import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for numeric type checking

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, static_folder):
        self.static_folder = static_folder
        self.plots_folder = os.path.join(static_folder, 'plots')
        self._ensure_folders()

    def _ensure_folders(self):
        """Ensure necessary folders exist"""
        if not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)

    def _save_plot(self, fig, filename):
        """Save plot to static folder and return URL path"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            full_filename = f"{filename}_{timestamp}.html"
            filepath = os.path.join(self.plots_folder, full_filename)
            fig.write_html(filepath)
            return f"/static/plots/{full_filename}"
        except Exception as e:
            logger.error(f"Error saving plot: {str(e)}")
            raise

    def create_feature_importance_plot(self, importance_dict):
        """Create feature importance visualization"""
        try:
            df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
            df = df.sort_values('Importance', ascending=True)

            fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                         title='Feature Importance',
                         labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'})

            return self._save_plot(fig, 'feature_importance')
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            raise

    def create_regression_results_plot(self, actual, predicted):
        """Create scatter plot of actual vs predicted values"""
        try:
            df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})

            fig = px.scatter(df, x='Actual', y='Predicted',
                             title='Actual vs Predicted Values',
                             labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'})

            min_val = min(df['Actual'].min(), df['Predicted'].min())
            max_val = max(df['Actual'].max(), df['Predicted'].max())

            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                     mode='lines', name='Perfect Prediction',
                                     line=dict(dash='dash')))
            return self._save_plot(fig, 'regression_results')
        except Exception as e:
            logger.error(f"Error creating regression results plot: {str(e)}")
            raise

    def create_clustering_plot(self, df, features, cluster_labels, centers=None):
        """Create clustering visualization (works best with 2-3 features)"""
        try:
            plot_df = df.copy()
            plot_df['Cluster'] = cluster_labels

            if len(features) == 2:
                fig = px.scatter(plot_df, x=features[0], y=features[1],
                                 color='Cluster',
                                 title='Cluster Assignments',
                                 labels={features[0]: features[0], features[1]: features[1]})

                if centers is not None:
                    center_df = pd.DataFrame(centers, columns=features)
                    fig.add_trace(go.Scatter(x=center_df[features[0]],
                                             y=center_df[features[1]],
                                             mode='markers',
                                             marker=dict(symbol='x', size=15, color='black'),
                                             name='Cluster Centers'))

            elif len(features) == 3:
                fig = px.scatter_3d(plot_df, x=features[0], y=features[1], z=features[2],
                                    color='Cluster', title='Cluster Assignments (3D)')

                if centers is not None:
                    center_df = pd.DataFrame(centers, columns=features)
                    fig.add_trace(go.Scatter3d(x=center_df[features[0]],
                                               y=center_df[features[1]],
                                               z=center_df[features[2]],
                                               mode='markers',
                                               marker=dict(symbol='x', size=8, color='black'),
                                               name='Cluster Centers'))
            else:
                raise ValueError("Clustering visualization supports only 2-3 features")

            return self._save_plot(fig, 'clustering_results')
        except Exception as e:
            logger.error(f"Error creating clustering plot: {str(e)}")
            raise

    def create_correlation_matrix(self, df, numeric_columns):
        """Create correlation matrix heatmap"""
        try:
            corr_matrix = df[numeric_columns].corr()

            fig = px.imshow(corr_matrix,
                            labels=dict(color="Correlation"),
                            title="Feature Correlation Matrix")

            return self._save_plot(fig, 'correlation_matrix')
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {str(e)}")
            raise

    def create_distribution_plot(self, df, column):
        """Create distribution plot for a numeric column (histogram)"""
        try:
            if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"Column '{column}' is not numeric or does not exist for histogram.")
            fig = px.histogram(df, x=column,
                               title=f'Distribution of {column}',
                               labels={column: column, 'count': 'Frequency'})

            return self._save_plot(fig, f'distribution_{column}')
        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            raise

    def create_boxplot(self, df, numeric_columns):
        """Create boxplot for numeric features"""
        try:
            # Filter to ensure only numeric columns are used
            valid_numeric_columns = [col for col in numeric_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            if not valid_numeric_columns:
                raise ValueError("No valid numeric columns provided for boxplot.")

            if len(valid_numeric_columns) == 1:
                fig = px.box(df, y=valid_numeric_columns[0],
                             title=f"Boxplot of {valid_numeric_columns[0]}",
                             labels={valid_numeric_columns[0]: valid_numeric_columns[0]})
            else:
                df_melted = df[valid_numeric_columns].melt(var_name="Feature", value_name="Value")
                fig = px.box(df_melted, x="Feature", y="Value",
                             title="Boxplot of Numeric Features",
                             labels={"Feature": "Feature", "Value": "Value"})
            return self._save_plot(fig, 'boxplot')
        except Exception as e:
            logger.error(f"Error creating boxplot: {str(e)}")
            raise

    def create_pairplot(self, df):
        """Create pairplot for numeric features"""
        try:
            # Use only numeric columns for pairplot
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.empty:
                raise ValueError("No numeric columns available for pairplot.")

            fig = px.scatter_matrix(numeric_df,
                                    dimensions=numeric_df.columns[:5],  # limit to first 5 for readability
                                    title="Pairplot of Numeric Features",
                                    labels={col: col for col in numeric_df.columns})
            fig.update_traces(diagonal_visible=False)
            return self._save_plot(fig, 'pairplot')
        except Exception as e:
            logger.error(f"Error creating pairplot: {str(e)}")
            raise

    def create_scatterplot(self, df, x_column, y_column):
        """Create a scatter plot between two numeric columns"""
        try:
            if x_column not in df.columns or y_column not in df.columns:
                raise ValueError("One or both specified columns for scatter plot do not exist.")
            if not pd.api.types.is_numeric_dtype(df[x_column]) or not pd.api.types.is_numeric_dtype(df[y_column]):
                raise ValueError("Both columns for scatter plot must be numeric.")

            fig = px.scatter(df, x=x_column, y=y_column,
                             title=f'Scatter Plot of {x_column} vs {y_column}',
                             labels={x_column: x_column, y_column: y_column})
            return self._save_plot(fig, f'scatterplot_{x_column}_{y_column}')
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            raise
