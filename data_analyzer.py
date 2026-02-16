from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import count, isnan, when, col
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.types import IntegerType, DoubleType, StringType, BooleanType
import json
import numpy as np
import pandas as pd
import logging
import traceback
import os
import uuid
from config import Config
# ModelTrainer is imported dynamically within methods to avoid circular dependencies
# and ensure it's loaded after its potential error handling in app.py.

logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, spark_manager, file_content=None, file_extension=None, spark_models_dir=None):
        self.spark_manager = spark_manager
        self.file_content = file_content
        self.file_extension = file_extension
        self.df = None
        self.spark_models_dir = spark_models_dir
        if file_content is not None and file_extension is not None:
            self.load_data()
    
    def load_data(self):
        """Load data directly from content into Spark DataFrame"""
        if self.file_content is not None and self.file_extension is not None:
            self.df = self.spark_manager.read_file_content(self.file_content, self.file_extension)
    
    def set_dataframe(self, df):
        """Set an existing Spark DataFrame"""
        self.df = df
    
    def _convert_to_serializable(self, obj):
        """Convert objects to JSON serializable format."""
        try:
            if hasattr(obj, 'toPandas'):  # Spark DataFrame
                return json.loads(obj.toPandas().to_json(orient='records'))
            if isinstance(obj, pd.DataFrame):
                return json.loads(obj.to_json(orient='records'))
            if isinstance(obj, pd.Series):
                return obj.tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.generic, np.int64, np.int32, np.float64, np.float32)):
                return obj.item()
            if isinstance(obj, dict):
                return {self._convert_to_serializable(k): self._convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [self._convert_to_serializable(item) for item in obj]
            if hasattr(obj, '__dict__'):  # Fallback for custom objects
                return self._convert_to_serializable(obj.__dict__)
            return obj
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return str(obj)

    def initial_analysis(self):
        """Perform initial analysis of the data"""
        pandas_df = self.df.toPandas()
        
        columns = pandas_df.columns.tolist()
        row_count = len(pandas_df)
        
        column_analysis = []
        for column in columns:
            column_info = {
                'name': column,
                'type': str(pandas_df[column].dtype),
                'null_count': int(pandas_df[column].isnull().sum()),
                'distinct_count': int(pandas_df[column].nunique()),
                'sample_values': pandas_df[column].dropna().head(5).tolist()
            }
            
            if np.issubdtype(pandas_df[column].dtype, np.number):
                column_info.update({
                    'min': float(pandas_df[column].min()),
                    'max': float(pandas_df[column].max()),
                    'mean': float(pandas_df[column].mean()),
                    'std': float(pandas_df[column].std())
                })
            
            column_analysis.append(column_info)
        
        result = {
            'row_count': row_count,
            'column_count': len(columns),
            'columns': column_analysis,
            'correlation_matrix': self._convert_to_serializable(
                pandas_df.select_dtypes(include=[np.number]).corr()
            ) if len(pandas_df.select_dtypes(include=[np.number]).columns) > 1 else None
        }
        
        return self._convert_to_serializable(result)
    
    def _prepare_ml_dataframe(self, df_in, feature_col_names, target_col_name=None, 
                              features_vec_col="features_raw", label_col="label"): # Renamed features_vec_col to features_raw
        """
        Prepares a DataFrame for ML by indexing string features/target and assembling features.
        Returns a tuple: (ml_ready_df, valid_assembler_inputs, model_input_schema_info, preprocessing_pipeline)
        
        Args:
            df_in (Spark DataFrame): The input DataFrame.
            feature_col_names (list): List of feature column names.
            target_col_name (str, optional): Name of the target column. Defaults to None.
            features_vec_col (str): Name for the assembled features vector column.
            label_col (str): Name for the target/label column in the prepared DataFrame.

        Returns:
            tuple: A tuple containing:
                - ml_ready_df (Spark DataFrame): The DataFrame with features assembled and labels prepared.
                - valid_assembler_inputs (list): List of column names actually used by VectorAssembler.
                - model_input_schema_info (list): List of dicts describing original features and target.
                - preprocessing_pipeline_for_trainer (Pipeline): The preprocessing pipeline to be fitted by ModelTrainer.
        """
        current_df = df_in
        
        assembler_input_feature_names = []
        string_indexer_stages = [] # Only for StringIndexers
        model_input_schema_info = [] # To store schema for the prediction endpoint

        # Process feature columns for indexing and schema info
        for col_name in feature_col_names:
            if col_name not in current_df.columns:
                raise ValueError(f"Feature column '{col_name}' not found in DataFrame.")
            
            spark_col_type = dict(current_df.dtypes)[col_name]

            if spark_col_type == 'string' or spark_col_type == 'boolean': # Handle boolean as categorical too
                indexed_name = col_name + "_indexed"
                string_indexer_stages.append(StringIndexer(inputCol=col_name, outputCol=indexed_name, handleInvalid="keep"))
                assembler_input_feature_names.append(indexed_name)
                model_input_schema_info.append({'name': col_name, 'original_type': spark_col_type, 'is_target': False})
            else: # Numeric
                if spark_col_type in ['int', 'long', 'float']:
                    current_df = current_df.withColumn(col_name, current_df[col_name].cast(DoubleType()))
                assembler_input_feature_names.append(col_name)
                model_input_schema_info.append({'name': col_name, 'original_type': spark_col_type, 'is_target': False})
        
        # Process target column for indexing or renaming/casting
        final_label_col_name_in_df = None
        if target_col_name:
            if target_col_name not in current_df.columns:
                raise ValueError(f"Target column '{target_col_name}' not found in DataFrame.")

            final_label_col_name_in_df = label_col
            target_col_spark_type = dict(current_df.dtypes)[target_col_name]
            model_input_schema_info.append({'name': target_col_name, 'original_type': target_col_spark_type, 'is_target': True})

            if target_col_spark_type == 'string' or target_col_spark_type == 'boolean':
                string_indexer_stages.append(StringIndexer(inputCol=target_col_name, outputCol=label_col, handleInvalid="error"))
            else:
                if target_col_name != label_col:
                    current_df = current_df.withColumnRenamed(target_col_name, label_col)
                current_df = current_df.withColumn(label_col, current_df[label_col].cast(DoubleType()))
            
        # Create and fit the StringIndexer pipeline
        if string_indexer_stages:
            string_indexer_pipeline = Pipeline(stages=string_indexer_stages)
            fitted_string_indexer_pipeline = string_indexer_pipeline.fit(current_df)
            current_df = fitted_string_indexer_pipeline.transform(current_df)
        
        # Filter out rows with nulls in the label column *after* indexing/renaming
        if final_label_col_name_in_df:
            logger.info(f"Filtering out rows where the label column '{final_label_col_name_in_df}' is null.")
            before_filter_count = current_df.count()
            current_df = current_df.na.drop(subset=[final_label_col_name_in_df])
            after_filter_count = current_df.count()
            logger.info(f"Rows before label null filter: {before_filter_count}, after: {after_filter_count}. Removed: {before_filter_count - after_filter_count}")
            if after_filter_count == 0:
                raise ValueError("All rows were removed after filtering for null labels. Check your target column or data quality.")

        valid_assembler_inputs = []
        seen_cols = set()
        for col_name in assembler_input_feature_names:
            if col_name in current_df.columns and col_name not in seen_cols:
                valid_assembler_inputs.append(col_name)
                seen_cols.add(col_name)
            elif col_name not in current_df.columns:
                logger.warning(f"Column '{col_name}' intended for assembler not found after pipeline. Skipping.")

        if not valid_assembler_inputs:
            raise ValueError("No valid feature columns available for VectorAssembler after processing.")

        # Define the remaining stages for the ModelTrainer's pipeline
        # This includes the VectorAssembler and StandardScaler
        assembler = VectorAssembler(inputCols=valid_assembler_inputs, outputCol=features_vec_col, handleInvalid="skip")
        scaler = StandardScaler(inputCol=features_vec_col, outputCol="features", withStd=True, withMean=True)
        
        # The preprocessing pipeline for the trainer will combine StringIndexers (if any),
        # the VectorAssembler, and the StandardScaler.
        preprocessing_pipeline_for_trainer = Pipeline(stages=string_indexer_stages + [assembler, scaler])

        # The prepared_df returned here is the one after label handling and null filtering.
        # The full pipeline (including assembler and scaler) will be fitted by ModelTrainer.
        return current_df, valid_assembler_inputs, model_input_schema_info, preprocessing_pipeline_for_trainer


    def suggest_analysis_type(self, target_feature=None):
        """Suggest appropriate analysis types based on data characteristics"""
        if not target_feature:
            return ['clustering', 'exploratory']
        
        col_type = str(self.df.schema[target_feature].dataType)
        distinct_count = self.df.select(target_feature).distinct().count()
        
        if 'int' in col_type.lower() or 'double' in col_type.lower() or 'float' in col_type.lower():
            if distinct_count < 10 and distinct_count > 1:
                return ['classification', 'regression']
            elif distinct_count >= 10:
                return ['regression']
            else:
                return ['exploratory']
        elif 'string' in col_type.lower() or 'boolean' in col_type.lower(): # Added boolean
            return ['classification']
        
        return ['exploratory']
    
    def analyze(self, analysis_type, selected_features=None, target_feature=None):
        """Perform the specified type of analysis"""
        # Import ModelTrainer dynamically to avoid circular dependencies and ensure it's loaded properly
        from models.model_training import ModelTrainer 
        
        if not selected_features:
            selected_features = self.df.columns
        
        if analysis_type == 'exploratory':
            return self._exploratory_analysis(selected_features)
        elif analysis_type == 'regression':
            return self._regression_analysis(selected_features, target_feature)
        elif analysis_type == 'classification':
            return self._classification_analysis(selected_features, target_feature)
        elif analysis_type == 'clustering':
            return self._clustering_analysis(selected_features)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

    def _exploratory_analysis(self, selected_features):
        """Perform exploratory data analysis"""
        pandas_df = self.df.select(selected_features).toPandas()
        results = {}
        for feature in selected_features:
            if np.issubdtype(pandas_df[feature].dtype, np.number):
                stats = pandas_df[feature].describe()
                results[feature] = {
                    'type': 'numeric',
                    'stats': {
                        'min': float(stats['min']),
                        'max': float(stats['max']),
                        'mean': float(stats['mean']),
                        'std': float(stats['std']),
                        'quartiles': {
                            '25%': float(stats['25%']),
                            '50%': float(stats['50%']),
                            '75%': float(stats['75%'])
                        }
                    },
                    'histogram': self._create_histogram(pandas_df[feature])
                }
            else:
                value_counts = pandas_df[feature].value_counts().head(10)
                results[feature] = {
                    'type': 'categorical',
                    'top_values': dict(zip(value_counts.index.astype(str), value_counts.values.astype(int)))
                }
        return self._convert_to_serializable(results)

    def _create_histogram(self, series):
        """Create histogram data for numeric columns"""
        hist, bin_edges = np.histogram(series.dropna(), bins='auto')
        return {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }

    def _regression_analysis(self, selected_features, target_feature):
        """Perform regression analysis, save the model, and return metrics & model info."""
        from models.model_training import ModelTrainer # Local import

        try:
            logger.info(f"Starting regression analysis. Features: {selected_features}, Target: {target_feature}")
            original_feature_cols = [f for f in selected_features if f != target_feature]

            if not original_feature_cols:
                raise ValueError("No feature columns selected for regression.")
            if not target_feature or target_feature not in self.df.columns:
                raise ValueError(f"Target column '{target_feature}' not found or not specified.")

            # Prepare ML DataFrame using the unified method
            prepared_df_for_ml, assembler_inputs_names, model_input_schema, preprocessing_pipeline_for_trainer = \
                self._prepare_ml_dataframe(self.df, original_feature_cols, target_feature)

            trainer = ModelTrainer(self.spark_manager.get_spark_session(), spark_models_dir=self.spark_models_dir)
            
            # Pass the prepared_df, model_input_schema and preprocessing_pipeline_for_trainer to the trainer
            training_results = trainer.train_regressor(
                prepared_df_for_ml,
                preprocessing_pipeline_for_trainer, # Pass the full pipeline to ModelTrainer
                model_input_schema,
                assembler_inputs_names,
                algorithm='linear_regression' # Default or configurable
            )

            # Extract necessary info from training_results
            rmse = training_results['rmse']
            model_uri_path = training_results['model_uri_path']
            
            # Get top 10 predictions for display
            prediction_sample = training_results.get('prediction_sample', [])
            prediction_sample_headers = training_results.get('prediction_sample_headers', [])

            return self._convert_to_serializable({
                'rmse': rmse,
                'model_uri_path': model_uri_path,
                'model_input_schema': model_input_schema,
                'type': 'regression_model_training',
                'prediction_sample': prediction_sample,
                'prediction_sample_headers': prediction_sample_headers,
                'target_column_original': target_feature, # Add original target column name
                'pipeline_label_column': 'label' # Add the internal label column name
            })
        except Exception as e:
            logger.error(f"Error in regression analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _classification_analysis(self, selected_features, target_feature):
        """Perform classification analysis, save the model, and return metrics & model info."""
        from models.model_training import ModelTrainer # Local import

        try:
            logger.info(f"Starting classification analysis. Features: {selected_features}, Target: {target_feature}")
            original_feature_cols = [f for f in selected_features if f != target_feature]

            if not original_feature_cols:
                raise ValueError("No feature columns selected for classification.")
            if not target_feature or target_feature not in self.df.columns:
                raise ValueError(f"Target column '{target_feature}' not found or not specified.")
            
            prepared_df_for_ml, assembler_inputs_names, model_input_schema, preprocessing_pipeline_for_trainer = \
                self._prepare_ml_dataframe(self.df, original_feature_cols, target_feature)

            trainer = ModelTrainer(self.spark_manager.get_spark_session(), spark_models_dir=self.spark_models_dir)
            
            training_results = trainer.train_classifier(
                prepared_df_for_ml,
                preprocessing_pipeline_for_trainer,
                model_input_schema,
                assembler_inputs_names,
                algorithm='random_forest' # Default or configurable
            )

            accuracy = training_results['accuracy']
            model_uri_path = training_results['model_uri_path']

            # Get top 10 predictions for display
            prediction_sample = training_results.get('prediction_sample', [])
            prediction_sample_headers = training_results.get('prediction_sample_headers', [])

            # Add correct_predictions and total_test_samples if available (from ModelTrainer)
            correct_predictions = training_results.get('correct_predictions')
            total_test_samples = training_results.get('total_test_samples')
            message = training_results.get('message') # From ModelTrainer if test set was empty

            return self._convert_to_serializable({
                'accuracy': accuracy,
                'model_uri_path': model_uri_path,
                'model_input_schema': model_input_schema,
                'type': 'classification_model_training',
                'prediction_sample': prediction_sample,
                'prediction_sample_headers': prediction_sample_headers,
                'target_column_original': target_feature,
                'pipeline_label_column': 'label',
                'correct_predictions': correct_predictions,
                'total_test_samples': total_test_samples,
                'message': message
            })
        except Exception as e:
            logger.error(f"Error in classification analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _clustering_analysis(self, selected_features):
        """Perform clustering analysis, save the model, and return metrics & model info."""
        from models.model_training import ModelTrainer # Local import

        try:
            logger.info(f"Starting clustering analysis. Features: {selected_features}")
            k = 5 # Default k for KMeans, can be made configurable

            if not selected_features:
                raise ValueError("No feature columns selected for clustering.")

            # For clustering, there is no target column, so target_col_name=None
            prepared_df_for_ml, assembler_inputs_names, model_input_schema, preprocessing_pipeline_for_trainer = \
                self._prepare_ml_dataframe(self.df, selected_features, target_col_name=None)
            
            trainer = ModelTrainer(self.spark_manager.get_spark_session(), spark_models_dir=self.spark_models_dir)
            
            training_results = trainer.train_clustering_model(
                prepared_df_for_ml,
                preprocessing_pipeline_for_trainer,
                model_input_schema,
                assembler_inputs_names,
                k=k # Pass k to the trainer
            )

            cluster_sizes = training_results['cluster_sizes']
            cluster_centers = training_results['cluster_centers']
            model_uri_path = training_results['model_uri_path']

            # Remap cluster centers to original feature names for better interpretation
            mapped_centers_list_of_lists = []
            if cluster_centers:
                # Get the actual feature names that went into the VectorAssembler for the model
                # These are the ones present in 'assembler_inputs_names_for_model' from ModelTrainer
                actual_assembler_inputs = training_results['assembler_inputs_names_for_model']
                
                for center_vector in cluster_centers:
                    current_center_ordered_by_selected_features = []
                    for original_feature_name in selected_features:
                        # Determine the name of the feature after StringIndexing if it was a string
                        original_type = None
                        for info in model_input_schema:
                            if info['name'] == original_feature_name and not info['is_target']:
                                original_type = info['original_type']
                                break
                        
                        name_in_assembler_inputs = original_feature_name
                        if original_type and ('string' in original_type.lower() or 'boolean' in original_type.lower()):
                            name_in_assembler_inputs = original_feature_name + "_indexed"
                    
                        try:
                            idx_in_vector = actual_assembler_inputs.index(name_in_assembler_inputs)
                            current_center_ordered_by_selected_features.append(center_vector[idx_in_vector])
                        except ValueError:
                            logger.warning(f"Could not map cluster center value for '{original_feature_name}' (expected as '{name_in_assembler_inputs}' in assembler inputs). Appending None.")
                            current_center_ordered_by_selected_features.append(None)
                    mapped_centers_list_of_lists.append(current_center_ordered_by_selected_features)

            # Get top 10 predictions for display for clustering
            prediction_sample = training_results.get('prediction_sample', [])
            prediction_sample_headers = training_results.get('prediction_sample_headers', [])

            metrics = {
                'k': k,
                'cluster_sizes': {int(k_val): int(v_val) for k_val, v_val in cluster_sizes.items()},
                'cluster_centers': mapped_centers_list_of_lists, 
                'cluster_center_feature_names': selected_features,
                'model_uri_path': model_uri_path,
                'model_input_schema': model_input_schema,
                'type': 'clustering_model_training',
                'prediction_sample': prediction_sample,
                'prediction_sample_headers': prediction_sample_headers
            }
            
            return self._convert_to_serializable(metrics)
        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
            logger.error(traceback.format_exc())
            raise
