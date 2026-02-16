from pyspark.ml.feature import (
    StandardScaler, MinMaxScaler, Imputer, StringIndexer, 
    OneHotEncoder, VectorAssembler, PCA
)
from pyspark.sql.functions import col, when, isnan, isnull, count
from pyspark.sql.types import NumericType
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def get_data_summary(self, df):
        """Get summary statistics and data quality metrics"""
        try:
            # Get basic info
            total_rows = df.count()
            
            # Analyze each column
            column_stats = []
            for column in df.columns:
                # Count nulls
                null_count = df.filter(
                    col(column).isNull() | 
                    isnan(column) | 
                    (col(column) == '')
                ).count()
                
                # Get distinct values
                distinct_count = df.select(column).distinct().count()
                
                # Get data type
                data_type = str(df.schema[column].dataType)
                
                # Get basic stats if numeric
                if isinstance(df.schema[column].dataType, NumericType):
                    stats = df.select(column).summary().collect()
                    stats_dict = {
                        row['summary']: float(row[column]) 
                        for row in stats 
                        if row[column] is not None
                    }
                else:
                    stats_dict = {}
                    # Get top 5 most frequent values for categorical
                    if distinct_count < 1000:
                        top_values = df.groupBy(column).count() \
                            .orderBy('count', ascending=False) \
                            .limit(5).collect()
                        stats_dict['top_values'] = [
                            {'value': row[column], 'count': row['count']} 
                            for row in top_values
                        ]
                
                column_stats.append({
                    'name': column,
                    'data_type': data_type,
                    'null_count': null_count,
                    'null_percentage': (null_count / total_rows) * 100,
                    'distinct_count': distinct_count,
                    'stats': stats_dict
                })
            
            return {
                'total_rows': total_rows,
                'total_columns': len(df.columns),
                'column_stats': column_stats
            }
        except Exception as e:
            logger.error(f"Error in get_data_summary: {str(e)}")
            raise
    
    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values in the dataset"""
        try:
            # Separate numeric and categorical columns
            numeric_cols = [
                f.name for f in df.schema.fields 
                if isinstance(f.dataType, NumericType)
            ]
            categorical_cols = [
                f.name for f in df.schema.fields 
                if not isinstance(f.dataType, NumericType)
            ]
            
            # Handle numeric columns
            if numeric_cols:
                imputer = Imputer(
                    inputCols=numeric_cols,
                    outputCols=numeric_cols,
                    strategy=strategy
                )
                df = imputer.fit(df).transform(df)
            
            # Handle categorical columns
            for col_name in categorical_cols:
                # Replace nulls with mode
                mode = df.groupBy(col_name).count() \
                    .orderBy('count', ascending=False) \
                    .first()[col_name]
                df = df.fillna(mode, subset=[col_name])
            
            return df
        except Exception as e:
            logger.error(f"Error in handle_missing_values: {str(e)}")
            raise
    
    def scale_features(self, df, numeric_cols, method='standard'):
        """Scale numeric features"""
        try:
            # Assemble features into vector
            assembler = VectorAssembler(
                inputCols=numeric_cols,
                outputCol='features_vector'
            )
            df = assembler.transform(df)
            
            # Scale features
            if method == 'standard':
                scaler = StandardScaler(
                    inputCol='features_vector',
                    outputCol='scaled_features',
                    withStd=True,
                    withMean=True
                )
            elif method == 'minmax':
                scaler = MinMaxScaler(
                    inputCol='features_vector',
                    outputCol='scaled_features'
                )
            else:
                raise ValueError(f"Unsupported scaling method: {method}")
            
            # Fit and transform
            df = scaler.fit(df).transform(df)
            return df
        except Exception as e:
            logger.error(f"Error in scale_features: {str(e)}")
            raise
    
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables"""
        try:
            # String indexing
            indexed = df
            indexed_cols = []
            for col_name in categorical_cols:
                indexer = StringIndexer(
                    inputCol=col_name,
                    outputCol=f"{col_name}_indexed",
                    handleInvalid="keep"
                )
                indexed = indexer.fit(indexed).transform(indexed)
                indexed_cols.append(f"{col_name}_indexed")
            
            # One-hot encoding
            encoder = OneHotEncoder(
                inputCols=indexed_cols,
                outputCols=[f"{col}_encoded" for col in categorical_cols]
            )
            encoded = encoder.fit(indexed).transform(indexed)
            
            return encoded
        except Exception as e:
            logger.error(f"Error in encode_categorical: {str(e)}")
            raise
    
    def reduce_dimensionality(self, df, feature_cols, n_components=2):
        """Perform dimensionality reduction using PCA"""
        try:
            # Assemble features
            assembler = VectorAssembler(
                inputCols=feature_cols,
                outputCol='features_vector'
            )
            assembled = assembler.transform(df)
            
            # Apply PCA
            pca = PCA(
                k=n_components,
                inputCol='features_vector',
                outputCol='pca_features'
            )
            pca_model = pca.fit(assembled)
            transformed = pca_model.transform(assembled)
            
            # Get explained variance ratio
            explained_variance = pca_model.explainedVariance.toArray()
            
            return transformed, explained_variance
        except Exception as e:
            logger.error(f"Error in reduce_dimensionality: {str(e)}")
            raise
    
    def detect_outliers(self, df, numeric_cols, method='zscore', threshold=3):
        """Detect outliers in numeric columns"""
        try:
            outliers = {}
            for col_name in numeric_cols:
                if method == 'zscore':
                    # Calculate z-score
                    stats = df.select(col_name).summary().collect()
                    mean = float(stats[1][col_name])
                    std = float(stats[2][col_name])
                    
                    # Mark outliers
                    outlier_df = df.withColumn(
                        f"{col_name}_outlier",
                        abs((col(col_name) - mean) / std) > threshold
                    )
                    outlier_count = outlier_df.filter(
                        col(f"{col_name}_outlier") == True
                    ).count()
                    
                    outliers[col_name] = {
                        'count': outlier_count,
                        'percentage': (outlier_count / df.count()) * 100
                    }
            
            return outliers
        except Exception as e:
            logger.error(f"Error in detect_outliers: {str(e)}")
            raise 