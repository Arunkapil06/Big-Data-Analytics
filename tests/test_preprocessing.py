import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import pandas as pd
import numpy as np
from preprocessing import DataPreprocessor

@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing"""
    return SparkSession.builder \
        .appName("TestDataPreprocessor") \
        .master("local[*]") \
        .getOrCreate()

@pytest.fixture(scope="session")
def sample_df(spark):
    """Create a sample DataFrame for testing"""
    # Create schema
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("salary", FloatType(), True),
        StructField("department", StringType(), True)
    ])
    
    # Create data
    data = [
        (1, "John", 30, 50000.0, "IT"),
        (2, "Alice", None, 60000.0, "HR"),
        (3, "Bob", 35, None, "IT"),
        (4, None, 40, 55000.0, None),
        (5, "Eve", 32, 65000.0, "HR")
    ]
    
    return spark.createDataFrame(data, schema)

@pytest.fixture
def preprocessor(spark):
    """Create DataPreprocessor instance"""
    return DataPreprocessor(spark)

def test_get_data_summary(preprocessor, sample_df):
    """Test data summary generation"""
    summary = preprocessor.get_data_summary(sample_df)
    
    assert summary['total_rows'] == 5
    assert summary['total_columns'] == 5
    
    # Check column stats
    col_stats = {stat['name']: stat for stat in summary['column_stats']}
    assert col_stats['age']['null_count'] == 1
    assert col_stats['salary']['null_count'] == 1
    assert col_stats['department']['distinct_count'] == 2

def test_handle_missing_values(preprocessor, sample_df):
    """Test missing value handling"""
    processed_df = preprocessor.handle_missing_values(sample_df)
    
    # Check if no nulls remain
    for column in processed_df.columns:
        assert processed_df.filter(
            processed_df[column].isNull()
        ).count() == 0

def test_scale_features(preprocessor, sample_df):
    """Test feature scaling"""
    numeric_cols = ['age', 'salary']
    scaled_df = preprocessor.scale_features(
        sample_df, 
        numeric_cols,
        method='standard'
    )
    
    assert 'scaled_features' in scaled_df.columns

def test_encode_categorical(preprocessor, sample_df):
    """Test categorical encoding"""
    categorical_cols = ['department']
    encoded_df = preprocessor.encode_categorical(sample_df, categorical_cols)
    
    assert 'department_encoded' in encoded_df.columns

def test_reduce_dimensionality(preprocessor, sample_df):
    """Test PCA dimensionality reduction"""
    numeric_cols = ['age', 'salary']
    transformed_df, variance = preprocessor.reduce_dimensionality(
        sample_df,
        numeric_cols,
        n_components=1
    )
    
    assert 'pca_features' in transformed_df.columns
    assert len(variance) == 1

def test_detect_outliers(preprocessor, sample_df):
    """Test outlier detection"""
    numeric_cols = ['age', 'salary']
    outliers = preprocessor.detect_outliers(
        sample_df,
        numeric_cols,
        method='zscore',
        threshold=3
    )
    
    assert 'age' in outliers
    assert 'salary' in outliers
    assert 'count' in outliers['age']
    assert 'percentage' in outliers['salary'] 