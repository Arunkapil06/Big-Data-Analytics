from pyspark.sql import SparkSession
from config import Config
import pandas as pd
import io
import logging
import json
import os
from subprocess import run, PIPE

class SparkManager:
    def __init__(self):
        # Set Spark logging level
        spark_logger = logging.getLogger('pyspark')
        spark_logger.setLevel(logging.WARNING)
        
        self.spark = SparkSession.builder \
            .appName(Config.SPARK_APP_NAME) \
            .master(Config.SPARK_MASTER) \
            .config('spark.driver.memory', '2g') \
            .config('spark.executor.memory', '2g') \
            .config('spark.driver.maxResultSize', '1g') \
            .config('spark.python.worker.memory', '1g') \
            .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1') \
            .config('spark.mongodb.input.uri', Config.MONGO_URI) \
            .config('spark.mongodb.output.uri', Config.MONGO_URI) \
            .config('spark.ui.showConsoleProgress', 'false') \
            .config('spark.sql.repl.eagerEval.enabled', 'false') \
            .config('spark.hadoop.fs.defaultFS', Config.HADOOP_FS_DEFAULT) \
            .config('spark.hadoop.dfs.client.use.datanode.hostname', 'true') \
            .config('spark.sql.execution.arrow.pyspark.enabled', 'true') \
            .config('spark.sql.execution.arrow.maxRecordsPerBatch', '10000') \
            .config('spark.sql.shuffle.partitions', '10') \
            .getOrCreate()
        
        # Set Spark internal logging to ERROR only
        self.spark.sparkContext.setLogLevel("ERROR")
    
    def upload_to_hdfs(self, local_path, hdfs_path):
        """Upload a file to HDFS"""
        try:
            # Check if HDFS path exists
            check_cmd = f"hadoop fs -test -e {hdfs_path}"
            result = run(check_cmd.split(), stderr=PIPE)
            
            if result.returncode == 0:
                # Path exists, remove it
                run(f"hadoop fs -rm -r {hdfs_path}".split(), check=True)
            
            # Upload file
            run(f"hadoop fs -put {local_path} {hdfs_path}".split(), check=True)
            logging.info(f"Successfully uploaded {local_path} to HDFS at {hdfs_path}")
            return True
        except Exception as e:
            logging.error(f"Error uploading to HDFS: {str(e)}")
            raise

    def read_from_hdfs(self, hdfs_path, file_format='csv'):
        """Read a file from HDFS"""
        try:
            if file_format == 'csv':
                return self.spark.read.csv(hdfs_path, header=True, inferSchema=True)
            elif file_format == 'json':
                return self.spark.read.json(hdfs_path)
            elif file_format == 'parquet':
                return self.spark.read.parquet(hdfs_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        except Exception as e:
            logging.error(f"Error reading from HDFS: {str(e)}")
            raise

    def read_file_content(self, file_content, file_extension):
        """Read file content directly without saving"""
        try:
            if file_extension == 'csv':
                # Read CSV content with chunking
                pandas_df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
                # Convert to Spark DataFrame in smaller chunks
                chunk_size = 10000
                if len(pandas_df) > chunk_size:
                    spark_df = None
                    for i in range(0, len(pandas_df), chunk_size):
                        chunk = pandas_df.iloc[i:i + chunk_size]
                        chunk_spark = self.spark.createDataFrame(chunk)
                        if spark_df is None:
                            spark_df = chunk_spark
                        else:
                            spark_df = spark_df.union(chunk_spark)
                    return spark_df
                else:
                    return self.spark.createDataFrame(pandas_df)
            elif file_extension == 'json':
                # Read JSON content with chunking
                pandas_df = pd.read_json(io.StringIO(file_content.decode('utf-8')))
                # Convert to Spark DataFrame in smaller chunks
                chunk_size = 10000
                if len(pandas_df) > chunk_size:
                    spark_df = None
                    for i in range(0, len(pandas_df), chunk_size):
                        chunk = pandas_df.iloc[i:i + chunk_size]
                        chunk_spark = self.spark.createDataFrame(chunk)
                        if spark_df is None:
                            spark_df = chunk_spark
                        else:
                            spark_df = spark_df.union(chunk_spark)
                    return spark_df
                else:
                    return self.spark.createDataFrame(pandas_df)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logging.error(f"Error reading file content: {str(e)}")
            raise

    def write_to_mongodb(self, dataframe, collection_name):
        """Write DataFrame directly to MongoDB"""
        try:
            # Convert Spark DataFrame to Pandas
            pandas_df = dataframe.toPandas()
            
            # Convert to records format (list of dictionaries)
            records = json.loads(pandas_df.to_json(orient='records'))
            
            # Get MongoDB client from config
            from pymongo import MongoClient
            client = MongoClient(Config.MONGO_URI)
            db = client[Config.DB_NAME]
            collection = db[collection_name]
            
            # Insert records
            if records:
                collection.insert_many(records)
            
            logging.info(f"Successfully wrote {len(records)} records to MongoDB collection {collection_name}")
        except Exception as e:
            logging.error(f"Error writing to MongoDB: {str(e)}")
            raise

    def read_from_mongodb(self, collection_name):
        """Read data from MongoDB into Spark DataFrame"""
        try:
            # Increase sampleSize for more robust schema inference
            return self.spark.read \
                .format("mongo") \
                .option("uri", f"{Config.MONGO_URI}.{collection_name}") \
                .option("sampleSize", 50000) \
                .option("inferSchema", "true") \
                .load()
        except Exception as e:
            logging.error(f"Error reading from MongoDB: {str(e)}")
            raise
    
    def get_spark_session(self):
        """Return the SparkSession instance"""
        return self.spark
    
    def stop(self):
        """Stop the SparkSession"""
        if self.spark:
            self.spark.stop() 