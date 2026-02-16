from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml import Pipeline, PipelineModel # Corrected typo here
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col # Corrected typo here
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType
import logging
import uuid
import os
import shutil
import traceback
import pandas as pd # Import pandas for toPandas()

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, spark, spark_models_dir=None):
        self.spark = spark
        self.spark_models_dir = spark_models_dir
        if self.spark_models_dir:
            os.makedirs(self.spark_models_dir, exist_ok=True)

    def _save_model(self, model):
        """Saves the trained model to a unique directory within spark_models_dir."""
        if not self.spark_models_dir:
            raise ValueError("spark_models_dir is not set in ModelTrainer.")
        
        model_name = f"spark_model_{uuid.uuid4()}"
        model_path = os.path.join(self.spark_models_dir, model_name)
        
        # Ensure the directory is clean if it already exists (e.g., from a failed previous save)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        # Convert path to URI for Spark, especially important for Windows
        if os.name == 'nt': # For Windows
            abs_model_path = os.path.abspath(model_path)
            uri_model_path = "file:///" + abs_model_path.replace("\\", "/")
        else: # For Unix-like systems
            uri_model_path = "file://" + os.path.abspath(model_path)

        model.save(uri_model_path)
        logger.info(f"Model saved to: {uri_model_path}")
        return uri_model_path

    def train_classifier(self, prepared_df, preprocessing_pipeline_for_trainer, model_input_schema, assembler_inputs_names_for_model, algorithm='random_forest'):
        """Train a classification model"""
        try:
            logger.info(f"Training {algorithm} classifier.")
            
            if algorithm == 'random_forest':
                classifier = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=10)
            elif algorithm == 'logistic_regression':
                classifier = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.1)
            elif algorithm == 'gbt_classifier':
                classifier = GBTClassifier(featuresCol="features", labelCol="label", maxIter=50, maxDepth=5)
            else:
                raise ValueError(f"Unsupported classification algorithm: {algorithm}")
            
            # Combine the preprocessing pipeline with the classifier
            full_pipeline = Pipeline(stages=preprocessing_pipeline_for_trainer.getStages() + [classifier])
            
            logger.info("Fitting classification pipeline...")
            model = full_pipeline.fit(prepared_df)
            
            predictions = model.transform(prepared_df)

            # Evaluate the model
            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
            accuracy = evaluator.evaluate(predictions)
            logger.info(f"Classification Model Accuracy: {accuracy}")

            # Get correct predictions count for display
            correct_predictions_df = predictions.filter(col("label") == col("prediction"))
            correct_predictions = correct_predictions_df.count()
            total_test_samples = predictions.count()

            message = "Model trained successfully."
            if total_test_samples == 0:
                accuracy = "N/A - Test set was empty"
                correct_predictions = 0
                message = "Warning: No samples in the test set after preprocessing. Accuracy cannot be calculated."

            model_uri_path = self._save_model(model)
            
            # Get top 10 predictions for display
            # Extract original feature names from model_input_schema where is_target is False
            original_feature_names = [
                info['name'] for info in model_input_schema if not info.get('is_target', False)
            ]
            
            # Also get the original target column name
            original_target_column_name = None
            for info in model_input_schema:
                if info.get('is_target', False):
                    original_target_column_name = info['name']
                    break

            # Create a list of columns to select, including original features, actual label, and prediction
            cols_to_select = [col(f) for f in original_feature_names]
            if original_target_column_name:
                cols_to_select.append(col("label").alias(f"{original_target_column_name}_indexed_label")) # Use indexed label
            cols_to_select.append(col("prediction"))

            # Select and convert to Pandas, then to list of dicts
            prediction_sample_pd = predictions.select(*cols_to_select).limit(10).toPandas()
            prediction_sample = prediction_sample_pd.to_dict(orient='records')
            
            # Define headers for the prediction sample table
            prediction_sample_headers = original_feature_names + [f"{original_target_column_name}_indexed_label", 'prediction'] if original_target_column_name else original_feature_names + ['prediction']

            return {
                'model': model,
                'accuracy': accuracy,
                'model_uri_path': model_uri_path,
                'model_input_schema': model_input_schema,
                'assembler_inputs_names_for_model': assembler_inputs_names_for_model,
                'correct_predictions': correct_predictions,
                'total_test_samples': total_test_samples,
                'message': message,
                'prediction_sample': prediction_sample,
                'prediction_sample_headers': prediction_sample_headers
            }
        except Exception as e:
            logger.error(f"Error in training classifier: {str(e)}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise

    def train_regressor(self, prepared_df, preprocessing_pipeline_for_trainer, model_input_schema, assembler_inputs_names_for_model, algorithm='linear_regression'):
        """Train a regression model"""
        try:
            logger.info(f"Training {algorithm} regressor.")

            if algorithm == 'linear_regression':
                regressor = LinearRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.1)
            elif algorithm == 'random_forest_regressor':
                regressor = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=100, maxDepth=10)
            elif algorithm == 'gbt_regressor':
                regressor = GBTRegressor(featuresCol="features", labelCol="label", maxIter=50, maxDepth=5)
            else:
                raise ValueError(f"Unsupported regression algorithm: {algorithm}")
            
            full_pipeline = Pipeline(stages=preprocessing_pipeline_for_trainer.getStages() + [regressor])
            
            logger.info("Fitting regression pipeline...")
            model = full_pipeline.fit(prepared_df)
            
            predictions = model.transform(prepared_df)
            evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
            rmse = evaluator.evaluate(predictions)
            logger.info(f"Regression Model RMSE: {rmse}")

            model_uri_path = self._save_model(model)

            # Get top 10 predictions for display
            original_feature_names = [
                info['name'] for info in model_input_schema if not info.get('is_target', False)
            ]
            original_target_column_name = None
            for info in model_input_schema:
                if info.get('is_target', False):
                    original_target_column_name = info['name']
                    break

            cols_to_select = [col(f) for f in original_feature_names]
            if original_target_column_name:
                cols_to_select.append(col("label").alias(original_target_column_name)) # Original label
            cols_to_select.append(col("prediction"))

            prediction_sample_pd = predictions.select(*cols_to_select).limit(10).toPandas()
            prediction_sample = prediction_sample_pd.to_dict(orient='records')
            
            prediction_sample_headers = original_feature_names + [original_target_column_name, 'prediction'] if original_target_column_name else original_feature_names + ['prediction']

            return {
                'model': model,
                'rmse': rmse,
                'model_uri_path': model_uri_path,
                'model_input_schema': model_input_schema,
                'assembler_inputs_names_for_model': assembler_inputs_names_for_model,
                'prediction_sample': prediction_sample,
                'prediction_sample_headers': prediction_sample_headers
            }
        except Exception as e:
            logger.error(f"Error in training regressor: {str(e)}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise

    def train_clustering_model(self, prepared_df, preprocessing_pipeline_for_trainer, model_input_schema, assembler_inputs_names_for_model, k=5):
        """Train a KMeans clustering model"""
        try:
            logger.info(f"Training KMeans clustering with k={k}.")

            kmeans = KMeans(featuresCol="features", k=k, seed=42)
            
            full_pipeline = Pipeline(stages=preprocessing_pipeline_for_trainer.getStages() + [kmeans])
            
            logger.info("Fitting clustering pipeline...")
            model = full_pipeline.fit(prepared_df)
            
            logger.info("Making predictions for evaluation...")
            predictions = model.transform(prepared_df)
            
            cluster_sizes = predictions.groupBy("prediction").count().collect()
            cluster_size_dict = {str(row["prediction"]): row["count"] for row in cluster_sizes}
            
            silhouette_score = 0.0 # Placeholder as silhouette score isn't directly available or straightforward for PipelineModel

            cluster_centers = None
            for stage in model.stages:
                if "KMeans" in stage.__class__.__name__:
                    cluster_centers = stage.clusterCenters()
                    break
            
            model_uri_path = self._save_model(model)

            # Get top 10 predictions for display
            # For clustering, 'prediction' is the cluster ID.
            # We want to show the original features and the assigned cluster ID.
            original_feature_names = [
                info['name'] for info in model_input_schema if not info.get('is_target', False)
            ]
            
            # Select original features and the 'prediction' (cluster ID) column
            cols_to_select = [col(f) for f in original_feature_names]
            cols_to_select.append(col("prediction").alias("Assigned_Cluster_ID"))
            
            prediction_sample_pd = predictions.select(*cols_to_select).limit(10).toPandas()
            prediction_sample = prediction_sample_pd.to_dict(orient='records')
            prediction_sample_headers = original_feature_names + ["Assigned_Cluster_ID"]


            return {
                'model': model,
                'cluster_sizes': cluster_size_dict,
                'silhouette_score': silhouette_score,
                'cluster_centers': cluster_centers,
                'model_uri_path': model_uri_path,
                'model_input_schema': model_input_schema,
                'assembler_inputs_names_for_model': assembler_inputs_names_for_model,
                'prediction_sample': prediction_sample,
                'prediction_sample_headers': prediction_sample_headers
            }
            
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise
