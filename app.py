import os
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
from config import Config
from database import MongoDB
from spark_manager import SparkManager
from data_analyzer import DataAnalyzer
# Attempt to import ModelTrainer with error handling
try:
    from models.model_training import ModelTrainer
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.critical(f"Failed to import ModelTrainer: {e}. Please ensure 'models' directory is a Python package (has an __init__.py) and 'model_training.py' exists and is error-free.")
    ModelTrainer = None

from visualization import Visualizer
import uuid
import json
import traceback
import logging
import datetime
import pandas as pd
import numpy as np
import webbrowser
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType
import threading
import atexit

LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'app.log')
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
SPARK_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spark_models')

os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SPARK_MODELS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

# Redirect stdout and stderr to logger
class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

logging.getLogger('py4j').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('pyspark').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

def clear_app_log_file():
    try:
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, 'w') as f:
                f.truncate(0)
            logger.info(f"Cleared log file: {LOG_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error clearing log file {LOG_FILE_PATH}: {str(e)}")

clear_app_log_file()

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling Spark DataFrames and NumPy types."""
    def default(self, obj):
        try:
            if hasattr(obj, 'toPandas'):
                # Convert Spark DataFrame to Pandas DataFrame, then to JSON records
                return json.loads(obj.toPandas().to_json(orient='records'))
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            if isinstance(obj, pd.Series):
                return obj.tolist()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.generic, np.int64, np.int32, np.float64, np.float32)):
                # Convert NumPy scalar types to Python native types
                return obj.item()
            if isinstance(obj, dict):
                # Recursively apply default for dictionary values
                return {k: self.default(v) for k, v in obj.items()}
            if isinstance(obj, list):
                # Recursively apply default for list elements
                return [self.default(v) for v in obj]
            return super().default(obj)
        except Exception as e:
            logger.error(f"JSON serialization error: {str(e)}")
            return str(obj)

app = Flask(__name__)
# Configure CORS to allow all origins for development
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
app.config.from_object(Config)

# Define allowed file extensions
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'json', 'xlsx'}
app.json_encoder = JSONEncoder

# Explicitly set maximum content length
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
# Enable chunked encoding (set to None if you want to allow arbitrary large files, handled by Spark)
app.config['MAX_CONTENT_LENGTH'] = None

# Initialize folders
try:
    Config.init_folders()
    logger.info("Folders initialized successfully")
except Exception as e:
    logger.error(f"Error initializing folders: {str(e)}")
    raise

# Initialize MongoDB connection
try:
    mongo = MongoDB()
    # Test MongoDB connection
    mongo.test_connection()
    logger.info("MongoDB connection established successfully")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    raise

# Initialize Spark
try:
    spark_manager = SparkManager()
    logger.info("Spark initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Spark: {str(e)}")
    raise

# Initialize visualizer
visualizer = Visualizer(os.path.join(app.root_path, 'static'))

@app.route('/')
def index():
    """Renders the main index.html page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads and performs initial data analysis."""
    try:
        logger.info("Starting file upload process")
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in app.config['ALLOWED_EXTENSIONS']:
            logger.warning(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Read file content
        file_content = file.read()
        
        # Initialize data analyzer with content and spark_models_dir
        logger.info("Initializing data analyzer")
        analyzer = DataAnalyzer(spark_manager, file_content, file_extension, spark_models_dir=SPARK_MODELS_DIR)
        
        # Get initial analysis
        logger.info("Performing initial analysis")
        analysis_results = analyzer.initial_analysis()
        logger.info("Initial analysis completed")
        
        # Generate unique collection name for this dataset
        collection_name = f"dataset_{str(uuid.uuid4()).replace('-', '_')}"
        
        # Store data in MongoDB
        logger.info("Storing data in MongoDB")
        spark_manager.write_to_mongodb(analyzer.df, collection_name)
        
        # Store metadata in MongoDB
        logger.info("Storing metadata in MongoDB")
        metadata = {
            'original_filename': file.filename,
            'collection_name': collection_name,
            'status': 'uploaded',
            'analysis': analysis_results,
            'created_at': datetime.datetime.utcnow()
        }
        file_id = mongo.insert_file_metadata(metadata)
        logger.info(f"Metadata stored with ID: {file_id}")
        
        response_data = {
            'message': 'File processed successfully',
            'file_id': str(file_id),
            'collection_name': collection_name,
            'analysis': analysis_results
        }
        
        logger.info("Upload process completed successfully")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/analyze/<file_id>', methods=['POST'])
def analyze_data(file_id):
    """Performs various data analyses based on the requested type."""
    try:
        # Check if ModelTrainer was successfully imported
        if ModelTrainer is None:
            logger.error("ModelTrainer class is not available due to an import error during startup.")
            return jsonify({'error': 'Model training functionality is unavailable due to an internal error. Check server logs for details.'}), 500

        logger.info(f"Starting analysis for file_id: {file_id}")
        analysis_type = request.json.get('analysis_type')
        selected_features = request.json.get('selected_features', [])
        target_feature = request.json.get('target_feature', None)
        
        # IMPORTANT FIX: Ensure target_feature is not in selected_features for ML tasks
        if analysis_type in ['regression', 'classification', 'clustering'] and target_feature:
            selected_features = [f for f in selected_features if f != target_feature]
            logger.info(f"Adjusted selected_features for ML task: {selected_features}")

        logger.info(f"Analysis parameters - type: {analysis_type}, features: {selected_features}, target: {target_feature}")
        
        # Get file metadata from MongoDB
        metadata = mongo.get_file_metadata(file_id)
        if not metadata:
            logger.warning(f"File not found: {file_id}")
            return jsonify({'error': 'File not found'}), 404
        
        # Get data from MongoDB collection
        collection_name = metadata.get('collection_name')
        if not collection_name:
            return jsonify({'error': 'Collection name not found in metadata'}), 400
            
        # Initialize analyzer with data from MongoDB, passing spark_models_dir
        logger.info(f"Reading data from collection: {collection_name}")
        df = spark_manager.read_from_mongodb(collection_name)
        analyzer = DataAnalyzer(spark_manager, spark_models_dir=SPARK_MODELS_DIR) # Pass spark_models_dir
        analyzer.set_dataframe(df)
        
        # Perform analysis
        logger.info("Starting analysis")
        results = analyzer.analyze(
            analysis_type=analysis_type,
            selected_features=selected_features,
            target_feature=target_feature
        )
        logger.info("Analysis completed")
        
        # Update metadata with results
        logger.info("Updating metadata with results")
        mongo.update_file_metadata(file_id, {
            'status': 'analyzed',
            'analysis_type': analysis_type,
            'results': results
        })
        
        logger.info("Analysis process completed successfully")
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.route('/favicon.ico')
def favicon():
    """Serves the favicon.ico file."""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST'])
def predict_from_model():
    """Handles prediction requests using a loaded model."""
    try:
        # Check if ModelTrainer was successfully imported
        if ModelTrainer is None:
            logger.error("ModelTrainer class is not available due to an import error during startup.")
            return jsonify({'error': 'Model prediction functionality is unavailable due to an internal error. Check server logs for details.'}), 500

        request_data = request.get_json()
        model_uri_path = request_data.get('model_uri_path')
        user_input_features = request_data.get('features')
        model_input_schema_desc = request_data.get('model_input_schema')

        if not all([model_uri_path, user_input_features, model_input_schema_desc]):
            return jsonify({'error': 'Missing model_uri_path, features, or model_input_schema in request'}), 400

        logger.info(f"Received prediction request for model: {model_uri_path}")
        logger.info(f"Input features: {user_input_features}")

        # Prepend 'file:///' to the model_uri_path to ensure Spark uses local file system
        # This is crucial for avoiding HDFS connection issues when loading from local paths
        local_model_path = model_uri_path
        if not local_model_path.startswith("file:///"):
            # Ensure forward slashes for URI, especially for Windows paths
            local_model_path = "file:///" + local_model_path.replace("\\", "/") 

        # Load the PipelineModel
        try:
            loaded_model = PipelineModel.load(local_model_path)
            logger.info(f"Successfully loaded model from {local_model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {local_model_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f"Error loading model: {str(e)}"}), 500

        # Prepare Spark DataFrame from user input features based on model_input_schema_desc
        data_row_values = []
        struct_fields = []
        
        # Filter schema for input features only (not target)
        schema_for_input_df = [info for info in model_input_schema_desc if not info.get('is_target', False)]

        for col_info in schema_for_input_df:
            col_name = col_info.get('name')
            original_type_str = col_info.get('original_type')
            
            if not col_name or not original_type_str:
                logger.error(f"Invalid column info in model_input_schema: {col_info}")
                return jsonify({'error': 'Invalid model_input_schema provided'}), 400

            raw_value = user_input_features.get(col_name)
            if raw_value is None:
                logger.error(f"Missing feature in input: {col_name}")
                return jsonify({'error': f'Missing required feature: {col_name}'}), 400

            casted_value = None
            spark_type = None
            try:
                if original_type_str.lower() in ['integer', 'int', 'long']:
                    casted_value = int(raw_value)
                    spark_type = IntegerType()
                elif original_type_str.lower() in ['double', 'float', 'decimal']:
                    casted_value = float(raw_value)
                    spark_type = DoubleType()
                elif original_type_str.lower() == 'string':
                    casted_value = str(raw_value)
                    spark_type = StringType()
                elif original_type_str.lower() == 'boolean':
                    if isinstance(raw_value, str):
                        casted_value = raw_value.lower() == 'true'
                    else:
                        casted_value = bool(raw_value)
                    spark_type = BooleanType()
                else:
                    logger.warning(f"Unknown type '{original_type_str}' for feature '{col_name}'. Attempting to cast to string.")
                    casted_value = str(raw_value)
                    spark_type = StringType()
            except ValueError as ve:
                logger.error(f"Type casting error for feature '{col_name}' (expected {original_type_str}, got '{raw_value}'): {str(ve)}")
                return jsonify({'error': f"Invalid value for feature '{col_name}'. Expected type {original_type_str}."}), 400
            
            data_row_values.append(casted_value)
            struct_fields.append(StructField(col_name, spark_type, True))

        if not data_row_values or not struct_fields:
             return jsonify({'error': 'No valid features processed from input schema for prediction.'}), 400

        data_for_df = [tuple(data_row_values)]
        schema_for_df = StructType(struct_fields)
        
        spark_session = spark_manager.get_spark_session()
        input_spark_df = spark_session.createDataFrame(data_for_df, schema=schema_for_df)
        logger.info("Input DataFrame for prediction schema:")
        input_spark_df.printSchema()
        logger.info("Input DataFrame for prediction data:")
        input_spark_df.show(truncate=False)

        # Make prediction
        predictions_df = loaded_model.transform(input_spark_df)
        logger.info("Predictions DataFrame schema:")
        predictions_df.printSchema()
        logger.info("Predictions DataFrame data:")
        predictions_df.show(truncate=False)

        if "prediction" not in predictions_df.columns:
            logger.error("'prediction' column not found in predictions output.")
            return jsonify({'error': "Failed to get 'prediction' column from model output."}), 500
            
        prediction_value = predictions_df.select("prediction").first()[0]
        
        logger.info(f"Prediction successful. Value: {prediction_value}")
        return jsonify({'prediction': prediction_value})

    except Exception as e:
        logger.error(f"Error in /predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/visualization/<file_id>/<plot_type>', methods=['POST'])
def generate_visualization(file_id, plot_type):
    """Generates various visualizations based on the requested plot type."""
    try:
        logger.info(f"Generating {plot_type} visualization for file_id: {file_id}")
        
        metadata = mongo.get_file_metadata(file_id)
        if not metadata:
            logger.warning(f"File metadata not found for ID: {file_id}")
            return jsonify({'error': 'File not found'}), 404

        collection_name = metadata.get('collection_name')
        if not collection_name:
            return jsonify({'error': 'Collection name not found'}), 400

        df = spark_manager.read_from_mongodb(collection_name)
        df_sample = df.limit(1000).toPandas() # Use a sample for visualization to avoid memory issues

        selected_features = request.json.get('selected_features', [])

        numeric_cols = df_sample.select_dtypes(include='number').columns.tolist()

        plot_path = None

        if plot_type == 'histogram':
            if not selected_features:
                return jsonify({'error': 'Histogram requires at least one selected feature.'}), 400
            if not selected_features[0] in df_sample.columns or not pd.api.types.is_numeric_dtype(df_sample[selected_features[0]]):
                 return jsonify({'error': f"Selected feature '{selected_features[0]}' is not numeric for histogram."}), 400
            plot_path = visualizer.create_distribution_plot(df_sample, selected_features[0])

        elif plot_type == 'boxplot':
            if not selected_features:
                return jsonify({'error': 'Box Plot requires at least one selected feature.'}), 400
            numeric_selected_features = [f for f in selected_features if f in df_sample.columns and pd.api.types.is_numeric_dtype(df_sample[f])]
            if not numeric_selected_features:
                return jsonify({'error': 'No numeric features selected for boxplot.'}), 400
            plot_path = visualizer.create_boxplot(df_sample, numeric_selected_features) 

        elif plot_type == 'scatterplot':
            if len(selected_features) < 2:
                return jsonify({'error': 'Scatter Plot requires at least two selected features.'}), 400
            if not all(f in df_sample.columns and pd.api.types.is_numeric_dtype(df_sample[f]) for f in selected_features[:2]):
                 return jsonify({'error': 'Both selected features for scatter plot must be numeric.'}), 400
            plot_path = visualizer.create_scatterplot(df_sample, selected_features[0], selected_features[1])

        elif plot_type == 'heatmap':
            if not numeric_cols:
                return jsonify({'error': 'No numeric columns available for heatmap'}), 400
            plot_path = visualizer.create_correlation_matrix(df_sample, numeric_cols)

        elif plot_type == 'pairplot':
            if not numeric_cols:
                return jsonify({'error': 'No numeric columns available for pairplot'}), 400
            plot_path = visualizer.create_pairplot(df_sample[numeric_cols])

        else:
            logger.warning(f"Unknown plot type requested: {plot_type}")
            return jsonify({'error': 'Unknown plot type'}), 400

        if not plot_path:
            logger.error("Failed to generate plot.")
            return jsonify({'error': 'Failed to generate plot'}), 500

        absolute_plot_path = os.path.join(app.root_path, plot_path.lstrip('/'))
        # Delete plot file after a short delay
        threading.Timer(2, lambda: delete_file_if_exists(absolute_plot_path)).start()
        logger.info(f"Scheduled deletion of {absolute_plot_path} in 2 seconds.")

        return jsonify({'plot_url': plot_path})

    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Visualization error: {str(e)}'}), 500

@app.route('/generate_report/<file_id>', methods=['POST'])
def generate_report(file_id):
    """Generates a comprehensive data analysis report in HTML and TXT formats."""
    try:
        # Check if ModelTrainer was successfully imported
        if ModelTrainer is None:
            logger.error("ModelTrainer class is not available due to an import error during startup.")
            return jsonify({'error': 'Report generation functionality is unavailable due to an internal error. Check server logs for details.'}), 500

        metadata = mongo.get_file_metadata(file_id)
        if not metadata:
            return jsonify({'error': 'File not found'}), 404

        analysis = metadata.get('analysis', {})
        collection_name = metadata.get('collection_name')
        if not collection_name:
            return jsonify({'error': 'Collection name not found'}), 400

        df = spark_manager.read_from_mongodb(collection_name)
        df_pd = df.toPandas() 

        ml_results = []
        # Instantiate ModelTrainer and DataAnalyzer here
        trainer = ModelTrainer(spark_manager.get_spark_session(), spark_models_dir=SPARK_MODELS_DIR)
        analyzer = DataAnalyzer(spark_manager, spark_models_dir=SPARK_MODELS_DIR)
        analyzer.set_dataframe(df) # Set the DataFrame for the analyzer

        # --- DATA SUMMARY ---
        total_rows = analysis.get('total_rows', len(df_pd))
        columns = [c['name'] for c in analysis.get('columns', [])] if analysis.get('columns') else list(df_pd.columns)
        
        # Determine numeric and categorical columns based on the full Spark DataFrame schema
        numeric_cols_spark = [f.name for f in df.schema.fields if isinstance(f.dataType, (IntegerType, DoubleType))]
        categorical_cols_spark = [f.name for f in df.schema.fields if isinstance(f.dataType, (StringType, BooleanType))]

        # Filter out _id if it's in numeric_cols_spark
        numeric_cols_spark = [col for col in numeric_cols_spark if col != '_id']
        categorical_cols_spark = [col for col in categorical_cols_spark if col != '_id']

        # --- FEATURE INFORMATION ---
        feature_info = []
        for col_name in numeric_cols_spark:
            col_min = round(df_pd[col_name].min(), 4) if not pd.isnull(df_pd[col_name].min()) else "N/A"
            col_max = round(df_pd[col_name].max(), 4) if not pd.isnull(df_pd[col_name].max()) else "N/A"
            col_mean = round(df_pd[col_name].mean(), 4) if not pd.isnull(df_pd[col_name].mean()) else "N/A"
            feature_info.append({'Feature': col_name, 'Min': col_min, 'Max': col_max, 'Mean': col_mean})

        # --- DATA TRANSFORMATION ---
        data_transformation = []
        # Simplified transformation reporting, as actual transformations are handled by _prepare_ml_dataframe
        if numeric_cols_spark:
            data_transformation.append({'Task': 'Numeric Feature Handling', 'Transformation': 'Casting to DoubleType, Vector Assembly, Standardization'})
        if categorical_cols_spark:
            data_transformation.append({'Task': 'Categorical Feature Handling', 'Transformation': 'String Indexing'})
        if numeric_cols_spark or categorical_cols_spark:
            data_transformation.append({'Task': 'Missing Value Handling', 'Transformation': 'Rows with null labels are dropped'})

        # --- MODEL TRAINING SUMMARY & PREDICTIONS ---
        # Regression
        reg_success = False
        reg_rmse = None
        
        # Attempt to find a suitable numeric target for regression
        reg_target_candidate = None
        # Prioritize columns that are numeric and have more than 10 distinct values
        potential_reg_targets = [f.name for f in df.schema.fields if isinstance(f.dataType, (IntegerType, DoubleType)) and df.select(f.name).distinct().count() > 10]
        if potential_reg_targets:
            reg_target_candidate = potential_reg_targets[0]
        elif numeric_cols_spark: # Fallback to any numeric if no clear regression target
            reg_target_candidate = numeric_cols_spark[0]

        if reg_target_candidate and reg_target_candidate != '_id':
            reg_features = [col_name for col_name in numeric_cols_spark + categorical_cols_spark if col_name != reg_target_candidate and col_name != '_id']
            if reg_features:
                try:
                    # Prepare data for regression using DataAnalyzer's method
                    prepared_df_reg, assembler_inputs_reg, model_input_schema_reg, preprocessing_pipeline_reg = \
                        analyzer._prepare_ml_dataframe(df, reg_features, reg_target_candidate)

                    if prepared_df_reg.count() > 0:
                        reg_result = trainer.train_regressor(
                            prepared_df_reg,
                            preprocessing_pipeline_reg,
                            model_input_schema_reg,
                            assembler_inputs_reg,
                            algorithm='linear_regression'
                        )
                        reg_rmse = reg_result.get('rmse', 'N/A')
                        ml_results.append({
                            'Task': 'Regression (Linear Regression)',
                            'Target': reg_target_candidate,
                            'Features': ', '.join(reg_features),
                            'Result': f"RMSE: {reg_rmse:.4f}" if isinstance(reg_rmse, (int, float)) else reg_rmse,
                            'prediction_sample': reg_result.get('prediction_sample', []),
                            'prediction_sample_headers': reg_result.get('prediction_sample_headers', [])
                        })
                        reg_success = True
                    else:
                        ml_results.append({'Task': 'Regression', 'Target': reg_target_candidate, 'Features': ', '.join(reg_features), 'Result': "Skipped: Cleaned data is empty."})
                except Exception as e:
                    ml_results.append({'Task': 'Regression', 'Target': reg_target_candidate, 'Features': ', '.join(reg_features), 'Result': f"Failed: {str(e)}"})
            else:
                ml_results.append({'Task': 'Regression', 'Target': '-', 'Features': '-', 'Result': "Skipped: Not enough features for regression."})
        else:
            ml_results.append({'Task': 'Regression', 'Target': '-', 'Features': '-', 'Result': "Skipped: No suitable numeric target for regression."})


        # Classification
        clf_success = False
        clf_accuracy = None

        # Attempt to find a suitable classification target
        clf_target_candidate = None
        # Prioritize categorical columns with 2-10 distinct values
        potential_clf_targets = [f.name for f in df.schema.fields if isinstance(f.dataType, (StringType, BooleanType)) and df.select(f.name).distinct().count() > 1 and df.select(f.name).distinct().count() <= 10]
        if potential_clf_targets:
            clf_target_candidate = potential_clf_targets[0]
        else: # Fallback to numeric columns with 2-10 distinct values
            potential_clf_targets_numeric = [f.name for f in df.schema.fields if isinstance(f.dataType, (IntegerType, DoubleType)) and df.select(f.name).distinct().count() > 1 and df.select(f.name).distinct().count() <= 10]
            if potential_clf_targets_numeric:
                clf_target_candidate = potential_clf_targets_numeric[0]

        if clf_target_candidate and clf_target_candidate != '_id':
            clf_features = [col_name for col_name in numeric_cols_spark + categorical_cols_spark if col_name != clf_target_candidate and col_name != '_id']
            if clf_features:
                try:
                    # Prepare data for classification using DataAnalyzer's method
                    prepared_df_clf, assembler_inputs_clf, model_input_schema_clf, preprocessing_pipeline_clf = \
                        analyzer._prepare_ml_dataframe(df, clf_features, clf_target_candidate)

                    if prepared_df_clf.count() > 0:
                        clf_result = trainer.train_classifier(
                            prepared_df_clf,
                            preprocessing_pipeline_clf,
                            model_input_schema_clf,
                            assembler_inputs_clf,
                            algorithm='random_forest'
                        )
                        clf_accuracy = clf_result.get('accuracy', 'N/A')
                        ml_results.append({
                            'Task': 'Classification (Random Forest)',
                            'Target': clf_target_candidate,
                            'Features': ', '.join(clf_features),
                            'Result': f"Accuracy: {clf_accuracy * 100:.2f}%" if isinstance(clf_accuracy, (int, float)) else clf_accuracy,
                            'prediction_sample': clf_result.get('prediction_sample', []),
                            'prediction_sample_headers': clf_result.get('prediction_sample_headers', [])
                        })
                        clf_success = True
                    else:
                        ml_results.append({'Task': 'Classification', 'Target': clf_target_candidate, 'Features': ', '.join(clf_features), 'Result': "Skipped: Cleaned data is empty."})
                except Exception as e:
                    ml_results.append({'Task': 'Classification', 'Target': clf_target_candidate, 'Features': ', '.join(clf_features), 'Result': f"Failed: {str(e)}"})
            else:
                ml_results.append({'Task': 'Classification', 'Target': '-', 'Features': '-', 'Result': "Skipped: Not enough features for classification."})
        else:
            ml_results.append({'Task': 'Classification', 'Target': '-', 'Features': '-', 'Result': "Skipped: No suitable target for classification."})


        # Clustering
        cluster_success = False
        cluster_sizes = None

        if len(numeric_cols_spark) >= 2: # Clustering typically requires at least 2 numeric features
            cluster_features = [col_name for col_name in numeric_cols_spark if col_name != '_id']
            if cluster_features:
                try:
                    # Prepare data for clustering using DataAnalyzer's method (no target)
                    prepared_df_cluster, assembler_inputs_cluster, model_input_schema_cluster, preprocessing_pipeline_cluster = \
                        analyzer._prepare_ml_dataframe(df, cluster_features, target_col_name=None)

                    if prepared_df_cluster.count() > 0:
                        cluster_result = trainer.train_clustering_model(
                            prepared_df_cluster,
                            preprocessing_pipeline_cluster,
                            model_input_schema_cluster,
                            assembler_inputs_cluster,
                            k=3 # Default k
                        )
                        cluster_sizes = cluster_result.get('cluster_sizes', 'N/A')
                        ml_results.append({
                            'Task': 'Clustering (KMeans, k=3)',
                            'Target': '-',
                            'Features': ', '.join(cluster_features),
                            'Result': f"Cluster Sizes: {cluster_sizes}",
                            'prediction_sample': cluster_result.get('prediction_sample', []),
                            'prediction_sample_headers': cluster_result.get('prediction_sample_headers', [])
                        })
                        cluster_success = True
                    else:
                        ml_results.append({'Task': 'Clustering', 'Target': '-', 'Features': '-', 'Result': "Skipped: Cleaned data is empty."})
                except Exception as e:
                    ml_results.append({'Task': 'Clustering', 'Target': '-', 'Features': ', '.join(cluster_features), 'Result': f"Failed: {str(e)}"})
            else:
                ml_results.append({'Task': 'Clustering', 'Target': '-', 'Features': '-', 'Result': "Skipped: Not enough features for clustering."})
        else:
            ml_results.append({'Task': 'Clustering', 'Target': '-', 'Features': '-', 'Result': "Skipped: Not enough numeric columns for clustering."})


        key_findings = []
        if reg_success and reg_rmse is not None and isinstance(reg_rmse, (int, float)):
            key_findings.append(f"Regression RMSE: {reg_rmse:.4f}")
        if clf_success and clf_accuracy is not None and isinstance(clf_accuracy, (int, float)):
            key_findings.append(f"Classification Accuracy: {clf_accuracy * 100:.2f}%")
        if cluster_success and cluster_sizes is not None:
            key_findings.append(f"Cluster sizes: {cluster_sizes}")

        conclusions = []
        if reg_success or clf_success or cluster_success:
            conclusions.append("Models were successfully trained and evaluated. See above for details.")
        else:
            conclusions.append("No models could be successfully trained on this dataset.")

        # --- HTML Report Assembly ---
        def html_table(rows, headers=None):
            if not rows:
                return "<p>No data available.</p>"
            if not headers:
                headers = list(rows[0].keys())
            table = '<table class="data-table"><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>'
            for row in rows:
                table += '<tr>' + ''.join(f'<td>{row.get(h, "")}</td>' for h in headers) + '</tr>'
            table += '</tbody></table>'
            return table

        def html_list(items):
            if not items:
                return "<p>No data available.</p>"
            return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"

        # --- TXT Report Assembly ---
        def txt_table(rows, headers=None):
            if not rows:
                return "No data available.\n"
            if not headers:
                headers = list(rows[0].keys())
            
            processed_rows = []
            for row_data in rows:
                processed_row = []
                for h in headers:
                    value = row_data.get(h, "")
                    if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                        processed_row.append(", ".join([f"{x:.4f}" if isinstance(x, float) else str(x) for x in value]))
                    elif isinstance(value, (int, float)):
                        processed_row.append(f"{value:.4f}" if isinstance(value, float) else str(value))
                    else:
                        processed_row.append(str(value))
                processed_rows.append(processed_row)

            string_headers = [str(h) for h in headers]

            col_widths = [max(len(string_headers[i]), max(len(row[i]) for row in processed_rows)) for i in range(len(headers))]
            
            header_line = " | ".join(string_headers[i].ljust(col_widths[i]) for i in range(len(headers)))
            sep_line = "-+-".join('-' * w for w in col_widths)
            data_lines = []
            for row_str_values in processed_rows:
                data_lines.append(" | ".join(row_str_values[i].ljust(col_widths[i]) for i in range(len(headers))))
            return header_line + "\n" + sep_line + "\n" + "\n".join(data_lines) + "\n"

        def txt_list(items, title=None):
            if not items:
                return "No data available.\n"
            out = ""
            if title:
                out += title + "\n" + "-" * len(title) + "\n"
            for item in items:
                out += f"- {item}\n"
            return out

        # --- DARK MODE CSS ---
        report_css = """
        <style>
        .report-container { font-family: 'Segoe UI', Arial, sans-serif; background: #181a1b; color: #f1f1f1; padding: 32px; border-radius: 12px; max-width: 900px; margin: 0 auto; box-shadow: 0 2px 12px #000a;}
        .report-title { font-size: 2.2em; font-weight: bold; margin-bottom: 0.2em; color: #e0e6f0;}
        .section-title { font-size: 1.3em; font-weight: 600; margin-top: 2em; margin-bottom: 0.5em; color: #a8c7fa;}
        .data-table { border-collapse: collapse; width: 100%; margin-bottom: 1.5em; background: #23272a;}
        .data-table th, .data-table td { border: 1px solid #333; padding: 8px 12px; text-align: left;}
        .data-table th { background: #232c3d; color: #a8c7fa;}
        .data-table tr:nth-child(even) { background: #23272a;}
        .data-table tr:nth-child(odd) { background: #181a1b;}
        .summary-block { background: #232c3d; padding: 12px 18px; border-radius: 8px; margin-bottom: 1.5em;}
        .key-findings { background: #2d2d1f; border-left: 4px solid #ffe066; padding: 12px 18px; border-radius: 8px; margin-bottom: 1.5em;}
        .conclusions { background: #1e2d24; border-left: 4px solid #34c759; padding: 12px 18px; border-radius: 8px; margin-bottom: 1.5em;}
        </style>
        """

        html_report = f"""
        {report_css}
        <div class="report-container">
            <div class="report-title">Data Analysis Report</div>
            <div class="summary-block">
                <b>File:</b> {metadata.get('original_filename', f"{file_id}.csv")}<br>
                <b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <b>Total Records:</b> {total_rows}<br>
                <b>Columns:</b> {', '.join(columns)}
            </div>

            <div class="section-title">Feature Information (Numeric)</div>
            {html_table(feature_info, headers=["Feature", "Min", "Max", "Mean"])}

            <div class="section-title">Data Transformation Steps</div>
            {html_table(data_transformation, headers=["Task", "Transformation"])}

            <div class="section-title">Model Training Summary</div>
            {html_table(ml_results, headers=["Task", "Target", "Features", "Result"])}
            
            """
        # Add prediction samples to HTML report
        for result in ml_results:
            if 'prediction_sample' in result and result['prediction_sample']:
                html_report += f"""
                <div class="section-title">Top 10 Predictions for {result['Task']}</div>
                {html_table(result['prediction_sample'], headers=result['prediction_sample_headers'])}
                """
        
        html_report += f"""
            <div class="section-title">Key Findings</div>
            <div class="key-findings">{html_list(key_findings)}</div>
            <div class="section-title">Conclusions</div>
            <div class="conclusions">{html_list(conclusions)}</div>
        </div>
        """

        # --- TXT Report Assembly ---
        txt_report = ""
        txt_report += "DATA ANALYSIS REPORT\n"
        txt_report += "=" * 60 + "\n"
        txt_report += f"File: {metadata.get('original_filename', f'{file_id}.csv')}\n"
        txt_report += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        txt_report += f"Total Records: {total_rows}\n"
        txt_report += f"Columns: {', '.join(columns)}\n"
        txt_report += "=" * 60 + "\n\n"

        txt_report += "Feature Information (Numeric)\n"
        txt_report += "-" * 30 + "\n"
        txt_report += txt_table(feature_info, headers=["Feature", "Min", "Max", "Mean"]) + "\n"

        txt_report += "Data Transformation Steps\n"
        txt_report += "-" * 30 + "\n"
        txt_report += txt_table(data_transformation, headers=["Task", "Transformation"]) + "\n"

        txt_report += "Model Training Summary\n"
        txt_report += "-" * 30 + "\n"
        txt_report += txt_table(ml_results, headers=["Task", "Target", "Features", "Result"]) + "\n"

        # Add prediction samples to TXT report
        for result in ml_results:
            if 'prediction_sample' in result and result['prediction_sample']:
                txt_report += f"\nTop 10 Predictions for {result['Task']}\n"
                txt_report += "-" * (len(f"Top 10 Predictions for {result['Task']}") + 5) + "\n"
                txt_report += txt_table(result['prediction_sample'], headers=result['prediction_sample_headers']) + "\n"

        txt_report += "\nKey Findings\n"
        txt_report += "-" * 30 + "\n"
        txt_report += txt_list(key_findings)
        txt_report += "\nConclusions\n"
        txt_report += "-" * 30 + "\n"
        txt_report += txt_list(conclusions)
        txt_report += "\n" + "=" * 60 + "\nEnd of Report\n" + "=" * 60 + "\n"

        # --- Save as .txt for download ---
        original_filename = metadata.get('original_filename', f"{file_id}.csv")
        base_name = os.path.splitext(original_filename)[0]
        report_filename = f"{base_name}_report.txt"
        report_path = os.path.join(REPORTS_DIR, report_filename)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(txt_report)
        logger.info(f"TXT report saved to {report_path}")

        return jsonify({
            'report_html': html_report,
            'txt_report_content': txt_report, # Pass TXT content directly
            'download_url': f"/download_report/{file_id}" # This endpoint will now serve the generated TXT
        })
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f"Error generating report: {str(e)}"}), 500
        
@app.route('/download_report/<file_id>', methods=['GET'])
def download_report(file_id):
    """Serves the generated report file for download."""
    report_dir = os.path.join(app.root_path, 'reports')
    metadata = mongo.get_file_metadata(file_id)
    if not metadata:
        return "Report not found (metadata missing)", 404
    original_filename = metadata.get('original_filename', f"{file_id}.csv")
    base_name = os.path.splitext(original_filename)[0]
    filename = f"{base_name}_report.txt"
    file_path = os.path.join(report_dir, filename)

    if not os.path.exists(file_path):
        logger.warning(f"Attempted to download non-existent report: {file_path}")
        return "Report not found on server", 404
    
    logger.info(f"Serving report file: {file_path}")
    return send_file(file_path, as_attachment=True)

def delete_file_if_exists(file_path):
    """Deletes a file if it exists."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
        else:
            logger.warning(f"Attempted to delete non-existent file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")

# Register cleanup function to delete app.log on application exit
@atexit.register
def cleanup_on_exit():
    """Function to be called on application exit for cleanup."""
    logger.info("Application is shutting down. Performing cleanup...")
    clear_app_log_file()
    logger.info("Cleanup complete.")

@app.route('/terminal-logs', methods=['GET'])
def get_terminal_logs():
    """Retrieves the last 1000 lines of the application log file, filtering out specific entries."""
    try:
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, 'r') as f:
                lines = f.readlines()[-1000:]
                
                filtered_lines = [
                    line for line in lines 
                    if "werkzeug" not in line and "GET /terminal-logs HTTP/1.1\" 200 -" not in line
                ]
                
                return jsonify({
                    'status': 'success',
                    'logs': filtered_lines
                })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Log file not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Error reading terminal logs: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Launch the Flask app.')
    parser.add_argument('--host', default='127.0.0.1', help='Host IP')
    parser.add_argument('--port', default=5000, type=int, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')
    args = parser.parse_args()

    # DO NOT REMOVE THIS LINE: This opens the browser when the Flask app starts.
    webbrowser.open(f'http://{args.host}:{args.port}/') 
    
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False) # use_reloader=False to prevent double execution
