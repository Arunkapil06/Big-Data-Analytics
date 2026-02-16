import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # MongoDB Configuration
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/data_analysis_platform')  # Default Compass connection
    DB_NAME = os.getenv('DB_NAME', 'data_analysis_platform')
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    ALLOWED_EXTENSIONS = {'csv', 'txt', 'json', 'parquet'}
    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB max file size
    
    # Spark Configuration
    SPARK_MASTER = os.getenv('SPARK_MASTER', 'local[*]')
    SPARK_APP_NAME = 'DataAnalysisPlatform'
    
    # Results Configuration
    RESULTS_FOLDER = os.path.join(os.getcwd(), 'results')
    
    # Hadoop Configuration
    HADOOP_FS_DEFAULT = os.getenv('HADOOP_FS_DEFAULT', 'hdfs://localhost:9000')
    HADOOP_USER_NAME = os.getenv('HADOOP_USER_NAME', 'hadoop')
    HDFS_UPLOAD_DIR = os.getenv('HDFS_UPLOAD_DIR', '/user/hadoop/uploads')
    
    # Large File Threshold (in bytes)
    LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB
    
    @staticmethod
    def init_folders():
        """Initialize necessary folders"""
        for folder in [Config.UPLOAD_FOLDER, Config.RESULTS_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                
    @staticmethod
    def get_mongodb_settings():
        """Get MongoDB connection settings"""
        return {
            'host': 'localhost',
            'port': 27017,
            'username': None,  # Set if you have authentication enabled
            'password': None   # Set if you have authentication enabled
        }
    
    @staticmethod
    def get_hdfs_path(filename):
        """Generate HDFS path for a file"""
        return os.path.join(Config.HDFS_UPLOAD_DIR, filename) 