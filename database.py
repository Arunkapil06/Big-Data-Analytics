from pymongo import MongoClient
from bson import ObjectId
from config import Config
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    def __init__(self):
        try:
            # Connect using the URI which includes database name
            self.client = MongoClient(Config.MONGO_URI)
            self.db = self.client.get_database()
            self.files_collection = self.db.files
            logger.info("MongoDB connected successfully via Compass connection string")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    def test_connection(self):
        """Test MongoDB connection"""
        try:
            # The ismaster command is cheap and does not require auth.
            self.client.admin.command('ismaster')
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {str(e)}")
            raise Exception(f"Could not connect to MongoDB: {str(e)}")
    
    def insert_file_metadata(self, metadata):
        """Insert file metadata and return the inserted ID"""
        try:
            result = self.files_collection.insert_one(metadata)
            logger.info(f"Inserted metadata with ID: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error inserting metadata: {str(e)}")
            raise
    
    def get_file_metadata(self, file_id):
        """Get file metadata by ID"""
        try:
            result = self.files_collection.find_one({'_id': ObjectId(file_id)})
            if result:
                logger.info(f"Retrieved metadata for file_id: {file_id}")
            else:
                logger.warning(f"No metadata found for file_id: {file_id}")
            return result
        except Exception as e:
            logger.error(f"Error getting metadata for file_id {file_id}: {str(e)}")
            return None
    
    def update_file_metadata(self, file_id, update_data):
        """Update file metadata"""
        try:
            result = self.files_collection.update_one(
                {'_id': ObjectId(file_id)},
                {'$set': update_data}
            )
            if result.modified_count > 0:
                logger.info(f"Updated metadata for file_id: {file_id}")
            else:
                logger.warning(f"No metadata updated for file_id: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating metadata for file_id {file_id}: {str(e)}")
            return False
    
    def list_files(self):
        """List all files metadata"""
        try:
            files = list(self.files_collection.find())
            logger.info(f"Retrieved {len(files)} files from database")
            return files
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return [] 