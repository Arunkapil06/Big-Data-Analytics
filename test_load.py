from spark_manager import SparkManager
from data_analyzer import DataAnalyzer

# Initialize Spark
spark_manager = SparkManager()

# Read the CSV file
with open('test_data.csv', 'rb') as f:
    file_content = f.read()

# Initialize analyzer
analyzer = DataAnalyzer(spark_manager, file_content, 'csv')

# Print the schema and first few rows
print("Schema:")
analyzer.df.printSchema()
print("\nFirst few rows:")
analyzer.df.show(5)

# Print available columns
print("\nAvailable columns:", analyzer.df.columns) 