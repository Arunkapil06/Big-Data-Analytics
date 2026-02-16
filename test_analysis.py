from spark_manager import SparkManager
from data_analyzer import DataAnalyzer

# Initialize Spark
spark_manager = SparkManager()

# Read the CSV file
with open('test_data.csv', 'rb') as f:
    file_content = f.read()

# Initialize analyzer
analyzer = DataAnalyzer(spark_manager, file_content, 'csv')

# Print columns before analysis
print("Columns before analysis:", analyzer.df.columns)

# Try to perform analysis
try:
    results = analyzer.analyze(
        analysis_type='clustering',
        selected_features=['salary', 'age'],
    )
    print("\nAnalysis results:", results)
except Exception as e:
    print("\nError during analysis:", str(e))
    
# Print columns after analysis
print("\nColumns after analysis:", analyzer.df.columns) 