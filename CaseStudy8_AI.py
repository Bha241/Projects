pip install pyspark

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Smart_Education_Analytics") \
    .master("local[*]") \
    .getOrCreate()

# 2. Load Educational Datasets (Mock Data Generation for Reproducibility)
# In a real scenario, you would use: spark.read.csv("path/to/data.csv", header=True, inferSchema=True)

students_data = [(1, "Alice", "CS"), (2, "Bob", "IT"), (3, "Charlie", "CS"), (4, "David", "ME"), (5, "Eve", "IT")]
scores_data = [(1, "Math", 1, 85.0), (1, "Physics", 1, 90.0), (2, "Math", 1, 75.0), (3, "Math", 1, 95.0), (4, "Physics", 1, 60.0), (5, "Math", 1, 88.0)]
attendance_data = [(1, 100, 95), (2, 100, 80), (3, 100, 98), (4, 100, 60), (5, 100, 85)]
placement_data = [(1, "TechCorp", 12.0, 1), (3, "DataInc", 15.0, 1), (5, "WebCo", 10.0, 1), (2, None, 0.0, 0), (4, None, 0.0, 0)]

students_df = spark.createDataFrame(students_data, ["student_id", "name", "major"])
scores_df = spark.createDataFrame(scores_data, ["student_id", "subject", "semester", "score"])
attendance_df = spark.createDataFrame(attendance_data, ["student_id", "total_classes", "attended_classes"])
placement_df = spark.createDataFrame(placement_data, ["student_id", "company", "package_lpa", "is_placed"])

print("Data Loaded Successfully.")

# Create an RDD from the scores data
scores_rdd = spark.sparkContext.parallelize(scores_data)

# Transformation 1: Filter students who scored more than 80
high_scorers_rdd = scores_rdd.filter(lambda x: x[3] > 80.0)

# Transformation 2: Map to extract only student_id and score
student_scores_map = high_scorers_rdd.map(lambda x: (x[0], x[3]))

# Action 1: Count the number of high scores
high_score_count = high_scorers_rdd.count()

# Action 2: Collect and print the results
print(f"Total High Scores (>80): {high_score_count}")
print("High Scorers (ID, Score):", student_scores_map.collect())

from pyspark import StorageLevel

# Create a Key-Value RDD (student_id as key, score as value)
kv_scores_rdd = scores_rdd.map(lambda x: (x[0], x[3]))

# Shuffle Operation: Calculate total score per student using reduceByKey
total_score_per_student = kv_scores_rdd.reduceByKey(lambda a, b: a + b)

# Persistence: Cache the resulting RDD to memory and disk to avoid recomputation
total_score_per_student.persist(StorageLevel.MEMORY_AND_DISK)

print("Total Score Per Student RDD (Persisted):")
print(total_score_per_student.collect())

# Unpersist when done
total_score_per_student.unpersist()

from pyspark.sql.functions import avg, col, round

# 1. Joins: Join Students, Scores, and Attendance DataFrames
academic_df = students_df.join(scores_df, on="student_id", how="inner") \
                         .join(attendance_df, on="student_id", how="inner")

# Calculate Attendance Percentage
academic_df = academic_df.withColumn("attendance_pct", round((col("attended_classes") / col("total_classes")) * 100, 2))

# 2. Aggregations: Average score and average attendance by Major
major_performance_df = academic_df.groupBy("major").agg(
    round(avg("score"), 2).alias("avg_score"),
    round(avg("attendance_pct"), 2).alias("avg_attendance")
)

major_performance_df.show()

# Register DataFrames as Temporary SQL Views
academic_df.createOrReplaceTempView("academic_records")
placement_df.createOrReplaceTempView("placements")

# 1. Analyze student attendance patterns
print("--- Attendance Patterns ---")
spark.sql("""
    SELECT major, AVG(attendance_pct) as avg_attendance
    FROM academic_records
    GROUP BY major
    ORDER BY avg_attendance DESC
""").show()

# 2. Determine subject-wise performance
print("--- Subject-wise Performance ---")
spark.sql("""
    SELECT subject, MAX(score) as max_score, MIN(score) as min_score, ROUND(AVG(score), 2) as avg_score
    FROM academic_records
    GROUP BY subject
""").show()

# 3. Identify top-performing students[cite: 2]
print("--- Top Performing Students ---")
spark.sql("""
    SELECT name, major, ROUND(AVG(score), 2) as overall_avg
    FROM academic_records
    GROUP BY name, major
    ORDER BY overall_avg DESC LIMIT 3
""").show()

# 4. Analyze placement trends[cite: 2]
print("--- Placement Trends by Major ---")
spark.sql("""
    SELECT a.major,
           COUNT(p.student_id) as total_students,
           SUM(p.is_placed) as placed_students,
           ROUND(AVG(p.package_lpa), 2) as avg_package
    FROM academic_records a
    JOIN placements p ON a.student_id = p.student_id
    GROUP BY a.major
""").show()

# 5. Generate semester-wise academic reports[cite: 2]
print("--- Semester-wise Academic Report ---")
spark.sql("""
    SELECT semester, ROUND(AVG(score), 2) as avg_semester_score, COUNT(student_id) as exams_taken
    FROM academic_records
    GROUP BY semester
""").show()

def academic_placement_etl_pipeline(students, scores, attendance, placements):
    # EXTRACT: (Data is already extracted into DataFrames via arguments)

    # TRANSFORM:
    # Clean data, calculate average scores, drop nulls, and join everything into a master analytical dataset
    avg_scores = scores.groupBy("student_id").agg(round(avg("score"), 2).alias("avg_score"))

    transformed_df = students.join(avg_scores, on="student_id", how="left") \
                             .join(attendance, on="student_id", how="left") \
                             .join(placements, on="student_id", how="left")

    # Add calculated columns and fill nulls for placement data
    final_df = transformed_df.withColumn("attendance_pct", round((col("attended_classes") / col("total_classes")) * 100, 2)) \
                             .fillna({"company": "Unplaced", "package_lpa": 0.0, "is_placed": 0})

    # LOAD:
    # Write the transformed data to Parquet format (simulated here by writing to a local temp directory)
    output_path = "/tmp/smart_education_analytics_master"
    final_df.write.mode("overwrite").parquet(output_path)
    print(f"ETL Pipeline Completed. Data loaded to {output_path}")

    return final_df

# Execute the Pipeline
master_analytics_df = academic_placement_etl_pipeline(students_df, scores_df, attendance_df, placement_df)
master_analytics_df.show()

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Prepare data for ML (Predicting 'is_placed' based on 'avg_score' and 'attendance_pct')
ml_df = master_analytics_df.select("avg_score", "attendance_pct", "is_placed").dropna()

# 1. Feature Engineering
assembler = VectorAssembler(inputCols=["avg_score", "attendance_pct"], outputCol="features")
feature_df = assembler.transform(ml_df)

# Select only features and label for training
model_data = feature_df.select("features", col("is_placed").alias("label"))

# 2. Train-Test Split
train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)

# 3. Model Implementation (Logistic Regression for Binary Classification)
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
lr_model = lr.fit(train_data)

# 4. Predictions & Evaluation
predictions = lr_model.transform(test_data)

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)

print(f"Model Evaluation - Area Under ROC: {auc}")
predictions.select("features", "label", "prediction", "probability").show()

# Stop Spark session
spark.stop()

pip install pyspark

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from pyspark.sql.functions import col, hour, month, to_timestamp, sum, avg, round
from datetime import datetime

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("Smart_Energy_Consumption_Analytics") \
    .master("local[*]") \
    .getOrCreate()

# 2. Mock Data Generation
consumer_data = [
    ("C1", "North", "Residential"), ("C2", "North", "Commercial"),
    ("C3", "South", "Industrial"), ("C4", "East", "Residential"),
    ("C5", "West", "Commercial")
]

meter_data = [
    ("M1", "C1", "2023-10-01 08:00:00", 2.5), ("M1", "C1", "2023-10-01 14:00:00", 3.1),
    ("M2", "C2", "2023-10-01 08:00:00", 15.0), ("M2", "C2", "2023-10-01 14:00:00", 18.5),
    ("M3", "C3", "2023-10-01 08:00:00", 150.0), ("M3", "C3", "2023-10-01 14:00:00", 165.0),
    ("M4", "C4", "2023-11-15 19:00:00", 4.2), ("M5", "C5", "2023-11-15 19:00:00", 12.0)
]

weather_data = [
    ("North", "2023-10-01 08:00:00", 22.0, 60), ("North", "2023-10-01 14:00:00", 28.5, 55),
    ("South", "2023-10-01 08:00:00", 26.0, 70), ("South", "2023-10-01 14:00:00", 32.0, 65),
    ("East", "2023-11-15 19:00:00", 18.0, 80), ("West", "2023-11-15 19:00:00", 24.0, 60)
]

# Create DataFrames
consumer_df = spark.createDataFrame(consumer_data, ["consumer_id", "region", "category"])
meter_df = spark.createDataFrame(meter_data, ["meter_id", "consumer_id", "timestamp", "energy_kwh"])
weather_df = spark.createDataFrame(weather_data, ["region", "timestamp", "temperature", "humidity"])

# Cast strings to timestamps
meter_df = meter_df.withColumn("timestamp", to_timestamp(col("timestamp")))
weather_df = weather_df.withColumn("timestamp", to_timestamp(col("timestamp")))

print("Data Loaded Successfully.")

# Convert meter DataFrame to RDD
meter_rdd = meter_df.rdd

# Transformation: Filter records where consumption is considered 'High' (e.g., > 10 kWh)
high_consumption_rdd = meter_rdd.filter(lambda row: row.energy_kwh > 10.0)

# Action 1: Count the number of high consumption events
high_consumption_count = high_consumption_rdd.count()

# Action 2: Collect and print the first 3 high consumption records
print(f"Total High Consumption Events (>10 kWh): {high_consumption_count}")
print("Sample Records:", high_consumption_rdd.take(3))

from pyspark import StorageLevel

# Create Key-Value pair RDD (Key: consumer_id, Value: energy_kwh)
kv_meter_rdd = meter_rdd.map(lambda row: (row.consumer_id, row.energy_kwh))

# Shuffle Operation: Calculate total energy consumed per consumer using reduceByKey
total_energy_per_consumer = kv_meter_rdd.reduceByKey(lambda a, b: a + b)

# Persistence: Cache to memory and disk
total_energy_per_consumer.persist(StorageLevel.MEMORY_AND_DISK)

print("Total Energy Consumption Per Consumer:")
print(total_energy_per_consumer.collect())

# Unpersist to free resources when done
total_energy_per_consumer.unpersist()

# Join Meter and Consumer Data
enriched_meter_df = meter_df.join(consumer_df, on="consumer_id", how="inner")

# Selection & Filtering: Select specific columns for 'Residential' users only
residential_df = enriched_meter_df.select("consumer_id", "region", "timestamp", "energy_kwh") \
                                  .filter(col("category") == "Residential")

# Grouping & Aggregation: Calculate Average and Total Consumption by Region
region_agg_df = enriched_meter_df.groupBy("region").agg(
    round(sum("energy_kwh"), 2).alias("total_energy"),
    round(avg("energy_kwh"), 2).alias("avg_energy")
)

print("Aggregated Energy by Region:")
region_agg_df.show()

# Register DataFrames as Temporary Views
enriched_meter_df.createOrReplaceTempView("energy_data")

# 1. Analyze hourly energy consumption
print("--- Hourly Energy Consumption ---")
spark.sql("""
    SELECT HOUR(timestamp) as hour_of_day, ROUND(SUM(energy_kwh), 2) as total_consumption
    FROM energy_data
    GROUP BY HOUR(timestamp)
    ORDER BY hour_of_day
""").show()

# 2. Determine region-wise electricity usage
print("--- Region-wise Electricity Usage ---")
spark.sql("""
    SELECT region, ROUND(SUM(energy_kwh), 2) as total_usage
    FROM energy_data
    GROUP BY region
    ORDER BY total_usage DESC
""").show()

# 3. Identify peak demand periods
print("--- Peak Demand Periods (Top 2 Hours) ---")
spark.sql("""
    SELECT HOUR(timestamp) as peak_hour, ROUND(SUM(energy_kwh), 2) as demand
    FROM energy_data
    GROUP BY HOUR(timestamp)
    ORDER BY demand DESC
    LIMIT 2
""").show()

# 4. Analyze consumer categories
print("--- Consumption by Consumer Category ---")
spark.sql("""
    SELECT category, COUNT(DISTINCT consumer_id) as total_consumers, ROUND(SUM(energy_kwh), 2) as total_usage
    FROM energy_data
    GROUP BY category
""").show()

# 5. Generate monthly consumption reports
print("--- Monthly Consumption Report ---")
spark.sql("""
    SELECT MONTH(timestamp) as month, ROUND(SUM(energy_kwh), 2) as monthly_usage
    FROM energy_data
    GROUP BY MONTH(timestamp)
    ORDER BY month
""").show()

def energy_weather_etl_pipeline(meters, consumers, weather):
    # EXTRACT: (Data is passed as DataFrames)

    # TRANSFORM:
    # 1. Join Meters with Consumers
    df_joined = meters.join(consumers, on="consumer_id", how="inner")

    # 2. Extract Date and Hour for accurate joining with weather data
    df_joined = df_joined.withColumn("date_hour", to_timestamp(col("timestamp")))
    weather_clean = weather.withColumn("date_hour", to_timestamp(col("timestamp"))) \
                           .drop("timestamp") # Drop to avoid duplicate columns

    # 3. Join with Weather data based on region and time
    master_df = df_joined.join(weather_clean, on=["region", "date_hour"], how="left")

    # 4. Feature Engineering: Flag peak hours (e.g., 14:00 / 2 PM)
    master_df = master_df.withColumn("is_peak_hour", (hour(col("date_hour")) == 14).cast("integer")) \
                         .fillna({"temperature": 25.0, "humidity": 50}) # Handle missing weather

    # LOAD: Write transformed data to Parquet
    output_path = "/tmp/smart_energy_master"
    master_df.write.mode("overwrite").parquet(output_path)
    print(f"ETL Complete. Master dataset saved to: {output_path}")

    return master_df

# Execute Pipeline
master_energy_df = energy_weather_etl_pipeline(meter_df, consumer_df, weather_df)
master_energy_df.show()

from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# 1. Prepare ML Data
ml_df = master_energy_df.select("category", "temperature", "humidity", "energy_kwh").dropna()

# 2. Feature Engineering (Categorical to Numerical)
indexer = StringIndexer(inputCol="category", outputCol="category_index")
encoder = OneHotEncoder(inputCol="category_index", outputCol="category_vec")
assembler = VectorAssembler(inputCols=["category_vec", "temperature", "humidity"], outputCol="features")

# 3. Train/Test Split
train_data, test_data = ml_df.randomSplit([0.8, 0.2], seed=42)

# 4. Model Implementation (Linear Regression for demand prediction)
lr = LinearRegression(featuresCol="features", labelCol="energy_kwh")

# 5. Build and Run Pipeline
pipeline = Pipeline(stages=[indexer, encoder, assembler, lr])
model = pipeline.fit(train_data)
predictions = model.transform(test_data)

# 6. Evaluation
evaluator = RegressionEvaluator(labelCol="energy_kwh", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"Model Evaluation - Root Mean Squared Error (RMSE): {rmse}")
predictions.select("category", "temperature", "energy_kwh", "prediction").show()

# Stop spark context
spark.stop()


