import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, avg, round, lit
from pyspark import StorageLevel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize Spark Session
def init_spark():
    print("--- Q1: Initializing Spark Session ---")
    spark = SparkSession.builder \
        .appName("Smart_Education_Analytics") \
        .master("local[*]") \
        .getOrCreate()
    # Suppress verbose logging
    spark.sparkContext.setLogLevel("WARN")
    print("Spark Session initialized successfully.")
    return spark

# Load or generate educational datasets
def load_datasets(spark, large_dataset=False):
    print("--- Q1: Loading Educational Datasets ---")
    if not large_dataset:
        # Standard mock data from the case study requirements
        students_data = [
            (1, "Alice", "CS"),
            (2, "Bob", "IT"),
            (3, "Charlie", "CS"),
            (4, "David", "ME"),
            (5, "Eve", "IT")
        ]
        scores_data = [
            (1, "Math", 1, 85.0),
            (1, "Physics", 1, 90.0),
            (2, "Math", 1, 75.0),
            (3, "Math", 1, 95.0),
            (4, "Physics", 1, 60.0),
            (5, "Math", 1, 88.0)
        ]
        attendance_data = [
            (1, 100, 95),
            (2, 100, 80),
            (3, 100, 98),
            (4, 100, 60),
            (5, 100, 85)
        ]
        placement_data = [
            (1, "TechCorp", 12.0, 1),
            (3, "DataInc", 15.0, 1),
            (5, "WebCo", 10.0, 1),
            (2, None, 0.0, 0),
            (4, None, 0.0, 0)
        ]
    else:
        # Generate a larger synthetic dataset for more robust ML training/evaluation
        print("Generating a larger synthetic dataset (100 students)...")
        students_data = []
        scores_data = []
        attendance_data = []
        placement_data = []
        
        majors = ["CS", "IT", "ME", "ECE"]
        companies = ["TechCorp", "DataInc", "WebCo", "DesignHub", None]
        
        import random
        random.seed(42)
        
        for i in range(1, 101):
            major = random.choice(majors)
            students_data.append((i, f"Student_{i}", major))
            
            # Scores for Math and Physics
            math_score = float(random.randint(50, 100))
            phys_score = float(random.randint(45, 100))
            scores_data.append((i, "Math", 1, math_score))
            scores_data.append((i, "Physics", 1, phys_score))
            
            # Attendance
            total_classes = 100
            attended = random.randint(50, 100)
            attendance_data.append((i, total_classes, attended))
            
            # Placement logic based on average score & attendance
            avg_score = (math_score + phys_score) / 2
            attendance_pct = attended
            
            is_placed = 0
            company = None
            package = 0.0
            
            if avg_score > 75 and attendance_pct > 75:
                is_placed = 1
                company = random.choice(companies[:-1])
                package = round(random.uniform(6.0, 16.0), 1)
            elif avg_score > 60 and attendance_pct > 65:
                # 50% chance
                if random.random() > 0.5:
                    is_placed = 1
                    company = random.choice(companies[:-1])
                    package = round(random.uniform(3.5, 7.5), 1)
            
            placement_data.append((i, company, package, is_placed))

    students_df = spark.createDataFrame(students_data, ["student_id", "name", "major"])
    scores_df = spark.createDataFrame(scores_data, ["student_id", "subject", "semester", "score"])
    attendance_df = spark.createDataFrame(attendance_data, ["student_id", "total_classes", "attended_classes"])
    placement_df = spark.createDataFrame(placement_data, ["student_id", "company", "package_lpa", "is_placed"])
    
    print("Data Loaded Successfully.")
    return students_df, scores_df, attendance_df, placement_df, scores_data

# Q2: RDD Implementation
def run_rdd_operations(spark, scores_data):
    print("\n--- Q2: RDD Implementation ---")
    scores_rdd = spark.sparkContext.parallelize(scores_data)
    
    # Transformation 1: Filter students who scored more than 80
    high_scorers_rdd = scores_rdd.filter(lambda x: x[3] > 80.0)
    
    # Transformation 2: Map to extract only student_id and score
    student_scores_map = high_scorers_rdd.map(lambda x: (x[0], x[3]))
    
    # Action 1: Count the number of high scores
    high_score_count = high_scorers_rdd.count()
    
    # Action 2: Collect and print
    results = student_scores_map.collect()
    print(f"Total High Scores (>80): {high_score_count}")
    print(f"High Scorers (ID, Score): {results}")
    return scores_rdd

# Q3: Key-Value Operations and Persistence
def run_key_value_operations(scores_rdd):
    print("\n--- Q3: Key-Value Operations and Persistence ---")
    # Create Key-Value RDD (student_id as key, score as value)
    kv_scores_rdd = scores_rdd.map(lambda x: (x[0], x[3]))
    
    # Shuffle Operation: Calculate total score per student using reduceByKey
    total_score_per_student = kv_scores_rdd.reduceByKey(lambda a, b: a + b)
    
    # Persistence: Cache to memory and disk
    total_score_per_student.persist(StorageLevel.MEMORY_AND_DISK)
    
    print("Total Score Per Student RDD (Persisted):")
    results = total_score_per_student.collect()
    print(results)
    
    # Unpersist when done
    total_score_per_student.unpersist()
    return results

# Q4: Spark DataFrame Operations
def run_dataframe_operations(students_df, scores_df, attendance_df):
    print("\n--- Q4: Spark DataFrame Operations ---")
    # Joins
    academic_df = students_df.join(scores_df, on="student_id", how="inner") \
                             .join(attendance_df, on="student_id", how="inner")
                             
    # Calculate Attendance Percentage
    academic_df = academic_df.withColumn("attendance_pct", round((col("attended_classes") / col("total_classes")) * 100, 2))
    
    # Aggregations: Average score and average attendance by Major
    major_performance_df = academic_df.groupBy("major").agg(
        round(avg("score"), 2).alias("avg_score"),
        round(avg("attendance_pct"), 2).alias("avg_attendance")
    )
    major_performance_df.show()
    return academic_df

# Q5: Exploratory Data Analysis and Spark SQL
def run_spark_sql_queries(spark, academic_df, placement_df):
    print("\n--- Q5: Exploratory Data Analysis and Spark SQL ---")
    # Register DataFrames as Temporary SQL Views
    academic_df.createOrReplaceTempView("academic_records")
    placement_df.createOrReplaceTempView("placements")
    
    # 1. Analyze student attendance patterns
    print("--- 1. Attendance Patterns by Major ---")
    spark.sql("""
        SELECT major, ROUND(AVG(attendance_pct), 2) as avg_attendance
        FROM academic_records
        GROUP BY major
        ORDER BY avg_attendance DESC
    """).show()
    
    # 2. Determine subject-wise performance
    print("--- 2. Subject-wise Performance ---")
    spark.sql("""
        SELECT subject, MAX(score) as max_score, MIN(score) as min_score, ROUND(AVG(score), 2) as avg_score
        FROM academic_records
        GROUP BY subject
    """).show()
    
    # 3. Identify top-performing students
    print("--- 3. Top Performing Students ---")
    spark.sql("""
        SELECT name, major, ROUND(AVG(score), 2) as overall_avg
        FROM academic_records
        GROUP BY name, major
        ORDER BY overall_avg DESC LIMIT 3
    """).show()
    
    # 4. Analyze placement trends
    print("--- 4. Placement Trends by Major ---")
    spark.sql("""
        SELECT a.major,
               COUNT(p.student_id) as total_students,
               SUM(p.is_placed) as placed_students,
               ROUND(AVG(p.package_lpa), 2) as avg_package
        FROM academic_records a
        JOIN placements p ON a.student_id = p.student_id
        GROUP BY a.major
    """).show()
    
    # 5. Generate semester-wise academic reports
    print("--- 5. Semester-wise Academic Report ---")
    spark.sql("""
        SELECT semester, ROUND(AVG(score), 2) as avg_semester_score, COUNT(student_id) as exams_taken
        FROM academic_records
        GROUP BY semester
    """).show()

# Q6: ETL Pipeline Development
def academic_placement_etl_pipeline(students, scores, attendance, placements, output_dir):
    print("\n--- Q6: ETL Pipeline Development ---")
    # EXTRACT is handled by parameters
    
    # TRANSFORM: Calculate average score, join data, and clean nulls
    avg_scores = scores.groupBy("student_id").agg(round(avg("score"), 2).alias("avg_score"))
    
    transformed_df = students.join(avg_scores, on="student_id", how="left") \
                             .join(attendance, on="student_id", how="left") \
                             .join(placements, on="student_id", how="left")
                             
    # Add calculated columns and fill nulls
    final_df = transformed_df.withColumn("attendance_pct", round((col("attended_classes") / col("total_classes")) * 100, 2)) \
                             .fillna({"company": "Unplaced", "package_lpa": 0.0, "is_placed": 0})
                             
    # LOAD: Write as Parquet
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "master_analytics")
    final_df.write.mode("overwrite").parquet(parquet_path)
    print(f"ETL Pipeline Completed. Data loaded to {parquet_path}")
    return final_df

# Q7: Machine Learning Implementation
def run_ml_prediction(master_analytics_df):
    print("\n--- Q7: Machine Learning Implementation ---")
    ml_df = master_analytics_df.select("avg_score", "attendance_pct", "is_placed").dropna()
    
    # Check if there are enough records for ML training
    row_count = ml_df.count()
    if row_count < 2:
        print("Not enough records in the dataset to train a machine learning model.")
        return
        
    # Feature Engineering
    assembler = VectorAssembler(inputCols=["avg_score", "attendance_pct"], outputCol="features")
    feature_df = assembler.transform(ml_df)
    model_data = feature_df.select("features", col("is_placed").alias("label"))
    
    # Train-Test Split (use 80/20 with seed 42)
    # If the dataset is small, split might result in empty set, handle that case safely
    train_data, test_data = model_data.randomSplit([0.8, 0.2], seed=42)
    
    if train_data.count() == 0 or test_data.count() == 0:
        print("Dataset size is too small for standard split, using entire dataset for both training and testing.")
        train_data = model_data
        test_data = model_data
        
    # Train Logistic Regression Model
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
    lr_model = lr.fit(train_data)
    
    # Make Predictions
    predictions = lr_model.transform(test_data)
    
    # Evaluate
    # Area under ROC requires both positive and negative classes in test set.
    # Check if we have positive and negative classes in test data
    labels = test_data.select("label").distinct().rdd.flatMap(lambda x: x).collect()
    if len(labels) > 1:
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(predictions)
        print(f"Model Evaluation - Area Under ROC: {auc}")
    else:
        print("Test set contains only one class. Area Under ROC cannot be computed. Standard outputs shown instead.")
        
    predictions.select("features", "label", "prediction", "probability").show()
    return lr_model

def main():
    large_mode = "--large" in sys.argv
    spark = init_spark()
    
    try:
        # Q1: Load Data
        students_df, scores_df, attendance_df, placement_df, scores_raw = load_datasets(spark, large_dataset=large_mode)
        
        # Q2: RDD Operations
        scores_rdd = run_rdd_operations(spark, scores_raw)
        
        # Q3: Key-Value & Persistence
        run_key_value_operations(scores_rdd)
        
        # Q4: DataFrame Operations
        academic_df = run_dataframe_operations(students_df, scores_df, attendance_df)
        
        # Q5: SQL EDA
        run_spark_sql_queries(spark, academic_df, placement_df)
        
        # Q6: ETL Pipeline
        output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "output"))
        master_df = academic_placement_etl_pipeline(students_df, scores_df, attendance_df, placement_df, output_directory)
        master_df.show()
        
        # Q7: ML Prediction
        run_ml_prediction(master_df)
        
    finally:
        print("\nStopping Spark Session...")
        spark.stop()
        print("Spark Session stopped.")

if __name__ == "__main__":
    main()
