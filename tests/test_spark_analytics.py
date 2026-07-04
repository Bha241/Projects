import pytest
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from src.spark_analytics import (
    load_datasets,
    run_rdd_operations,
    run_key_value_operations,
    run_dataframe_operations,
    academic_placement_etl_pipeline
)

@pytest.fixture(scope="module")
def spark_session():
    spark = SparkSession.builder \
        .appName("Test_Smart_Education_Analytics") \
        .master("local[2]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    yield spark
    spark.stop()

def test_load_datasets(spark_session):
    students_df, scores_df, attendance_df, placement_df, scores_raw = load_datasets(spark_session, large_dataset=False)
    
    assert students_df.count() == 5
    assert scores_df.count() == 6
    assert attendance_df.count() == 5
    assert placement_df.count() == 5
    
    # Check schema columns
    assert "student_id" in students_df.columns
    assert "major" in students_df.columns
    assert "score" in scores_df.columns
    assert "attended_classes" in attendance_df.columns
    assert "is_placed" in placement_df.columns

def test_rdd_operations(spark_session):
    students_df, scores_df, attendance_df, placement_df, scores_raw = load_datasets(spark_session, large_dataset=False)
    scores_rdd = run_rdd_operations(spark_session, scores_raw)
    
    print("TEST RDD COLLECT:", scores_rdd.collect())
    filtered = scores_rdd.filter(lambda x: x[3] > 80.0)
    print("TEST FILTERED COLLECT:", filtered.collect())
    high_scorers_count = filtered.count()
    print("TEST COUNT:", high_scorers_count)
    assert high_scorers_count == 4

def test_key_value_operations(spark_session):
    students_df, scores_df, attendance_df, placement_df, scores_raw = load_datasets(spark_session, large_dataset=False)
    scores_rdd = spark_session.sparkContext.parallelize(scores_raw)
    results = run_key_value_operations(scores_rdd)
    
    # Student 1 has two scores: Math (85.0) and Physics (90.0) -> Total = 175.0
    student_1_total = [r[1] for r in results if r[0] == 1][0]
    assert student_1_total == 175.0

def test_dataframe_operations(spark_session):
    students_df, scores_df, attendance_df, placement_df, scores_raw = load_datasets(spark_session, large_dataset=False)
    academic_df = run_dataframe_operations(students_df, scores_df, attendance_df)
    
    # Check that attendance_pct column exists and is calculated correctly
    assert "attendance_pct" in academic_df.columns
    
    # Check Alice (student_id=1) attendance_pct = (95/100)*100 = 95.0
    alice_row = academic_df.filter(col("student_id") == 1).collect()
    assert len(alice_row) > 0
    # Wait, col is from pyspark.sql.functions, let's import it or filter directly
    alice_row = academic_df.filter(academic_df.student_id == 1).first()
    assert alice_row.attendance_pct == 95.0

def test_etl_pipeline(spark_session, tmpdir):
    students_df, scores_df, attendance_df, placement_df, scores_raw = load_datasets(spark_session, large_dataset=False)
    
    output_dir = str(tmpdir.mkdir("output"))
    final_df = academic_placement_etl_pipeline(students_df, scores_df, attendance_df, placement_df, output_dir)
    
    # Verify that Parquet files were created
    parquet_path = os.path.join(output_dir, "master_analytics")
    assert os.path.exists(parquet_path)
    
    # Verify row count
    assert final_df.count() == 5
    
    # Verify fillna operations
    # Bob (student_id=2) was not placed
    bob_row = final_df.filter(final_df.student_id == 2).first()
    assert bob_row.company == "Unplaced"
    assert bob_row.package_lpa == 0.0
    assert bob_row.is_placed == 0
    
    # Alice (student_id=1) was placed
    alice_row = final_df.filter(final_df.student_id == 1).first()
    assert alice_row.company == "TechCorp"
    assert alice_row.package_lpa == 12.0
    assert alice_row.is_placed == 1
