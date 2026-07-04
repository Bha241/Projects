from fastapi import FastAPI, Query
import uvicorn
import os
import sys

# Ensure parent directory is in PYTHONPATH so we can import src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spark_analytics import init_spark, load_datasets, academic_placement_etl_pipeline, run_ml_prediction

app = FastAPI(
    title="Smart Education Analytics API",
    description="REST API for predicting student placement probabilities using Apache Spark MLlib.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Smart Education Analytics Platform API",
        "status": "running",
        "endpoints": {
            "status": "GET /",
            "predict": "GET /predict?avg_score=85&attendance_pct=90"
        }
    }

@app.get("/predict")
def predict_placement(
    avg_score: float = Query(..., description="Average exam score of the student (0-100)"),
    attendance_pct: float = Query(..., description="Attendance percentage (0-100)")
):
    spark = init_spark()
    try:
        # Load datasets and run ETL to get the master dataset
        students_df, scores_df, attendance_df, placement_df, scores_raw = load_datasets(spark)
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "output"))
        master_df = academic_placement_etl_pipeline(students_df, scores_df, attendance_df, placement_df, output_dir)
        
        # Train model on the master dataset
        model = run_ml_prediction(master_df)
        if model is None:
            return {"error": "Could not train prediction model due to insufficient data."}
            
        # Predict for the input features
        from pyspark.ml.feature import VectorAssembler
        
        # Create a single-row DataFrame for prediction
        input_data = spark.createDataFrame([(avg_score, attendance_pct)], ["avg_score", "attendance_pct"])
        assembler = VectorAssembler(inputCols=["avg_score", "attendance_pct"], outputCol="features")
        feature_df = assembler.transform(input_data)
        
        predictions = model.transform(feature_df)
        result = predictions.select("prediction", "probability").first()
        
        # Extract probabilities and prediction
        prob_unplaced, prob_placed = result.probability.toArray().tolist()
        prediction = int(result.prediction)
        
        return {
            "avg_score": avg_score,
            "attendance_pct": attendance_pct,
            "prediction": {
                "is_placed": bool(prediction),
                "label": "Placed" if prediction == 1 else "Unplaced"
            },
            "probabilities": {
                "placed": round(prob_placed, 4),
                "unplaced": round(prob_unplaced, 4)
            }
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        print("Stopping Spark session inside API request...")
        spark.stop()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
