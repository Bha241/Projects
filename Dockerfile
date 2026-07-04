FROM python:3.10-slim

# Install system dependencies, including Java JRE (required for Apache Spark)
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Set working directory
WORKDIR /app

# Install PySpark and required Python libraries
RUN pip install --no-cache-dir pyspark==3.5.5 numpy fastapi uvicorn

# Copy the application source code and tests into the container
COPY src/ /app/src/
COPY tests/ /app/tests/

# Set Python path to ensure module imports resolve correctly
ENV PYTHONPATH=/app

# Expose the API port
EXPOSE 8000

# Default command runs the FastAPI server
CMD ["python", "src/app.py"]
