# Dockerfile
FROM python:3.9-slim

# Install dependencies
RUN pip install fastapi uvicorn torch scikit-learn transformers 

# Copy model and app
COPY app.py /app/app.py
COPY utils/ /app/utils/
COPY artifacts/ /app/artifacts/
WORKDIR /app

# Expose the port for FastAPI
EXPOSE 8080

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
