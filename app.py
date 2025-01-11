from utils.request import PredictionRequest, ParseInputInstances
from utils.response import PredictionResponse, GenerateResponses

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get(os.getenv('AIP_HEALTH_ROUTE', '/health'), status_code=200)
def health():
    return {}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the details of the request and validation error
    logger.error(f"Validation error for request: {await request.body()}")
    logger.error(f"Error details: {exc.errors()}")

    # Respond with a detailed error message
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body,
        },
    )

@app.post(os.getenv('AIP_PREDICT_ROUTE', "/predict"), response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        inputs = ParseInputInstances(request)
        return GenerateResponses(inputs)
    except Exception as e:
        logger.warning(f"Error input: {str(request)}")
        raise HTTPException(status_code=500, detail=str(e))