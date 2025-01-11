# Google Cloud Model Deployment 101

## Introduction

This repository provides a beginner-friendly template to deploy a machine learning model on Google Cloud using the Vertex AI platform.

## Prerequisites

Before getting started, ensure you have the following:

- Access to the Vertex AI service on Google Cloud.

- A machine learning model ready for deployment.

- A local environment where you can create and test Docker images.

## Example Model Overview

This guide uses a mock retail prediction model as an example. The model takes the following three transaction features as inputs:

- Price: A numerical feature (float).

- Country: A categorical feature (string). Any string value is acceptable.

- Description: A variable-length string feature.
The model predicts the quantity (numerical, float) for the transaction. 

## Deployment Steps

### Prepare the Docker Image

- Place the model parameters and other required resources in the `./artifact` directory.

- Create an `app.py` file to define how the model interacts with user requests.

> **Tip:** You can use the `app.py` template provided in this repository as-is without modification.
> 
> **Note:** The `health()` function in the template is a basic implementation of a health check. You may customize it to fit your specific requirements.

- In `app.py`, you might notice the following imports:

    ```python
    from utils.request import PredictionRequest, ParseInputInstances
    from utils.response import PredictionResponse, GenerateResponses
    ```

    These classes and interfaces have not been completely implemented yet. We will implement them in the next step.

- In the `./utils` directory, you will find two template files: `request.py` and `response.py`. You need to complete these templates based on your specific requirements:

    - `./utils/request.py`:

        - This file defines the `PredictionRequest` class, which represents a list of `Instance` objects. You need to define the schema for these `Instance` objects. For example:

        ```python
        class Instance(BaseModel):
            price: float
            country: str
            description: str
        ```

        - It also defines how raw requests from users will be processed. You need to create your own input parser class (e.g., `InputParser`) as a derived class of BaseInputParser. You must implement the following two static methods:

            - `Initialize()`: Handles any preparation needed for parsing input, such as loading a text encoder model (e.g., DistilBERT in this example). If no preparation is needed, provide an empty implementation, but ensure the method is defined.

            - `Parse()`: Defines how the parser processes the input. The input will be an instance of `PredictionRequest` (a list of `Instance` objects). The output of this method will be passed to the generator defined in `./utils/response.py`.

    - `./utils/response.py`: 

        - This file defines the `PredictionResponse` class, which contains a list of predictions.

        - You need to implement a `PredictionGenerator` class derived from `BaseGenerator`. This class must include the following two static methods:

            - `Initialize()`: Handles any preparation needed for prediction, such as loading the model you want to deploy.

            - `Generate()`: Computes the predicted results and returns them. The results should be formatted as a list, even if there is only one prediction.

### Test the Docker Image Locally

- Use the following command to build the Docker image:

```bash
docker build -t demo-model .
```

After building, you can verify the image was created by listing your local Docker images:

```bash
docker images
```

- Run a container

Start a container from the Docker image:

```bash
docker run -p 9090:8080 demo-model
```

In this command, `-p 9090:8080` maps your local port `9090` to port `8080` inside the container.

> **Why use port 9090?** Port `8080` is commonly used for other purposes on some platforms, so using `9090` as the external port helps avoid potential conflicts. You can choose a different local port if needed. 

- Prepare a request demo for testing, such as the `request_demo.json` file provided in this repository. Additionally, you may want to create a demo with an invalid format (e.g., `invalid_request_demo.json`) to test how your deployment handles invalid input.

- Open another terminal and run the `local_request.sh` script to send the request. 


```bash
sh local_request.sh
```

This script uses a curl command to interact with the service. If you customized the port number in the previous step, make sure to update it accordingly in the script.

An expected response might look like the following:

```json
{"predictions":[[3.971233367919922],[-6.595061302185059],[1.1781799793243408]]}
```

For an invalid request, you might receive a response like this:

```json
{
  "detail": [
    {
      "type": "string_type",
      "loc": ["body", "instances", 2, "country"],
      "msg": "Input should be a valid string",
      "input": 2
    }
  ],
  "body": {
    "instances": [
      {"price": 1.0, "country": "Philippines", "description": "da coconut nut is not a nut..."},
      {"price": 20.3, "country": "United States", "description": "Oh, Shenandoah, I long to hear you"},
      {"price": 3.2, "country": 2, "description": "Bright copper kettles and warm woolen mittens"}
    ]
  }
}
```

If there are any issues in your implementation of `./utils/request.py` or `./utils/response.py`, you may encounter error messages like:

```json
{"detail":"name 'np' is not defined"}
```

Make sure to address any unexpected output before proceeding to the next steps. Debugging these errors will help ensure that your implementation is robust and handles both valid and invalid inputs correctly.

### Deploy to Vertex AI

- Push the Model Image to Google Container Registry (GCR)

Run the following commands to authenticate Docker with GCR, tag the Docker image, and push it to your Google Cloud project:

```bash
gcloud auth configure-docker
docker tag demo-model gcr.io/[PROJECT-ID]/demo-model:latest
docker push gcr.io/[PROJECT-ID]/demo-model:latest
```
Replace `[PROJECT-ID]` with your own Google Cloud project ID.

- Upload the Model Image from GCR to the Model Registry

Select a region based on your requirements (e.g., proximity to your user base, hardware availability, cost, etc.), then upload the model to the Vertex AI model registry. Replace [REGION] with your chosen region (e.g., us-central1):

```bash
gcloud ai models upload \
  --region=[REGION] \
  --display-name=torch-model \
  --container-image-uri=gcr.io/[PROJECT-ID]/torch-model:latest
```

Once the upload is complete, retrieve the model ID either via the Google Cloud Console or by running the following command:

```bash
gcloud ai models list --region=[REGION]
```

- Create an endpoint in the corresponding region:

Creating an endpoint is independent of the model upload process (i.e., you can create the endpoint before or after registering the model). Run the following command to create an endpoint:

```bash
gcloud ai endpoints create --region=[REGION] --display-name=[DISPLAY_NAME]
```

Replace `[DISPLAY_NAME]` with a name of your choice, such as `demo-endpoint`.

To list the existing endpoints in a specific region, use the following command:

```bash
gcloud ai endpoints list --region=[REGION]
```

- Deploy the model to an endpoint: 

To deploy a model to an endpoint, run:

```bash
gcloud ai endpoints deploy-model [ENDPOINT-ID] \
  --region=[REGION] \
  --model=[MODEL-ID] \
  --machine-type=[MACHINE-TYPE] \
  --display-name=[NAME]
```

> Replace `[ENDPOINT-ID]` with the ID of the endpoint where the model will be deployed.
>
> Replace `[MODEL-ID]` with the ID of the model you uploaded to the model registry.
>
> Replace `[MACHINE-TYPE]` with the machine type of your choice (e.g., `n1-standard-4` for a standard machine type).
>
> Replace `[NAME]` with a display name for this deployment, such as `model-v1`.

> **Note:** The `[MACHINE-TYPE]` determines the compute resources allocated for the model serving. Choose one that suits your performance and cost requirements.

### Test the deployed model

To test your deployed model, you can run the following script in the CLI:

```bash
#!/bin/bash

REGION="xxx" # Your region here
ENDPOINT_ID="xxx" # Your endpoint id here
PROJECT_ID="xxx" # Your project id here
INPUT_DATA_FILE='./request_demo.json'

curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
"https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict" \
-d "@${INPUT_DATA_FILE}"
```

> Replace `REGION`, `ENDPOINT_ID`, and `PROJECT_ID` with your specific values.
> 
> Ensure `request_demo.json` contains valid input data for your model.

You can also test your deployed model directly in the Vertex AI console by navigating to your endpoint and using the "Deploy & Test" tab to send requests and view responses.