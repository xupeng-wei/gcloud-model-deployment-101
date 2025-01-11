from pydantic import BaseModel
from typing import List, Optional

class PredictionResponse(BaseModel):
    predictions: list
    

"""
    Base Generator class
"""

class BasePredictionGenerator:
    @staticmethod
    def Initialize():
        raise NotImplementedError()

    @staticmethod
    def Generate(inputs):
        raise NotImplementedError()
    
"""
    Example class for generating responses
"""

import torch
import torch.nn as nn

class RidgeRegressionModel(nn.Module):
    def __init__(self, dim_in, alpha = 1.0):
        super(RidgeRegressionModel, self).__init__()
        self.layer = nn.Linear(dim_in, 1)
        self.alpha = 1.0
    
    def forward(self, x, y = None):
        y_pred = self.layer(x)
        if y is None:
            return y_pred
        mse = (y_pred - y.view(y_pred.shape)).square().mean()
        loss = mse
        for param in self.layer.parameters():
            loss += self.alpha * param.square().sum()
        return y_pred, loss

class PredictionGenerator(BasePredictionGenerator):
    initialized = False
    model = None
    kInputDim = 105
    
    @staticmethod
    def Initialize():
        if not PredictionGenerator.initialized:
            PredictionGenerator.model = RidgeRegressionModel(PredictionGenerator.kInputDim)
            PredictionGenerator.model.load_state_dict(torch.load("artifacts/torch_model.pt"))
            # PredictionGenerator.model = torch.jit.load("torch_model.pt")
            PredictionGenerator.model.eval()
        return
    
    @staticmethod
    def Generate(inputs):
        if not PredictionGenerator.initialized: PredictionGenerator.Initialize()
        with torch.no_grad():
            output = PredictionGenerator.model(inputs.to(torch.float32))
        return output.tolist()


"""
    The following function generates the responses
"""

def GenerateResponses(inputs) -> PredictionResponse:
    return PredictionResponse(predictions=PredictionGenerator.Generate(inputs))