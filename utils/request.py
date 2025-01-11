from pydantic import BaseModel
from typing import List, Optional

"""
    Example class of Instance
"""

class Instance(BaseModel):
    price: float
    country: str
    description: str
    
"""
    Request class
"""
    
class PredictionRequest(BaseModel):
    instances: List[Instance]
    

"""
    Base Parser class
"""
class BaseInputParser:
    @staticmethod
    def Initialize():
        raise NotImplementedError()
        
    @staticmethod
    def Parse(instances: List[Instance]):
        raise NotImplementedError()
    
"""
    Example class for input instance processing
"""
        

import pickle
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class InputParser(BaseInputParser):
    initialized = False
    scaler = None # for price processor
    encoder = None # for country processor
    tokenizer = None # for description processor
    bert = None
    
    @staticmethod
    def Initialize():
        if not InputParser.initialized:
            with open("artifacts/scaler.pkl", "rb") as f:
                InputParser.scaler = pickle.load(f)
            with open("artifacts/one_hot_encoder.pkl", "rb") as f:
                InputParser.encoder = pickle.load(f)
            InputParser.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model_config = DistilBertConfig(
                dim = 64,
                hidden_dim = 64 * 4,
                n_heads = 4, # config.n_heads 12 must divide config.dim 64 evenly
            )
            InputParser.bert = DistilBertModel(model_config)
            InputParser.bert.eval()
        return
    
    @staticmethod
    def GenerateTruncatedLogPrice(price, lower_price: float = -2.0, upper_price: float = 15.0) -> None:
        return np.log1p(np.array(price).reshape(-1, 1).clip(lower_price, upper_price) - lower_price)

    @staticmethod
    def ProcessPrice(price, lower_price: float = -2.0, upper_price: float = 15.0) -> dict:
        log_price = InputParser.GenerateTruncatedLogPrice(price)
        return np.array(InputParser.scaler.transform(log_price.reshape(-1, 1))).reshape(-1)

    @staticmethod
    def ProcessCountry(country):
        return InputParser.encoder.transform(np.array([country]).reshape(-1, 1)).todense()

    @staticmethod
    def ProcessDesc(desc):
        inputs = InputParser.tokenizer(desc, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = InputParser.bert(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        return embeddings.detach().numpy()      

    @staticmethod
    def process_input(instance: Instance):
        price = InputParser.ProcessPrice(instance.price)
        c_code = InputParser.ProcessCountry(instance.country)
        desc_embed = InputParser.ProcessDesc(instance.description)
        agg = np.hstack([price.reshape(1, -1), c_code.reshape(1, -1), desc_embed.reshape(1, -1)])
        return torch.tensor(agg)

    @staticmethod
    def Parse(instances: List[Instance]):
        if not InputParser.initialized: InputParser.Initialize()
        input_tensors = [InputParser.process_input(instance) for instance in instances]
        inputs = torch.cat(input_tensors, dim=0).to(torch.float32)
        return inputs

    
"""
    The following function processes the input instances
"""

def ParseInputInstances(request: PredictionRequest):
    return InputParser.Parse(request.instances)