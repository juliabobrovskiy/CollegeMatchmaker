import os
import numpy as np
from pydantic import BaseModel, Field
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime


#with open('./src/model_pipeline.pkl', 'rb') as f:
    #model_predictor = pickle.load(f)

dir_name = os.path.dirname(__file__)
model = joblib.load(dir_name + '/../model_pipeline.pkl')

app = FastAPI()

#joblib.load(model_path)


#
class HousingData(BaseModel):
    MedInc: float = Field(..., gt=0)
    HouseAge: float = Field(..., gt=0)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., gt=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)

class PredictionInput(BaseModel):
    data: List[HousingData]

class PredictionOutput(BaseModel):
    prediction: Optional[List[float]] = None





@app.get("/hello")
async def hello(name: str):
 #   if not name:
#        raise HTTPException(status_code=404, detail="Name parameter is required")
#The server cannot find the requested resource. In the browser, this means the URL is not recognized. In an API, this can also mean that the endpoint is valid but the resource itself does not exist..
    detail = f"hello {name}"
    return {"message": detail}


@app.get("/")
def not_implemented():
    raise HTTPException(status_code=501, detail="Not Implemented")


# The following endpoints are handled automatically by FastAPI

#@app.get("/docs")
#async def get_docs():
#    return app.openapi(), 200

#@app.get("/openapi.json")
#async def get_openapi():
#    return get_openapi(title="FastAPI Application", version="3", routes=app.routes)


@app.post("/predict", response_model = PredictionOutput)
async def predict(input_data: PredictionInput):
#    if model_predictor is None:
#        return {"error":"model not found"}

    x = np.array([list(item.dict().values()) for item in input_data.data])
    try:
        prediction = model.predict(x)
        print(f'Prediction: {prediction}')
        print(type(prediction))
        prediction_list = prediction.tolist()
        print(f'prediction list: {prediction_list}')
        print(type(prediction_list))
        return PreditionOutput(prediction = prediction_list)

    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health():
    current_time = datetime.now().isoformat()
    return {"time": current_time}
