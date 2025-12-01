
'''
To start server:
uvicorn app_api:app --host 127.0.0.1 --port 5000 --reload



curl -X GET http://127.0.0.1:5000/health
curl -X GET http://127.0.0.1:5000/stats
curl -X POST http://127.0.0.1:5000/predict_model -H "Content-Type: application/json" -d "{\"Sex\": 0, \"Pclass\": 3, \"Age\": 22.0, \"Fare\": 7.2500}"
'''
from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd 
from pydantic import BaseModel 

import uvicorn

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

request_count = 0

class PredictionInput(BaseModel):
    Sex: int
    Pclass: int
    Age: float
    Fare: float

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    new_data = pd.DataFrame({
        "Sex":  [input_data.Sex],
        "Pclass": [input_data.Pclass],
        "Age": [input_data.Age],
        "Fare": [input_data.Fare]
    })

    prediction = model.predict(new_data)

    result = "Survived" if prediction[0] == 1 else "Not survived"

    return {"prediction": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)