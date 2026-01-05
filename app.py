from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

class Input(BaseModel):
   department:object
   region:object
   education:object
   gender:object
   recruitment_channel:object
   previous_year_rating:float
   KPIs_met: int
   awards_won:int
   avg_training_score:int


class Output(BaseModel):
   is_promoted:int


@app.get("/")
def read_root():
   return {"Hello":"World"}

@app.post("/predict")
def pr(input:Input) -> Output:
   model = joblib.load("jobchg_pipeline_model.pkl")
   X_input = pd.DataFrame([input.dict()])
   prediction = model.predict(X_input)
   return Output(is_promoted=prediction[0])

