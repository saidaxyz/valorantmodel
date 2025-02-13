from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import os

# Загружаем модель
model_path = "models/logistic_regression_model"
model = mlflow.sklearn.load_model(model_path)

# Создаем FastAPI приложение
app = FastAPI()


# Определяем входные данные
class MatchData(BaseModel):
    Rating_A: float
    Average_Combat_Score_A: float
    Kills_Deaths_KD_A: float
    Headshot_Percent_A: float
    First_Kills_A: float
    Rating_B: float
    Average_Combat_Score_B: float
    Kills_Deaths_KD_B: float
    Headshot_Percent_B: float
    First_Kills_B: float


@app.post("/predict")
def predict_winner(data: MatchData):
    # Подготовка данных
    input_data = np.array([
        data.Rating_A, data.Average_Combat_Score_A, data.Kills_Deaths_KD_A,
        data.Headshot_Percent_A, data.First_Kills_A, data.Rating_B,
        data.Average_Combat_Score_B, data.Kills_Deaths_KD_B,
        data.Headshot_Percent_B, data.First_Kills_B
    ]).reshape(1, -1)

    # Предсказание модели
    prediction = model.predict(input_data)
    winner = "Team A" if prediction[0] == 1 else "Team B"

    return {"Predicted Winner": winner}
