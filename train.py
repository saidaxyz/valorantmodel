import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Загружаем данные
data_path = "data/preprocessed_data.csv"
df = pd.read_csv(data_path)

# Выбираем признаки (features) и целевую переменную (target)
features = [
    "Rating_A", "Average Combat Score_A", "Kills - Deaths (KD)_A", "Headshot %_A", "First Kills_A",
    "Rating_B", "Average Combat Score_B", "Kills - Deaths (KD)_B", "Headshot %_B", "First Kills_B"
]
X = df[features]
y = df["Winner"]

# Проверяем NaN
print("Проверяем пропущенные значения перед обработкой:")
print(X.isnull().sum())

# Заполняем NaN средними значениями
X = X.fillna(X.mean())

print("✅ Пропущенные значения заполнены!")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализируем MLflow
mlflow.set_experiment("Valorant Match Prediction")

with mlflow.start_run():
    # Выбираем и обучаем модель (логистическая регрессия)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Делаем предсказания
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Логируем параметры и метрики в MLflow
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_metric("accuracy", accuracy)

    # Сохраняем модель
    os.makedirs("models", exist_ok=True)
    model_path = "models/logistic_regression_model"
    mlflow.sklearn.save_model(model, model_path)

    print(f"✅ Модель обучена! Точность: {accuracy:.4f}")
    print(f"📁 Модель сохранена в: {model_path}")