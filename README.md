# Valorant Match Prediction

## 📌 Описание проекта
Этот проект использует машинное обучение для предсказания победителя матча в игре Valorant на основе статистики команд. 

## 🚀 Функционал
- Предобработка данных и их анализ
- Обучение модели классификации с использованием **Logistic Regression**
- Логирование метрик с помощью **MLflow**
- Развёртывание модели через **FastAPI**
- Обёртывание в **Docker**

## 📂 Структура проекта
```
valorantmodel/
│── data/                    # Данные (игнорируются в Git)
│── models/                  # Сохранённые модели
│── scripts/                 # Код проекта
│   ├── preprocess.py        # Предобработка данных
│   ├── train.py             # Обучение модели
│   ├── main.py              # API FastAPI
│── requirements.txt         # Зависимости проекта
│── Dockerfile               # Docker-конфигурация
│── README.md                # Описание проекта
```

## 🛠 Установка и запуск
### 1️⃣ Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2️⃣ Предобработка данных
```bash
python scripts/preprocess.py
```

### 3️⃣ Обучение модели
```bash
python scripts/train.py
```

### 4️⃣ Запуск API с FastAPI
```bash
uvicorn scripts.main:app --host 0.0.0.0 --port 8000
```

### 5️⃣ Запуск через Docker
```bash
docker build -t valorant-prediction .
docker run -p 8000:8000 valorant-prediction
```

## 📝 Использование API
После запуска API можно отправлять запросы на предсказание матча.
Пример запроса:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
    "Rating_A": 1.05,
    "Average_Combat_Score_A": 230,
    "Kills_Deaths_KD_A": 1.1,
    "Headshot_Percent_A": 25.4,
    "First_Kills_A": 5,
    "Rating_B": 0.95,
    "Average_Combat_Score_B": 210,
    "Kills_Deaths_KD_B": 1.0,
    "Headshot_Percent_B": 23.8,
    "First_Kills_B": 4
}'
```
Пример ответа:
```json
{
    "Predicted Winner": "Team A"
}
```

## 📊 Логирование с MLflow
Для отслеживания экспериментов с моделями используется **MLflow**.
Запуск MLflow:
```bash
mlflow ui
```
Затем открой в браузере: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 🏆 Авторы
- **Saida Smakova** 👩‍💻
