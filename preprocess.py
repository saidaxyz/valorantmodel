import pandas as pd
import os

# Пути к данным
data_path = "data/vct_2021/matches"
overview_path = os.path.join(data_path, "overview.csv")
maps_scores_path = os.path.join(data_path, "maps_scores.csv")

# Загружаем данные
overview_df = pd.read_csv(overview_path)
maps_scores_df = pd.read_csv(maps_scores_path)

# Выбираем нужные столбцы из overview
player_stats = overview_df[
    ["Match Name", "Team", "Rating", "Average Combat Score", "Kills - Deaths (KD)", "Headshot %", "First Kills"]
]

# Конвертируем Headshot % в числовой формат
player_stats["Headshot %"] = player_stats["Headshot %"].str.replace("%", "").astype(float)

# Выбираем только числовые признаки
numeric_columns = ["Rating", "Average Combat Score", "Kills - Deaths (KD)", "Headshot %", "First Kills"]
team_stats = player_stats.groupby(["Match Name", "Team"])[numeric_columns].mean().reset_index()

# Разделяем данные по командам
team_A_stats = team_stats.rename(columns=lambda x: x + "_A" if x not in ["Match Name", "Team"] else x)
team_B_stats = team_stats.rename(columns=lambda x: x + "_B" if x not in ["Match Name", "Team"] else x)

# Объединяем с результатами матчей
dataset = maps_scores_df.merge(team_A_stats, left_on=["Match Name", "Team A"], right_on=["Match Name", "Team"])
dataset = dataset.merge(team_B_stats, left_on=["Match Name", "Team B"], right_on=["Match Name", "Team"])

# Проверяем колонки
print("Колонки в dataset перед удалением:", dataset.columns)

# Удаляем только существующие колонки
columns_to_drop = [col for col in ["Team_A", "Team_B"] if col in dataset.columns]
dataset.drop(columns=columns_to_drop, inplace=True)

# Создаем целевую переменную (1 - победа Team A, 0 - победа Team B)
dataset["Winner"] = (dataset["Team A Score"] > dataset["Team B Score"]).astype(int)

# Сохраняем подготовленный датасет
os.makedirs("data", exist_ok=True)
dataset.to_csv("data/preprocessed_data.csv", index=False)

print("Данные успешно обработаны и сохранены в data/preprocessed_data.csv")
