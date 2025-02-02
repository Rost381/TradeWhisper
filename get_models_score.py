import os
import re
import pandas as pd

# Регулярное выражение для поиска значений в имени файла
file_pattern = re.compile(r"DOGE_USDT_\d+_([-+]?\d+\.\d+)\.zip")

# Список для хранения данных
data = []

# Текущая директория
current_dir = os.getcwd()

# Проходим по всем поддиректориям вида models_<SUFFIX>
for root, dirs, files in os.walk(current_dir):
    base_dir = os.path.basename(root)
    if base_dir.startswith("models_"):
        suffix = base_dir[len("models_"):]
        for file in files:
            match = file_pattern.fullmatch(file)
            if match:
                value = float(match.group(1))
                data.append({"Suffix": suffix, "Value": value})

# Создаем DataFrame
df = pd.DataFrame(data)

# Сортируем по убыванию значения Value
df = df.sort_values(by="Value", ascending=False)

# Сохраняем DataFrame в файл
output_file = os.path.join(current_dir, "models_score.csv")
df.to_csv(output_file, index=False)

print(f"Данные сохранены в файл: {output_file}")
