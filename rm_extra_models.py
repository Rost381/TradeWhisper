import os
import re

# Регулярное выражение для проверки файлов
pattern = re.compile(r"DOGE_USDT_\d+\.zip")

# Получаем текущую директорию
current_dir = os.getcwd()

# Проходим по всем поддиректориям вида models_*
for root, dirs, files in os.walk(current_dir):
    if os.path.basename(root).startswith("models_"):
        for file in files:
            if file.endswith(".zip") and pattern.fullmatch(file):
                file_path = os.path.join(root, file)
                print(f"Удаляется файл: {file_path}")
                os.remove(file_path)

print("Готово!")
