import os
import re

# Регулярное выражение для поиска файлов вида DOGE_USDT_nn_NN.nn
FILE_PATTERN = re.compile(r"DOGE_USDT_\d+_([-+]?\d*\.\d+|\d+)")

# Получаем список всех поддиректорий в текущей директории
current_dir = os.getcwd()
model_dirs = [d for d in os.listdir(current_dir) if os.path.isdir(d) and d.startswith("models_")]

for model_dir in model_dirs:
    max_file = None
    max_value = float('-inf')

    # Ищем файлы, соответствующие шаблону
    for file_name in os.listdir(model_dir):
        match = FILE_PATTERN.match(file_name)
        if match:
            # Извлекаем значение NN.nn и преобразуем в число
            value = float(match.group(1))
            if value > max_value:
                max_value = value
                max_file = file_name

    # Если найден максимальный файл
    if max_file:
        # Путь к максимальному файлу
        max_file_path = os.path.join(model_dir, max_file)

        # Добавляем расширение .zip, если его нет
        if not max_file.endswith(".zip"):
            new_max_file_path = f"{max_file_path}.zip"
            os.rename(max_file_path, new_max_file_path)
            print(f"Переименован файл: {max_file} -> {new_max_file_path}")
        else:
            new_max_file_path = max_file_path

        # Удаляем остальные файлы, кроме максимального
        for file_name in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file_name)
            if file_path != new_max_file_path and FILE_PATTERN.match(file_name) and not file_name.endswith(".zip"):
                os.remove(file_path)
                print(f"Удалён файл: {file_path}")

print("Обработка всех директорий завершена!")