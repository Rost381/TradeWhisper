import os
import shutil

# Убедимся, что есть директория tensorboard
TARGET_DIR = "tensorboard"
os.makedirs(TARGET_DIR, exist_ok=True)

# Получаем список всех поддиректорий в текущей директории
current_dir = os.getcwd()
dirs = [d for d in os.listdir(current_dir) if os.path.isdir(d) and d.startswith("tensorboard_")]

for dir_name in dirs:
    # Извлекаем <SUFFIX> из имени директории
    suffix = dir_name.split("tensorboard_")[-1]
    
    # Путь к новой директории
    new_path = os.path.join(TARGET_DIR, suffix)

    # Перемещаем директорию с переименованием
    shutil.move(dir_name, new_path)

    print(f"Перемещено: {dir_name} -> {new_path}")

print("Все директории успешно перемещены и переименованы!")
