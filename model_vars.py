
import asyncio
import subprocess
from itertools import product
import os

suffix_letter = ["L", #Long
                 "R", #Reward multiplier
                 "V" # Version of indicators set
]
suffix_number = [5, # Параметры history_size, window_size, get_full_data
                 1,  
                 3,]

# Генерация диапазонов для каждого суффикса
ranges = [range(num + 1) for num in suffix_number]

# Генерация всех комбинаций
combinations = [
    "".join(f"{letter}{num}" for letter, num in zip(suffix_letter, combo))
    for combo in product(*ranges)
]

print(combinations)
print(len(combinations))
# Вывод результата
list_var = []
dir_count = 0
for combination in combinations:
    if os.path.exists(f"models_{combination}"):
        dir_count += 1
        list_var.append(combination)
    
print(dir_count)
print(list_var)
ds = set(combinations) - set(list_var)
print(sorted(list(ds)))



# # Генерация всех комбинаций
# combinations = [
#     "".join(f"{letter}{num}" for letter, num in zip(suffix_letter, combo))
#     for combo in product(*ranges)
# ]

# async def run_combination(combination):
#     """Запуск файла bk.py с параметром combination."""
#     process = await asyncio.create_subprocess_exec(
#         "python", "bk.py", combination,
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE
#     )
#     stdout, stderr = await process.communicate()

#     if stdout:
#         print(f"[{combination}] stdout: {stdout.decode().strip()}")
#     if stderr:
#         print(f"[{combination}] stderr: {stderr.decode().strip()}")

# async def main():
#     """Асинхронный запуск всех комбинаций."""
#     tasks = [run_combination(combination) for combination in combinations]
#     await asyncio.gather(*tasks)

# # Запуск основной программы
# if __name__ == "__main__":
#     asyncio.run(main())
