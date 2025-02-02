
import asyncio
import os
import subprocess
import time
from itertools import product
from datetime import timedelta


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


list_var = []
dir_count = 0
for combination in combinations:
    if os.path.exists(f"models_{combination}"):
        if combination[:2] != "L0":
            dir_count += 1
            list_var.append(combination)
    
# print(dir_count)
# print(list_var)
ds = list(set(combinations) - set(list_var))
# combinations = ds
combinations = ["L0R1V1",
                "L1R0V1",
                "L2R1V1",
                "L2R1V3",
                "L4R0V0",
                "L1R0V3",
]
print(f"{combinations=}")



async def run_combination(combination):
    """Запуск Main_wraper.py с параметром combination."""
    process = await asyncio.create_subprocess_exec(
        "python", "Main_wraper.py", combination,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if stdout:
        print(f"[{combination}] stdout: {stdout.decode().strip()}")
    if stderr:
        print(f"[{combination}] stderr: {stderr.decode().strip()}")

async def main():
    """Асинхронный запуск всех комбинаций."""
    tasks = [run_combination(combination) for combination in combinations]
    await asyncio.gather(*tasks)

# Запуск основной программы
if __name__ == "__main__":
    print("Main running....")
# ---  Timer
    start_time = time.time()    
    asyncio.run(main())
    print(f"Расчет для: {combinations}")
    print(f"Количество: {len(combinations)}")
    # ---  Timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Script completed in: {timedelta(seconds=elapsed_time)}")      