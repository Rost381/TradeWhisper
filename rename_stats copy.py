import os

def rename_csv_files():
    # Get the current working directory
    current_dir = os.getcwd()

    # Iterate through all items in the current directory
    for item in os.listdir(current_dir):
        # Check if the item is a directory and starts with "stats_"
        if os.path.isdir(item) and item.startswith("stats_"):
            dir_path = os.path.join(current_dir, item)

            # Define file paths
            result_csv = os.path.join(dir_path, "result.csv")
            stat_csv = os.path.join(dir_path, "stat.csv")

            # Rename result.csv to result_default.csv if it exists
            if os.path.exists(result_csv):
                new_result_csv = os.path.join(dir_path, "result_default.csv")
                os.rename(result_csv, new_result_csv)
                print(f"Renamed: {result_csv} -> {new_result_csv}")

            # Rename stat.csv to stat_default.csv if it exists
            if os.path.exists(stat_csv):
                new_stat_csv = os.path.join(dir_path, "stat_default.csv")
                os.rename(stat_csv, new_stat_csv)
                print(f"Renamed: {stat_csv} -> {new_stat_csv}")

if __name__ == "__main__":
    rename_csv_files()
