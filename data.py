
import os

file_path = r"train.csv"
print("avawu click-though rate")
if os.path.exists(file_path):
    size_bytes = os.path.getsize(file_path)

    print("File size:")
    print(f"{size_bytes / (1024**2):.2f} MB")
    print(f"{size_bytes / (1024**3):.2f} GB")
else:
    print("File not found! Check the file path.")
