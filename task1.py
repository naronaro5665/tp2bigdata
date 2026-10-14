import os
import time
import threading

import pandas as pd
import psutil

FILE = r"train.csv"

process = psutil.Process(os.getpid())
peak = 0
stop = False


def monitor():
    
    global peak
    while not stop:
        peak = max(peak, process.memory_info().rss / 1024**3)
        time.sleep(0.2)


if __name__ == "__main__":
    print("=== Task 1: Standard Pandas ===")
    print("File size:", round(os.path.getsize(FILE) / 1024**3, 2), "GB")
    print("Total RAM:", round(psutil.virtual_memory().total / 1024**3, 1), "GB")
    print("Memory before loading:",
          round(process.memory_info().rss / 1024**3, 2), "GB")

    threading.Thread(target=monitor, daemon=True).start()
    start = time.time()

    try:
        df = pd.read_csv(FILE)                      
        total_rows = len(df)
        total_clicks = int(df["click"].sum())

        print("Dataset loaded successfully!")
        print("Shape:", df.shape)
        print("Total rows:", total_rows)
        print("Total clicks:", total_clicks)
        print("Click rate:", round(total_clicks / total_rows, 4))

    except MemoryError:
        print("MemoryError: Not enough memory!")

    except Exception as e:
        print("Error:", e)

    finally:
        stop = True
        print("Execution time:", round(time.time() - start, 2), "seconds")
        print("Memory after loading:",
              round(process.memory_info().rss / 1024**3, 2), "GB")
        print("Peak memory:", round(peak, 2), "GB")