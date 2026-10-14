import os
import time
import threading

import pandas as pd
import psutil

FILE = r"train.csv"
CHUNK_SIZE = 100_000

process = psutil.Process(os.getpid())
peak = 0
stop = False


def monitor():
    global peak
    while not stop:
        peak = max(peak, process.memory_info().rss / 1024**3)
        time.sleep(0.2)


if __name__ == "__main__":
    print("=== Task 2: Pandas Chunking (100,000 rows) ===")

    threading.Thread(target=monitor, daemon=True).start()
    start = time.time()

    total_rows = 0
    total_clicks = 0
    n_chunks = 0

    for chunk in pd.read_csv(FILE, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
        total_clicks += int(chunk["click"].sum())
        n_chunks += 1

        if n_chunks % 50 == 0:
            print(f"  chunk {n_chunks} | rows so far: {total_rows:,}")

    stop = True

    print("Number of chunks:", n_chunks)
    print("Total rows:", total_rows)
    print("Total clicks:", total_clicks)
    print("Click rate:", round(total_clicks / total_rows, 4))
    print("Execution time:", round(time.time() - start, 2), "seconds")
    print("Peak memory:", round(peak, 2), "GB")