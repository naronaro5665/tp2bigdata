# task3.py  (مستقل، Avazu)
import os, time, threading
import pandas as pd
import psutil
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

FILE, COL, RESULTS = "train.csv", "click", "results.csv"
METHOD = "Dask DataFrame"

peak = 0
stop = False


def monitor():
    global peak
    proc = psutil.Process(os.getpid())
    while not stop:
        rss = proc.memory_info().rss
        for c in proc.children(recursive=True):
            try:
                rss += c.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        peak = max(peak, rss)
        time.sleep(0.2)


def save_result(time_s, status):
    row = pd.DataFrame([dict(method=METHOD, time_s=round(time_s, 2),
                             peak_ram_gb=round(peak / 1e9, 3), status=status)])
    if os.path.exists(RESULTS):
        old = pd.read_csv(RESULTS)
        row = pd.concat([old[old.method != METHOD], row])
    row.to_csv(RESULTS, index=False)


if __name__ == "__main__":
    print(f"=== Task 3: Dask === (CPU cores: {os.cpu_count()})\n")

    # ---------- (أ) Lazy Evaluation ----------
    t0 = time.time()
    ddf = dd.read_csv(FILE, usecols=[COL, "hour", "C1"],
                      dtype={COL: "int8", "hour": "int32", "C1": "int16"},
                      blocksize="64MB")
    expr = ddf[COL].mean()
    print(f"Lazy phase: {time.time()-t0:.3f}s (لم تُقرأ أي بيانات بعد)")
    print("npartitions:", ddf.npartitions)
    print("Type before compute:", type(expr))
    print("Value before compute:", expr, "\n")

    # ---------- (ب) رسم الـ graph ----------
    try:
        ddf.partitions[:4][COL].mean().visualize(filename="dask_graph.png")
        print("Saved: dask_graph.png\n")
    except Exception as e:
        print("visualize skipped (ثبّت graphviz):", str(e)[:100], "\n")

    # ---------- (ج) غير محسَّن: قراءتان منفصلتان ----------
    print("--- Not optimized (2 separate computes) ---")
    t0 = time.time()
    with ProgressBar():
        m = ddf[COL].mean().compute()
        s = ddf[COL].sum().compute()
    t_not_opt = time.time() - t0
    print(f"mean={m:.4f} | sum={s:,.0f} | time={t_not_opt:.1f}s\n")

    # ---------- (د) محسَّن: عمود واحد + compute واحد ----------
    print("--- Optimized (1 column + single compute) ---")
    ddf2 = dd.read_csv(FILE, usecols=[COL], dtype={COL: "int8"},
                       blocksize="64MB")
    threading.Thread(target=monitor, daemon=True).start()
    t0 = time.time()
    with ProgressBar():
        m2, s2, cnt = dask.compute(
            ddf2[COL].mean(), ddf2[COL].sum(), ddf2[COL].count()
        )
    t_opt = time.time() - t0
    stop = True

    print(f"mean={m2:.4f} | sum={s2:,.0f} | rows={cnt:,}")
    print(f"time={t_opt:.1f}s | peak RAM={peak/1e9:.2f} GB")
    print(f"Speedup (not optimized / optimized): {t_not_opt/t_opt:.2f}x")

    save_result(t_opt, "OK")