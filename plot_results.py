
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)                      # works even if launched from another folder

RESULTS_CSV = "results.csv"
OUT_DIR = "tp02_charts"

TASKS = {
    "Standard Pandas": "task1.py",
    "Pandas Chunking": "task2.py",
    "Dask": "task3.py",
}
ORDER = list(TASKS)
COLORS = {"Standard Pandas": "#d9534f", "Pandas Chunking": "#5bc0de", "Dask": "#5cb85c"}
NUM_COLS = ("time_s", "memory_gb", "rows", "clicks")

# Values of the first run (used only if results.csv does not exist yet)
DEFAULTS = [
    dict(method="Standard Pandas", time_s=370.66, memory_gb=6.96,
         rows=40428967, clicks=6865066, status="OK"),
    dict(method="Pandas Chunking", time_s=344.85, memory_gb=0.16,
         rows=40428967, clicks=6865066, status="OK"),
    dict(method="Dask", time_s=245.71, memory_gb=0.38,
         rows=40428967, clicks=6865066, status="OK"),
]


# ---------------------------------------------------------------- run + parse
def _num(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1).replace(",", "")) if m else None


def parse(output):
    """Extract time, memory, rows and clicks from the printed output of a task."""
    memory = None
    for label in ("Peak memory", "Maximum observed process RAM",
                  "Process RAM after computation", "Memory after loading"):
        memory = _num(label + r":\s*([\d.]+)\s*GB", output)
        if memory is not None:
            break
    return dict(
        time_s=_num(r"Execution time:\s*([\d.]+)", output),
        memory_gb=memory,
        rows=_num(r"Total rows:\s*([\d,]+)", output, int),
        clicks=_num(r"Total clicks:\s*([\d,]+)", output, int),
    )


def run_script(script):
    """Run a task and print its output live. Returns (return_code, full_output)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    p = subprocess.Popen([sys.executable, script], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env=env, cwd=BASE_DIR)
    lines = []
    for line in p.stdout:
        print(line, end="")
        lines.append(line)
    p.wait()
    return p.returncode, "".join(lines)


def run_all():
    rows = []
    for method, script in TASKS.items():
        print(f"\n>>> Running {script} ({method}) ...")
        if not os.path.exists(script):
            print(f"!! {script} not found")
            rows.append(dict(method=method, status="Missing script"))
            continue
        code, out = run_script(script)
        if re.search(r"^MemoryError", out, re.MULTILINE):
            status = "MemoryError"
        else:
            status = "OK" if code == 0 else "Failed"
        rows.append(dict(method=method, status=status, **parse(out)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved: {RESULTS_CSV}")
    return df


# -------------------------------------------------------------------- compare
def normalize(df):
    """Accept old/new formats of results.csv and return one row per method, in ORDER."""
    df = df.rename(columns={"peak_ram_gb": "memory_gb"}).copy()
    df["method"] = df["method"].replace({"Dask DataFrame": "Dask"})
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = float("nan")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "status" not in df.columns:
        df["status"] = "OK"
    df = df.drop_duplicates("method", keep="last").set_index("method").reindex(ORDER)
    df["status"] = df["status"].fillna("Not run")
    return df


def fmt(value, spec, suffix=""):
    return "N/A" if pd.isna(value) else format(value, spec) + suffix


def compare(raw):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = normalize(raw)

    base = df.loc["Standard Pandas"]
    base_ok = base["status"] == "OK"
    out = df.copy()
    out["click_rate_%"] = out["clicks"] / out["rows"] * 100
    out["faster_than_pandas_%"] = (base["time_s"] - out["time_s"]) / base["time_s"] * 100
    out["speedup_x"] = base["time_s"] / out["time_s"]
    out["memory_less_than_pandas_x"] = base["memory_gb"] / out["memory_gb"]

    # a failed method (MemoryError, Failed, ...) or a failed baseline gives no valid comparison
    ratio_cols = ["faster_than_pandas_%", "speedup_x", "memory_less_than_pandas_x"]
    if not base_ok:
        out[ratio_cols] = float("nan")
    out.loc[out["status"] != "OK", ratio_cols] = float("nan")
    valid = out[out["status"] == "OK"]

    # 1) correctness check: all methods must give the same result
    same_rows = out["rows"].dropna().nunique() <= 1
    same_clicks = out["clicks"].dropna().nunique() <= 1

    # 2) print
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(out.round(2).to_string())
    print("\nSame rows count   :", same_rows)
    print("Same clicks count :", same_clicks)
    if not (same_rows and same_clicks):
        print("!! WARNING: the methods did not return the same result, check the outputs.")
    if not base_ok:
        print(f"Note: Standard Pandas status is '{base['status']}', "
              "so speedup and memory ratios vs Pandas are not computed.")
    if valid["time_s"].notna().any():
        print("Fastest           :", valid["time_s"].idxmin(), f"({valid['time_s'].min():.2f} s)")
    if valid["memory_gb"].notna().any():
        print("Lowest memory     :", valid["memory_gb"].idxmin(), f"({valid['memory_gb'].min():.2f} GB)")

    # 3) save csv
    out.round(4).to_csv(os.path.join(OUT_DIR, "comparison.csv"))

    # 4) charts (a failed/missing method is drawn as 0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    colors = [COLORS[m] for m in out.index]

    bars = ax[0].bar(out.index, out["time_s"].fillna(0), color=colors)
    ax[0].set_title("Execution time (s)")
    ax[0].bar_label(bars, fmt="%.1f s")

    bars = ax[1].bar(out.index, out["memory_gb"].fillna(0), color=colors)
    ax[1].set_title("Memory (GB)")
    ax[1].bar_label(bars, fmt="%.2f GB")

    for a in ax:
        a.tick_params(axis="x", rotation=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison.png"), dpi=200)
    plt.close()

    # 5) markdown table for the README
    lines = [
        "| Method | Time (s) | Memory (GB) | Rows | Clicks | Click rate | Faster than Pandas |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, r in out.iterrows():
        label = m if r["status"] == "OK" else f"{m} ({r['status']})"
        lines.append(
            f"| {label} | {fmt(r['time_s'], '.2f')} | {fmt(r['memory_gb'], '.2f')} | "
            f"{fmt(r['rows'], ',.0f')} | {fmt(r['clicks'], ',.0f')} | "
            f"{fmt(r['click_rate_%'], '.2f', '%')} | {fmt(r['faster_than_pandas_%'], '.1f', '%')} |"
        )
    with open(os.path.join(OUT_DIR, "results_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved in {OUT_DIR}/: comparison.csv, comparison.png, results_table.md")


# ----------------------------------------------------------------------- main
if __name__ == "__main__":
    if "run" in sys.argv[1:]:
        results = run_all()
    elif os.path.exists(RESULTS_CSV):
        results = pd.read_csv(RESULTS_CSV)
    else:
        results = pd.DataFrame(DEFAULTS)
        results.to_csv(RESULTS_CSV, index=False)
        print(f"{RESULTS_CSV} not found: created from the default values of the first run.")
    compare(results)