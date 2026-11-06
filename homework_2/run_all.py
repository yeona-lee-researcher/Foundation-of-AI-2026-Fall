"""Reproduce Homework 2 from a clean clone, without manual path edits.

Usage: python run_all.py --smoke | --problem 1 | --problem 2 | --problem 3 | --problem robustness | --problem figures | --problem all
The scripts in code/ run as they are; this wrapper only supplies input paths and the working directory.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# The solution scripts print Korean progress messages. Force UTF-8 on both ends so that
# piped/redirected output (e.g. an automated grader on Windows, cp949/cp1252) never crashes.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def smoke() -> None:
    """Validate inputs/versions without running training or modifying datasets."""
    import numpy as np
    import pandas as pd
    import torch
    import sklearn
    import matplotlib
    try:
        import pyarrow  # noqa: F401; required for pd.read_parquet
        arrow_status = "available"
    except ModuleNotFoundError:
        arrow_status = "MISSING: pip install -r requirements.txt"
    expected = {
        "train.csv": ["x1", "x2", "y"],
        "val.csv": ["x1", "x2", "y"],
        "test.csv": ["x1", "x2"],
    }
    for filename, columns in expected.items():
        path = DATA / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        frame = pd.read_csv(path)
        if frame.columns.tolist() != columns:
            raise ValueError(f"Unexpected columns in {filename}: {frame.columns.tolist()}")
        print(f"{filename}: {len(frame)} rows; columns={columns}")
    retail_path = DATA / "retail.csv"
    taxi_path = DATA / "taxi_trips.parquet"
    if not retail_path.exists() or not taxi_path.exists():
        raise FileNotFoundError("Both retail.csv and taxi_trips.parquet are required for full reproduction")
    retail_header = pd.read_csv(retail_path, nrows=2)
    for column in ["StockCode", "CustomerID", "Quantity"]:
        if column not in retail_header.columns:
            raise ValueError(f"Missing {column} in retail.csv")
    print(f"retail.csv: {retail_path.stat().st_size} bytes")
    print(f"taxi_trips.parquet: {taxi_path.stat().st_size} bytes; parquet reader={arrow_status}")
    print(f"Python: {sys.version.split()[0]}, NumPy: {np.__version__}, pandas: {pd.__version__}")
    print(f"PyTorch: {torch.__version__}, sklearn: {sklearn.__version__}, matplotlib: {matplotlib.__version__}")
    print("Input schema checks passed. This check does not train any model.")


def run_one(problem: str) -> None:
    # The Problem 3 robustness check writes next to the Problem 3 outputs.
    output_dir = OUTPUTS / {"robustness": "problem3", "figures": "report_figures"}.get(problem, f"problem{problem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_name = "run_log_robustness.txt" if problem == "robustness" else "run_log.txt"
    title = {"robustness": "Problem 3 robustness check", "figures": "Report Figures 1-4"}.get(problem, f"Problem {problem}")
    command = [sys.executable]
    if problem == "1":
        command += [str(ROOT / "code/problem1_taxi.py"), str(DATA / "taxi_trips.parquet")]
    elif problem == "2":
        command += [str(ROOT / "code/problem2_retail.py"), str(DATA / "retail.csv")]
    elif problem == "3":
        # The original Problem 3 script reads local CSV names. Copy three small
        # inputs into the run directory rather than requiring the user to edit paths.
        for filename in ("train.csv", "val.csv", "test.csv"):
            shutil.copy2(DATA / filename, output_dir / filename)
        command += [str(ROOT / "code/problem3_rks_mlp.py")]
    elif problem == "robustness":
        for filename in ("train.csv", "val.csv"):
            shutil.copy2(DATA / filename, output_dir / filename)
        command += [str(ROOT / "code/problem3_robustness.py")]
    elif problem == "figures":
        # Redraws report Figures 1-4 from data/ and the committed results/ tables.
        command += [str(ROOT / "code/report_figures.py"), str(ROOT)]
    else:
        raise ValueError(problem)
    print(f"\n==== {title} ====", flush=True)
    print("Running:", " ".join(command), flush=True)
    with (output_dir / log_name).open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=output_dir, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"{title} failed (exit={return_code}); see {output_dir / log_name}")
    if problem == "3":
        import numpy as np
        for filename in ("rks_test_predictions.csv", "mlp_test_predictions.csv"):
            preds = np.loadtxt(output_dir / filename)
            expected = np.loadtxt(DATA / "test.csv", delimiter=",", skiprows=1).shape[0]
            if preds.shape != (expected,) or not np.isfinite(preds).all():
                raise AssertionError(f"Unexpected prediction shape/values in {filename}: {preds.shape}")
        print("Problem 3 predictions verified: both files match unlabeled test row count.")
    print(f"{title} complete. Outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Check imports and input files only")
    parser.add_argument("--problem", choices=["1", "2", "3", "robustness", "figures", "all"],
                        help="all = Problems 1, 2, 3, the Problem 3 robustness check and report Figures 1-4")
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.problem:
        for problem in (["1", "2", "3", "robustness", "figures"] if args.problem == "all" else [args.problem]):
            run_one(problem)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
