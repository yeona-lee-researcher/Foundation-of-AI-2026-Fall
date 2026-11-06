# Foundations of AI (DATA 37711) · Fall 2026 · Homework Code

Homework code for the course. Each homework lives in its own directory
(`homework_1/`, `homework_2/`, …) and is self-contained: it has its own
`README.md`, `requirements.txt`, data inputs, and a single entry point, so an
agent or reviewer can clone this repository and rerun the submitted code.

| Homework | Directory | Entry point | Status |
| --- | --- | --- | --- |
| Homework 2 | [`homework_2/`](homework_2/README.md) | `python run_all.py` | Submitted; full rerun verified from a clean clone |

## Quick start (Homework 2)

```bash
git clone https://github.com/yeona-lee-researcher/Foundation-of-AI-2026-Fall.git
cd Foundation-of-AI-2026-Fall/homework_2

python3.11 -m venv .venv           # Windows: py -3.11 -m venv .venv   (Python 3.11 or 3.12)
source .venv/bin/activate          # Windows Git Bash: source .venv/Scripts/activate
                                   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt    # ~5 min (PyTorch)

python run_all.py --smoke          # ~15 s: checks imports, data files and schemas (no training)
python run_all.py --problem all    # 17-25 min on CPU: reruns Problems 1, 2, 3 and the Problem 3 robustness check
```

Results are written to `homework_2/outputs/` (ignored by Git). See
[homework_2/README.md](homework_2/README.md) for what each problem produces,
the expected numbers, and how to check them against the submitted results.

## Repository conventions

- Python 3.11 or 3.12. Each homework pins its own dependencies in `requirements.txt`.
- No absolute paths: every script is run through the homework's `run_all.py`, which resolves paths relative to itself.
- Generated outputs, virtual environments and caches are git-ignored. The final submitted artifacts (prediction CSVs, result tables, report PDF) are committed under each homework's `submissions/`, `results/` and `docs/` folders.
- Text files are stored with LF line endings; data files are stored byte-for-byte (see `.gitattributes`).
