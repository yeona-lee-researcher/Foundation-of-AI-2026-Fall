# DATA 37711 · Homework 2

This directory holds the **three problems of Homework 2** (not three separate homeworks):

| Problem | Topic | Script | Input |
| --- | --- | --- | --- |
| 1 (a–e) | Heteroscedastic Gaussian MLP for taxi trip duration; P(trip < 45 min) | `code/problem1_taxi.py` | `data/taxi_trips.parquet` |
| 2 (a–e) | Retail quantity: target encoding + ridge vs. embedding MLPs | `code/problem2_retail.py` | `data/retail.csv` |
| 3 (a–d) | Random Kitchen Sinks + ridge vs. one-hidden-layer MLP (2-D regression) | `code/problem3_rks_mlp.py` | `data/train.csv`, `data/val.csv`, `data/test.csv` |
| 3, robustness check | Wider grids and repeated seeds for the 3(b)/3(c) selections; why RKS does better in 3(d) | `code/problem3_robustness.py` | `data/train.csv`, `data/val.csv` |

`run_all.py` is the single entry point. It runs each script with the right input paths and working directory, so **no file paths need to be edited**.

### Grading checklist (every sub-question of the handout)

The written answers are in **`docs/HW2_Revised_Submission.pdf`** (9 pages). The same text is in plain LaTeX at `docs/latex/main.tex`, which is easier to read programmatically. The "Evidence" column names the file, or the line of a `run_all.py` log (`outputs/problem<N>/run_log.txt`), that reproduces each number in the report.

| Item | Handout asks for | Answer in report | Evidence in this repo |
| --- | --- | --- | --- |
| **1(a)** | Link function from g(x) ∈ R² to f_μ, f_σ | p.1 §1(a): μ = g₁, σ = softplus(g₂) + 10⁻³ | `code/problem1_taxi.py`, `predict()` |
| **1(b)** | Loss under Y\|X ~ N(f_μ, f_σ²) | p.1 §1(b): Gaussian negative log-likelihood | `code/problem1_taxi.py`, `nll()` |
| **1(c)** | PyTorch/JAX MLP with mean and std output; baseline(s); train/val/test split | p.1–2 §1(c): 7→64→2 ReLU MLP, constant-Normal baseline, 366,487 / 78,533 / 78,534 split; up to 300 epochs with early stopping on validation NLL (patience 30), best epoch 169 | Problem 1 log lines `split train/validation/test …`, `epoch <n> validation NLL …` (one per epoch), `epochs run …`, `epoch, validation NLL …` |
| **1(d)** | How to get P(trip < 45 min) | p.2 §1(d): Φ((45 − μ)/σ); Figure 1 | `code/problem1_taxi.py`, `report()` |
| **1(e)** | Proper scoring rule + accuracy at threshold 0.5 on the test set | p.2 §1(e) table: Brier 0.003343 vs. baseline 0.004495; accuracy 99.6065 % vs. 99.5505 %; 121 of 353 long trips flagged | Problem 1 log lines `constant …`, `MLP …` (reproduce exactly) |
| **2(a)** | Cardinality of both features; is one-hot feasible; train/val/test split | p.3 §2(a): 3,665 StockCode, 4,339 CustomerID; dense one-hot ≈ 12.74 GB; 70/15/15 split; Quantity quantiles; Figure 2 | Problem 2 log lines starting `(a)` (row count, unique values, one-hot size, quantiles) and `분할된 행 수` ("split sizes") |
| **2(b)** | Target encoding + model with suitable loss; test evaluation | p.3–4 §2(b): out-of-fold target encoding, log1p target, ridge with α chosen on validation (and why α = 0.1 at the grid edge does not matter); Figure 3; test log-MSE 0.409972 | Problem 2 log lines `alpha=… / validation log-MSE=…` and `선택된 alpha` ("selected alpha"); `results/hw2_ridge_alpha_sweep.csv`; `results/hw2_retail_results.csv` |
| **2(c)** | NN with an embedding layer per feature; test evaluation; compare with 2(b) | p.4 §2(c): two 8-dim embeddings → 32-unit MLP, early stopping (best epoch 98, stopped at 108); test log-MSE 0.281836 | Problem 2 log line `hybrid=False, epochs=…, best epoch=…`; `results/hw2_retail_results.csv` row `two embeddings + MLP` |
| **2(d)** | NN using embeddings and target-encoded features together; compare | p.4 §2(d): 18-dim input (best epoch 61, stopped at 71); test log-MSE 0.283321 | Problem 2 log line `hybrid=True, …`; `results/hw2_retail_results.csv` row `target encoding + embeddings + MLP` |
| **2(e)** | Intuition for the embedding layer; findings on the test set | p.4–5 §2(e): what an embedding is, four-model test table, Figure 4, test log-MSE split by how often each ID appears in training | `results/hw2_retail_results.csv`; `results/hw2_retail_by_frequency.csv`; Problem 2 log lines `(e) test log-MSE by how often the ID appears in train` and `(e) 가장 큰 test Quantity` ("largest test Quantity" and its share of the baseline's squared error) |
| **3(a)** | Visualize the training data | p.6 §3(a): Figure 5 | `docs/latex/figures/3a_training_scatter.png` |
| **3(b)** | RKS + ridge; sweep one hyperparameter; **two** plots (train MSE, val MSE); select on validation | p.6–7 §3(b): sweep over D ∈ {20,…,320}, table, Figures 6a/6b, D = 20 | `docs/latex/figures/3b_rks_training_mse.png`, `3b_rks_validation_mse.png`; Problem 3 log lines `RKS D=…` |
| **3(c)** | One-hidden-layer MLP; sweep one hyperparameter; two plots; select on validation | p.7 §3(c): hidden width ∈ {16, 32, 64, 128}, table, Figures 7a/7b, width = 128 | `docs/latex/figures/3c_mlp_training_mse.png`, `3c_mlp_validation_mse.png`; Problem 3 log lines `MLP width=…` |
| **3(b), 3(c)** | (Grid edges) Is the selected D / width really the best? | p.8–9 "Robustness of the selections": wider grids (D 8–320, width 8–1024) with 20 / 5 repeated seeds; Figure 9 | `outputs/problem3/run_log_robustness.txt`; `outputs/problem3/problem3_outputs/3e_*.csv`, `docs/latex/figures/3e_robustness.png` |
| **3(d)** | Refit both; report validation MSE; discuss which is better **and why** | p.7–8 §3(d): RKS 0.036124 vs. MLP 0.038674; p.9 "Why RKS does slightly better": RBF kernel-ridge limit, paired standard error, smoothness argument, effect of the random draw | `results/p3_validation_comparison.csv`; robustness log lines `RBF kernel ridge …`, `standard error of a 200-point validation MSE …`, `MLP(128, retrained with the submitted settings) minus RKS(D=20) …` |
| **3(d)** | Submit `rks_test_predictions.csv`, `mlp_test_predictions.csv`: one prediction per row, in `test.csv` order | p.7 §3(d) | `submissions/` (200 rows each, no header; `run_all.py` checks the row count) |
| **3(d)** | Visualize both fitted surfaces; compare with 3(a) | p.8: Figure 8 and discussion | `docs/latex/figures/3d_three_panel_comparison.png` |
| Code | "Make sure to attach your code" | p.9 "Reproducibility notes" | `code/problem1_taxi.py`, `code/problem2_retail.py`, `code/problem3_rks_mlp.py`, `code/problem3_robustness.py`, `code/report_figures.py` |
| Course rule | An agent can clone the repo and run the code | n/a | This README (sections 1–4), `requirements.txt`, `run_all.py` |

Some log lines of Problems 2 and 3 are printed in Korean (the code is annotated in Korean); English glosses are given in quotes above. Every number in those lines is labeled by the surrounding text, e.g. `RKS D= 20 | train MSE=… | val MSE=…`.

**Figures.** Figures 5–8 (Problem 3) are written by `code/problem3_rks_mlp.py` and are regenerated by `run_all.py --problem 3`; Figure 9 is written by `code/problem3_robustness.py` (`run_all.py --problem robustness`). Figures 1–4 are written by `code/report_figures.py` (`run_all.py --problem figures`): Figure 1 is an illustrative sketch of two Normal densities (not data), Figure 2 is drawn from `data/retail.csv`, and Figures 3–4 from `results/hw2_ridge_alpha_sweep.csv` and `results/hw2_retail_results.csv`.

---

## 1. Setup

**Requirements:** Python **3.11 or 3.12**. CPU is enough; a CUDA GPU is used automatically by Problems 2 and 3 if available. About 1 GB of disk for the virtual environment (PyTorch).

Create the virtual environment with a supported interpreter. If plain `python` is a different version, select one explicitly:

```bash
# from the repository root
cd homework_2
python3.11 -m venv .venv         # macOS / Linux   (or python3.12)
py -3.11 -m venv .venv           # Windows         (or py -3.12)
```

Activate it:

```bash
source .venv/bin/activate        # macOS / Linux
source .venv/Scripts/activate    # Windows, Git Bash
.venv\Scripts\activate.bat       # Windows, cmd
.venv\Scripts\Activate.ps1       # Windows, PowerShell (if blocked: Set-ExecutionPolicy -Scope Process Bypass)
```

Alternatively, skip activation and call the venv interpreter directly, e.g. `.venv/bin/python run_all.py --smoke` or `.venv\Scripts\python run_all.py --smoke`.

Install dependencies (~5 min, mostly PyTorch):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Tip (Linux, no GPU):** the default PyTorch wheel on Linux bundles CUDA libraries (several GB). For a smaller CPU-only install, run
> `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` **before** `pip install -r requirements.txt`.

`requirements.txt`:

```
numpy>=1.26,<3
pandas>=2.1,<3
scikit-learn>=1.4,<2
matplotlib>=3.8,<4
torch>=2.3,<3
pyarrow>=15,<24
```

(`pyarrow` is the Parquet reader that Problem 1 needs for `pandas.read_parquet`.)

## 2. Run

All commands are run from `homework_2/` with the virtual environment active.

| Command | What it does | Time (CPU, measured) |
| --- | --- | --- |
| `python run_all.py --smoke` | Checks imports, that all data files exist, the column names of `train.csv`, `val.csv`, `test.csv` and `retail.csv`, and that a Parquet reader is available. Trains nothing. | ~15 s |
| `python run_all.py --problem 1` | Problem 1: trains the taxi MLP (up to 300 epochs, early stopping; stops at epoch 199) and prints test metrics | ~65 s |
| `python run_all.py --problem 2` | Problem 2: ridge sweep + two embedding MLPs (up to 100 epochs each) | 14–20 min |
| `python run_all.py --problem 3` | Problem 3: RKS sweep, MLP width sweep, final models, plots, test predictions | 30–40 s |
| `python run_all.py --problem robustness` | Problem 3 robustness check: wider D / width grids with repeated seeds, RBF kernel ridge, paired standard error, Figure 9 | ~3 min |
| `python run_all.py --problem figures` | Redraws report Figures 1–4 from `data/` and `results/` into `outputs/report_figures/` | ~10 s |
| `python run_all.py --problem all` | Problems 1 → 2 → 3 → robustness check → figures in sequence | 17–25 min |

Timings were measured on a Windows 11 laptop, CPU only. `run_all.py` streams each script's console output to the terminal and also saves it to `outputs/problem<N>/run_log.txt` (the robustness check writes `outputs/problem3/run_log_robustness.txt`). It exits with a non-zero code if a script fails. For Problem 3 it also checks that both prediction files have exactly one finite value per row of `test.csv`.

> The scripts print partly Korean progress messages (the code is annotated line by line in Korean). `run_all.py` forces UTF-8 for both its own output and the child scripts, so this works in any console or when output is piped/redirected (no `PYTHONIOENCODING` setup needed). If you run a script in `code/` directly on Windows and see garbled or encoding errors, set `PYTHONIOENCODING=utf-8` first.

## 3. Outputs

Everything generated goes to `outputs/`, which is git-ignored:

```
outputs/
├── problem1/run_log.txt                    # split sizes, validation curve, test metrics (see "Problem 1 output format" below)
├── problem2/
│   ├── hw2_retail_results.csv              # test log-MSE, RMSLE, MAE, RMSE for the four models
│   ├── hw2_ridge_alpha_sweep.csv           # validation log-MSE per ridge alpha
│   ├── hw2_retail_by_frequency.csv         # 2(e): test log-MSE by how often each ID appears in train
│   └── run_log.txt
├── report_figures/                         # Figures 1-4 of the report (--problem figures)
└── problem3/
    ├── rks_test_predictions.csv            # 200 predictions, one per line, no header, same order as test.csv
    ├── mlp_test_predictions.csv            # same format
    ├── run_log.txt
    ├── run_log_robustness.txt               # robustness check (--problem robustness)
    └── problem3_outputs/
        ├── 3a_training_scatter.png
        ├── 3b_rks_sweep.csv, 3b_rks_training_mse.png, 3b_rks_validation_mse.png
        ├── 3c_mlp_sweep.csv, 3c_mlp_history_width_{16,32,64,128}.csv,
        │   3c_mlp_training_mse.png, 3c_mlp_validation_mse.png
        ├── 3d_model_comparison.csv, 3d_final_mlp_learning_curve.csv, 3d_mlp_learning_curve.png,
        │   3d_rks_surface.png, 3d_mlp_surface.png, 3d_three_panel_comparison.png
        ├── 3_submittable_english_draft.txt
        └── 3e_rks_extended_D.csv, 3e_mlp_extended_width.csv, 3e_mlp_extended_width_runs.csv,
            3e_robustness.png                # robustness check
```

**Problem 1 output format.** `problem1_taxi.py` prints:

```
split train/validation/test <n_train> <n_val> <n_test>
test trips >= 45 min <count>
epoch <n> validation NLL <value>        (one line per epoch)
epochs run <number of epochs trained before early stopping>
model test_NLL Brier accuracy late_hits
constant <test NLL> <Brier> <accuracy> <late_hits>
MLP <test NLL> <Brier> <accuracy> <late_hits>
epoch, validation NLL <chosen epoch> <best validation NLL>
```

`accuracy` is the fraction of test trips classified correctly by thresholding P(trip < 45 min) at 0.5. `late_hits` counts test trips that actually took ≥ 45 min and that the model also classified as ≥ 45 min.

(`run_all.py` also copies `train.csv` / `val.csv` / `test.csv` into `outputs/problem3/`, because the Problem 3 script reads its inputs from its working directory.)

## 4. Expected results (submitted vs. clean-clone rerun)

The submitted numbers are in `results/`, `submissions/` and the report. The "Rerun" columns come from a full `run_all.py --problem all` in a fresh Python 3.11.9 venv (numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, torch 2.15.0+cpu, pyarrow 23.0.1).

**Problem 1** (test split; Brier and accuracy for the event "trip < 45 min"). The rerun matches the submission exactly.

| Model | Test NLL | Brier | Accuracy |
| --- | --- | --- | --- |
| Constant Normal baseline | 3.476824 | 0.004495 | 99.5505 % |
| Heteroscedastic MLP (best val. epoch 169 of 199) | 2.706100 | 0.003343 | 99.6065 % |

The MLP flags 121 of the 353 test trips that took ≥ 45 min (`late_hits`); the baseline flags none.

**Problem 2** (test split; lower is better). Ridge and the baseline match exactly. The MLPs differ in the 4th decimal place across PyTorch versions; the ranking is unchanged.

| Model | log-MSE submitted | log-MSE rerun | MAE submitted | MAE rerun |
| --- | --- | --- | --- | --- |
| constant baseline | 1.009571 | 1.009571 | 11.6955 | 11.6955 |
| target encoding + ridge (α = 0.1) | 0.409972 | 0.409972 | 8.5201 | 8.5201 |
| two embeddings + MLP | **0.281836** | **0.282086** | **7.3757** | **7.3701** |
| target encoding + embeddings + MLP | 0.283321 | 0.283844 | 7.4918 | 7.4860 |

**Problem 3** (selected on the validation split; `test.csv` has no labels, so there is no test MSE).

| Model | Selected | Val. MSE submitted | Val. MSE rerun |
| --- | --- | --- | --- |
| RKS + ridge (σ = 1, λ = 0.01) | D = 20 | **0.036124** | **0.036124** |
| One-hidden-layer MLP (Adam, lr 0.002, early stopping) | width = 128 | 0.038674 | 0.038685 |

Test predictions: the rerun `rks_test_predictions.csv` has exactly the same 200 values as `submissions/rks_test_predictions.csv`. The rerun `mlp_test_predictions.csv` differs from the submitted file by at most 0.0023 (mean 0.0004). Compare values, not bytes (the MLP file differs slightly by design; both files use LF line endings).

**Problem 3 robustness check** (`--problem robustness`; validation MSE, lower is better). RKS numbers are deterministic.

| Setting | Seed as submitted (42) | Mean over repeated seeds (sd): RKS = 20 draws other than 42; MLP = 5 seeds including 42 |
| --- | --- | --- |
| RKS D = 18 / 20 / 24 / 40 | 0.0892 / 0.0361 / 0.0349 / 0.0366 | 0.194 / 0.140 (0.137) / 0.078 / 0.0380 (0.0013) — 20 random-feature draws |
| MLP width 128 / 256 / 1024 | 0.0387 / 0.0381 / 0.0384 | 0.0383 / 0.0381 / 0.0382 (sd ≈ 0.0005) — 5 initialization seeds |
| RBF kernel ridge (D → ∞ limit of RKS) | 0.0359 | — |
| MLP(128) − RKS(D = 20), paired | 0.0026 ± 0.0021 (standard error) | — |

To compare a rerun yourself:

```bash
python -c "import numpy as np; [print(m, np.abs(np.loadtxt(f'outputs/problem3/{m}_test_predictions.csv')-np.loadtxt(f'submissions/{m}_test_predictions.csv')).max()) for m in ('rks','mlp')]"
```

**Why small differences appear:** all seeds are fixed (NumPy, PyTorch, the data split, the RKS random features). Ridge/RKS solutions are closed-form and reproduce exactly. MLP training goes through PyTorch kernels whose floating-point summation order can change between PyTorch versions, BLAS libraries, and CPU vs. GPU, so the MLP numbers can move slightly.

## 5. Submitted artifacts

| Path | Contents |
| --- | --- |
| `submissions/rks_test_predictions.csv` | Final Problem 3 RKS predictions for `test.csv` (200 lines, no header, same row order) |
| `submissions/mlp_test_predictions.csv` | Final Problem 3 MLP predictions (same format) |
| `results/hw2_retail_results.csv` | Problem 2 final test metrics table |
| `results/p3_validation_comparison.csv` | Problem 3 final train/validation MSE table |
| `results/hw2_ridge_alpha_sweep.csv` | Problem 2(b) validation log-MSE for each ridge α (input of Figure 3) |
| `results/hw2_retail_by_frequency.csv` | Problem 2(e) test log-MSE by how often the product / customer ID appears in training (from the reproducibility rerun) |
| `docs/HW2_Revised_Submission.pdf` | Written report (9 pages) |
| `docs/latex/` | LaTeX source of the report (`main.tex` + `figures/`); build with XeLaTeX, see [docs/latex/README.md](docs/latex/README.md) |
| `MANIFEST_SHA256.txt` | SHA-256 of every file in this directory (check with `sha256sum -c MANIFEST_SHA256.txt`) |

## 6. Study notebooks (optional)

The notebooks are line-by-line study notes for the same code, with no stored outputs. They are **not** needed for grading; `run_all.py` is the reproducible path.

| Folder | Language |
| --- | --- |
| `notebooks/` | Original Korean notes |
| `notebooks_en/` | English translation. Only markdown, comments and printed messages are translated; the code is identical. |

Each notebook reads its data from **its own folder** (e.g. `pd.read_csv("train.csv")`). To run one, copy the data next to it first. The copies are git-ignored:

```bash
python -m pip install jupyter
cp data/* notebooks_en/                 # Windows PowerShell: Copy-Item data\* notebooks_en\
jupyter notebook notebooks_en/problem3_study.ipynb
```

Note: Problem 2 in the notebook takes the same ~14 minutes as the script.

## 7. Directory layout

```
homework_2/
├── README.md               # this file
├── requirements.txt
├── run_all.py              # single entry point (smoke test / per-problem runs)
├── MANIFEST_SHA256.txt
├── code/                   # solution scripts (annotated in Korean), Problem 3 robustness check, report figures
├── data/                   # course-provided inputs
├── submissions/            # final Problem 3 prediction CSVs
├── results/                # final result tables
├── docs/                   # report PDF + LaTeX source
├── notebooks/              # study notebooks (Korean)
└── notebooks_en/           # study notebooks (English translation)
```

## 8. Notes on methodology

- Problems 1 and 2 use a seeded row-level train/validation/test split inside the script (70/15/15). Problem 3 uses the course-provided train/val/test files.
- Hyperparameters (ridge α, RKS D, MLP width, early-stopping epoch) are chosen on the **validation** split only. Test data is used once for final reporting and was not used for tuning.
- `data/` contains the course-provided datasets so that the code runs straight from a clone.
