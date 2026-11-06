"""Homework 2 · 보고서 Figure 1-4를 다시 그리는 코드.
실행: python run_all.py --problem figures   (약 10초; 결과는 outputs/report_figures/)
입력: data/retail.csv, results/hw2_ridge_alpha_sweep.csv, results/hw2_retail_results.csv
  Figure 1: 정규분포 두 개로 'P(Y < 45)'가 평균과 표준편차에 모두 달려 있음을 보이는 설명용 그림(데이터 아님).
  Figure 2: retail.csv 전체 Quantity의 분포(원래 단위, log1p 단위).
  Figure 3: ridge alpha별 검증 log-MSE (problem2_retail.py가 저장한 표).
  Figure 4: 네 모델의 test 지표 (problem2_retail.py가 저장한 표).
"""
import sys  # 명령줄에서 homework_2 폴더 경로를 받는다.
from math import erf  # 표준정규 누적분포 계산에 쓴다.
from pathlib import Path  # 경로 처리.

import numpy as np  # 수치 계산.
import pandas as pd  # CSV 읽기.
import matplotlib  # 그림 저장.
matplotlib.use("Agg")  # 화면 없이 PNG로 저장한다.
import matplotlib.pyplot as plt  # 그래프.
from matplotlib.ticker import FormatStrFormatter  # 축 숫자 형식.

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]  # homework_2 폴더.
OUT = Path(".")  # run_all.py가 outputs/report_figures/를 작업 폴더로 정해 준다.
NAVY, TEAL, ROSE, SLATE, PLUM = "#233650", "#2E8B7A", "#C8726C", "#6F7F95", "#9B7FB5"  # 보고서 색.
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#B8C4D0",
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": "#EEF1F4", "axes.titlecolor": NAVY})


def phi(x, mu, sd):  # 정규분포 밀도.
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


# %% Figure 1: 같은 평균 35분, 표준편차 5분 vs 12분 → 45분 미만 확률이 달라진다.
x = np.linspace(0, 85, 800)
fig, ax = plt.subplots(figsize=(9, 3.6))
for mu, sd, color in [(35, 5, TEAL), (35, 12, ROSE)]:
    y = phi(x, mu, sd)
    p = 0.5 * (1 + erf((45 - mu) / (sd * np.sqrt(2))))  # Φ((45-μ)/σ) = (1 + erf(z/√2)) / 2.
    ax.plot(x, y, color=color, linewidth=2, label=f"Illustrative Normal: mean = {mu}, sd = {sd}  →  P(Y < 45) = {p:.3f}")
    ax.fill_between(x, y, where=x < 45, color=color, alpha=0.18)
ax.axvline(45, color=NAVY, linestyle="--", linewidth=1.3, label="45-minute threshold")
ax.set_xlabel("Trip duration (minutes)")
ax.set_ylabel("Probability density")
ax.set_title("Problem 1 - Probability of a trip lasting less than 45 minutes")
ax.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "P1_gaussian_probability_explained.png", dpi=200)
plt.close(fig)

# %% Figure 2: 실제 retail.csv의 Quantity 분포.
qty = pd.read_csv(ROOT / "data" / "retail.csv", usecols=["Quantity"])["Quantity"].dropna().to_numpy(float)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
ax = axes[0]
bins = np.logspace(0, np.log10(qty.max()) + 0.05, 55)  # 로그 간격 구간.
ax.hist(qty, bins=bins, color=TEAL, alpha=0.8, edgecolor="white", linewidth=0.4)
ax.set_xscale("log")
ax.set_yscale("log")
for q, label in [(np.median(qty), "Median"), (np.quantile(qty, 0.99), "P99")]:
    ax.axvline(q, color=NAVY, linestyle="--", linewidth=1.2)
    ax.text(q * 1.12, ax.get_ylim()[1] * 0.3, f"{label}: {q:g}", rotation=90, color=NAVY, fontsize=8, va="top")
ax.set_xlabel("Original Quantity (log x-axis)")
ax.set_ylabel("Number of transactions (log scale)")
ax.set_title("Concentrated small purchases, long right tail", fontsize=10)
ax = axes[1]
ax.hist(np.log1p(qty), bins=60, color=ROSE, alpha=0.8, edgecolor="white", linewidth=0.4)
ax.set_xlabel("log(1 + Quantity)")
ax.set_ylabel("Transactions")
ax.set_title("log1p compresses extreme magnitudes", fontsize=10)
fig.suptitle(f"Problem 2 - Quantity distribution (all {len(qty):,} rows)", color=NAVY)
fig.tight_layout()
fig.savefig(OUT / "P2_quantity_heavy_tail.png", dpi=200)
plt.close(fig)

# %% Figure 3: ridge alpha 선택 (검증 데이터만 사용).
sweep = pd.read_csv(ROOT / "results" / "hw2_ridge_alpha_sweep.csv")
best = sweep.loc[sweep["validation_log_MSE"].idxmin()]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(sweep["alpha"], sweep["validation_log_MSE"], "o-", color=TEAL, linewidth=2)
ax.set_xscale("log")
ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))  # 오프셋 표기 없이 실제 값으로 표시한다.
ax.set_xlabel("Ridge alpha (log scale)")
ax.set_ylabel("Validation log-MSE")
ax.set_title(f"Ridge penalty chosen on validation: minimum at alpha = {best['alpha']:g}")
spread = sweep["validation_log_MSE"].max() - sweep["validation_log_MSE"].min()
ax.text(0.03, 0.92, f"All five scores lie within {spread:.1e} of each other", transform=ax.transAxes,
        color=SLATE, fontsize=8.5, va="top")
fig.tight_layout()
fig.savefig(OUT / "P2_ridge_alpha_sweep.png", dpi=200)
plt.close(fig)

# %% Figure 4: test 지표 네 모델 비교.
res = pd.read_csv(ROOT / "results" / "hw2_retail_results.csv")
names = {"constant baseline": "Constant baseline", "target encoding + ridge": "Target encoding + ridge",
         "two embeddings + MLP": "Embeddings + MLP", "target encoding + embeddings + MLP": "Hybrid MLP"}
labels = [names.get(m, m) for m in res["model"]]
colors = [SLATE, TEAL, ROSE, PLUM][:len(res)]
ypos = np.arange(len(res))[::-1]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, col, title, fmt in [(axes[0], "log_MSE", "Log-MSE", "{:.3f}"), (axes[1], "MAE", "MAE (Quantity units)", "{:.3f}")]:
    ax.barh(ypos, res[col], color=colors, height=0.6)
    for yy, v in zip(ypos, res[col]):
        ax.text(v * 1.02, yy, fmt.format(v), va="center", fontsize=9, color=NAVY)
    ax.set_xlim(0, res[col].max() * 1.25)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(ypos, labels)
axes[1].set_yticks(ypos, [""] * len(res))
ax = axes[2]
ax.scatter(res["RMSE"], ypos, color=colors, s=70, zorder=3)
for yy, v in zip(ypos, res["RMSE"]):
    ax.text(v + 0.06, yy + 0.12, f"{v:.3f}", fontsize=8.5, color=NAVY)
lo, hi = res["RMSE"].min(), res["RMSE"].max()
ax.set_xlim(lo - 1.0, hi + 1.0)
ax.set_yticks(ypos, [""] * len(res))
ax.set_title("RMSE (Quantity units)")
ax.set_xlabel("Zoomed axis: small differences")
fig.suptitle("Problem 2 - Held-out test metrics", color=NAVY)
fig.tight_layout()
fig.savefig(OUT / "P2_model_metrics.png", dpi=200)
plt.close(fig)
print("Saved P1_gaussian_probability_explained.png, P2_quantity_heavy_tail.png, P2_ridge_alpha_sweep.png, P2_model_metrics.png")
