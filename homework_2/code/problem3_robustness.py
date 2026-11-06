"""Homework 2 · Problem 3 robustness check (supports the discussion in 3(b)-(d)).
실행: python run_all.py --problem robustness  (필요한 파일: train.csv, val.csv; 약 3분)
problem3_rks_mlp.py에서 고른 D=20, width=128이 탐색 범위의 끝이라서 생기는 질문을 확인한다.
  1) D를 20보다 작게/크게 넓히면 어떻게 되는가? 랜덤 특징을 다시 뽑으면 결과가 얼마나 흔들리는가?
  2) MLP width를 128보다 크게 하면 좋아지는가? 초기화 시드를 바꾸면 얼마나 흔들리는가?
  3) 두 모델의 검증 MSE 차이는 검증 데이터 200개의 오차 범위보다 큰가?
  4) 데이터의 노이즈 수준과 RKS의 극한(D→∞, RBF kernel ridge)은 어디인가?
선택 규칙과 제출 예측은 바꾸지 않는다. 이 파일은 확인용 분석만 한다. test.csv는 사용하지 않는다.
"""
import copy  # 검증 성능이 가장 좋은 MLP 가중치를 복사해 둔다.
from pathlib import Path  # 결과 파일 경로를 다룬다.

import numpy as np  # 행렬 계산과 난수.
import pandas as pd  # CSV 읽기와 결과 표.
import matplotlib  # 그림 저장.
matplotlib.use("Agg")  # 화면 없이 PNG로 저장한다.
import matplotlib.pyplot as plt  # 선 그래프.
import torch  # MLP 학습.
from torch import nn  # 신경망 층.
from torch.utils.data import DataLoader, TensorDataset  # 미니배치.
from sklearn.kernel_ridge import KernelRidge  # RKS의 D→∞ 극한인 RBF 커널 ridge.
from sklearn.neighbors import NearestNeighbors  # 가까운 점끼리 y 차이로 노이즈를 추정한다.

OUTPUT_DIR = Path("problem3_outputs")  # problem3_rks_mlp.py와 같은 결과 폴더를 쓴다.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 폴더가 없으면 만든다.
train_df = pd.read_csv("train.csv")  # 학습 600행.
val_df = pd.read_csv("val.csv")  # 검증 200행.
X_train, y_train = train_df[["x1", "x2"]].to_numpy(float), train_df["y"].to_numpy(float)  # 학습 입력/정답.
X_val, y_val = val_df[["x1", "x2"]].to_numpy(float), val_df["y"].to_numpy(float)  # 검증 입력/정답.


def mse(y_true, y_pred):  # 평균제곱오차.
    return float(np.mean((np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()) ** 2))


# %% RKS: problem3_rks_mlp.py의 RandomKitchenSinks와 같은 정의 (w ~ N(0, I/σ²), b ~ U[0, 2π], 절편은 벌점 없음)
def rks_fit_predict(D, X_eval, sigma=1.0, ridge_lambda=0.01, seed=42):  # D개 특징으로 학습하고 X_eval을 예측한다.
    h = D // 2  # sin/cos 쌍의 개수.
    W = np.random.default_rng(seed).normal(0.0, 1.0 / sigma, size=(h, 2)).T  # 랜덤 주파수 (2,h).
    b = np.random.default_rng(seed + 1).uniform(0.0, 2.0 * np.pi, size=h)  # 랜덤 위상.

    def design(X):  # [1, sin, cos, sin, cos, ...] 디자인 행렬.
        angle = X @ W + b
        phi = np.empty((len(X), D))
        phi[:, 0::2], phi[:, 1::2] = np.sin(angle), np.cos(angle)
        return np.column_stack([np.ones(len(X)), phi])

    A = design(X_train)  # 학습 디자인 행렬.
    penalty = np.eye(D + 1)
    penalty[0, 0] = 0.0  # 절편은 벌점에서 뺀다.
    coef = np.linalg.solve(A.T @ A + ridge_lambda * penalty, A.T @ y_train)  # 정규방정식의 해.
    return [design(X) @ coef for X in X_eval]  # 요청한 입력마다 예측을 돌려준다.


# 1) D 범위를 넓히고, seed 42(제출과 같은 특징) + 다른 랜덤 특징 20세트로 반복한다.
SEEDS_RKS = range(100, 120)  # 제출에 쓰지 않은 20개 시드.
rks_rows = []
for D in [8, 12, 16, 18, 20, 24, 30, 40, 60, 80, 160, 320]:
    tr_pred, va_pred = rks_fit_predict(D, [X_train, X_val])  # 제출과 같은 seed 42.
    other = [mse(y_val, rks_fit_predict(D, [X_val], seed=s)[0]) for s in SEEDS_RKS]  # 다른 랜덤 특징들.
    rks_rows.append({"D": D, "train_mse_seed42": mse(y_train, tr_pred), "val_mse_seed42": mse(y_val, va_pred),
                     "val_mse_mean_20draws": np.mean(other), "val_mse_sd_20draws": np.std(other, ddof=1),
                     "val_mse_min_20draws": np.min(other), "val_mse_max_20draws": np.max(other)})
rks_table = pd.DataFrame(rks_rows)
rks_table.to_csv(OUTPUT_DIR / "3e_rks_extended_D.csv", index=False)
print("RKS (sigma=1, lambda=0.01): extended D grid; seed 42 = submitted features, plus 20 other random draws")
print(rks_table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

# 큰 D에서 ridge 벌점을 키우면 과적합이 줄어드는지 확인한다.
lam_rows = []
for D in [20, 80, 320]:
    for lam in [0.01, 0.1, 1.0, 10.0]:
        tr_pred, va_pred = rks_fit_predict(D, [X_train, X_val], ridge_lambda=lam)
        lam_rows.append({"D": D, "lambda": lam, "train_mse": mse(y_train, tr_pred), "val_mse": mse(y_val, va_pred)})
print("\nRKS: D x ridge lambda (seed 42)")
print(pd.DataFrame(lam_rows).to_string(index=False, float_format=lambda v: f"{v:.6f}"))


# %% MLP: problem3_rks_mlp.py의 fit_mlp와 같은 학습 규칙 (Adam lr 0.002, batch 64, 최대 600 epoch, patience 45)
class OneHiddenMLP(nn.Module):  # 2 → width → 1, ReLU.
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_mlp(width, seed):  # 학습 후 (best epoch, train MSE, val MSE, val 예측)을 돌려준다.
    torch.manual_seed(seed)  # 초기화와 미니배치 순서를 시드로 고정한다.
    model = OneHiddenMLP(width)
    xt, yt = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    xv, yv = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()
    best, best_epoch, best_state, wait = float("inf"), 0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, 601):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss_fn(model(bx), by).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xv), yv).item()
        if val < best - 1e-8:  # 검증 MSE가 개선되면 가중치를 저장한다.
            best, best_epoch, best_state, wait = val, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if wait >= 45:  # 45 epoch 연속 개선이 없으면 멈춘다.
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(xv).numpy()
        return best_epoch, mse(y_train, model(xt).numpy()), mse(y_val, val_pred), val_pred


# 2) width 범위를 넓히고, 초기화 시드 5개(42 = 제출과 같은 시드)로 반복한다.
mlp_rows, mlp128_val_pred = [], None
for width in [8, 16, 32, 64, 128, 256, 512, 1024]:
    for seed in [42, 0, 1, 2, 3]:
        epoch, tr_mse, va_mse, va_pred = fit_mlp(width, seed)
        mlp_rows.append({"hidden_width": width, "seed": seed, "best_epoch": epoch, "train_mse": tr_mse, "val_mse": va_mse})
        if width == 128 and seed == 42:
            mlp128_val_pred = va_pred  # 제출 설정의 검증 예측을 아래 비교에 쓴다.
mlp_runs = pd.DataFrame(mlp_rows)
mlp_runs.to_csv(OUTPUT_DIR / "3e_mlp_extended_width_runs.csv", index=False)
mlp_table = mlp_runs.groupby("hidden_width").agg(
    val_mse_seed42=("val_mse", "first"), val_mse_mean_5seeds=("val_mse", "mean"),
    val_mse_sd_5seeds=("val_mse", "std"), val_mse_min_5seeds=("val_mse", "min"), val_mse_max_5seeds=("val_mse", "max"),
    train_mse_mean_5seeds=("train_mse", "mean")).reset_index()
mlp_table.to_csv(OUTPUT_DIR / "3e_mlp_extended_width.csv", index=False)
print("\nMLP: extended width grid; seed 42 = submitted initialization, plus seeds 0-3")
print(mlp_table.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

# %% 3), 4) 노이즈 수준, D→∞ 극한, 두 모델 차이의 표준오차
_, nn_index = NearestNeighbors(n_neighbors=2).fit(X_train).kneighbors(X_train)  # 각 점의 가장 가까운 이웃.
noise_upper = float(np.mean((y_train - y_train[nn_index[:, 1]]) ** 2) / 2)  # 이웃끼리 f가 조금 달라서 위쪽으로 치우친 추정.
best_kr = None
for lengthscale in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:  # RBF 폭 (최적값이 범위 안쪽에 오도록 넓게).
    for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:  # 커널 ridge 벌점.
        kr = KernelRidge(kernel="rbf", gamma=1.0 / (2 * lengthscale ** 2), alpha=alpha).fit(X_train, y_train - y_train.mean())
        val = mse(y_val, kr.predict(X_val) + y_train.mean())
        if best_kr is None or val < best_kr[0]:
            best_kr = (val, lengthscale, alpha, mse(y_train, kr.predict(X_train) + y_train.mean()))
rks20_val_pred = rks_fit_predict(20, [X_val])[0]  # 제출 RKS(D=20, seed 42)의 검증 예측.
diff = (y_val - mlp128_val_pred) ** 2 - (y_val - rks20_val_pred) ** 2  # 같은 검증 점에서 오차 차이(paired).
se_val = np.std((y_val - rks20_val_pred) ** 2, ddof=1) / np.sqrt(len(y_val))  # 검증 MSE 하나의 표준오차.
print("\nNoise level and limits")
print(f"variance of y (train): {y_train.var():.4f}")
print(f"nearest-neighbour noise estimate (upper bound): {noise_upper:.4f}")
print(f"RBF kernel ridge (D -> infinity limit of RKS), best on validation: val MSE {best_kr[0]:.6f}, "
      f"train MSE {best_kr[3]:.6f} (lengthscale {best_kr[1]}, alpha {best_kr[2]})")
print(f"standard error of a 200-point validation MSE: {se_val:.4f}")
print(f"MLP(128, retrained with the submitted settings) minus RKS(D=20) validation MSE: {diff.mean():.6f} +- {np.std(diff, ddof=1) / np.sqrt(len(diff)):.6f} (paired SE)")

# %% 그림: 시드 42(제출 설정) 선, 다른 시드들의 최소~최대 범위(음영)와 평균. 작은 D의 큰 오차 때문에 y축은 로그.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
panels = [
    (axes[0], rks_table, "D", "20draws", "20 other random-feature draws", "D (number of random features, log scale)",
     "RKS (σ = 1, λ = 0.01): extended D grid", 20),
    (axes[1], mlp_table, "hidden_width", "5seeds", "5 initialization seeds (incl. 42)", "hidden width (log scale)",
     "One-hidden-layer MLP: extended width grid", 128),
]
for ax, table, x_name, tag, band_label, x_label, title, chosen in panels:
    ax.fill_between(table[x_name], table[f"val_mse_min_{tag}"], table[f"val_mse_max_{tag}"], alpha=0.2,
                    color="tab:orange", label=f"min–max over {band_label}")
    ax.plot(table[x_name], table[f"val_mse_mean_{tag}"], "s--", color="tab:orange", label="mean over those runs")
    ax.plot(table[x_name], table["val_mse_seed42"], "o-", color="tab:blue", label="seed 42 (as submitted)")
    ax.axhline(best_kr[0], color="gray", linestyle=":", label=f"RBF kernel ridge, D → ∞ ({best_kr[0]:.4f})")
    ax.axvline(chosen, color="tab:blue", alpha=0.3, linewidth=6)  # 제출에서 고른 값.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 2.0)
    ax.set_xlabel(x_label)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(True, which="both", alpha=0.25)
axes[0].set_ylabel("validation MSE (log scale)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "3e_robustness.png", dpi=180)
plt.close(fig)
print("\nSaved 3e_rks_extended_D.csv, 3e_mlp_extended_width.csv, 3e_mlp_extended_width_runs.csv, 3e_robustness.png")
