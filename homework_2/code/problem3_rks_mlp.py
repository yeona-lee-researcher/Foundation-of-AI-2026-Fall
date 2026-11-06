# -*- coding: utf-8 -*-
"""Homework 2 · Problem 3(a)–(d): RKS + Ridge vs one-hidden-layer MLP.

실행 방법 (파일이 같은 폴더에 있을 때): python problem3_rks_mlp.py  (homework_2 폴더에서는 python run_all.py --problem 3)
필요한 파일: train.csv, val.csv, test.csv
의존 패키지: numpy, pandas, matplotlib, torch
모든 실행문 옆에는 '이 줄이 무엇을 하는가'에 대한 한국어 설명을 적었다.
중요: test.csv에는 정답 y가 없기 때문에 test MSE를 계산하지 않는다.
"""

# %% 0. 라이브러리: 먼저 필요한 모든 이름을 정의한다.
import copy  # 학습 중 가장 좋은 신경망 가중치를 안전하게 복사한다.
from pathlib import Path  # CSV 및 결과 파일의 경로를 OS에 독립적으로 다룬다.
import numpy as np  # 배열, 행렬 곱, 난수, 선형방정식 해법을 제공한다.
import pandas as pd  # CSV 파일을 읽고 실험 결과를 표로 저장한다.
import matplotlib  # 그림을 파일로 저장하는 Matplotlib의 기본 패키지다.
matplotlib.use("Agg")  # 화면이 없는 환경에서도 PNG 이미지를 저장하도록 설정한다.
import matplotlib.pyplot as plt  # 산점도, 손실 곡선, 등고면을 그린다.
import torch  # 신경망을 계산하고 자동미분하는 PyTorch다.
from torch import nn  # nn.Module, nn.Linear, nn.ReLU, nn.MSELoss 등을 사용한다.
from torch.utils.data import DataLoader, TensorDataset  # 훈련 데이터를 미니배치로 공급한다.

# %% 1. 실험의 고정 설정: 이 변수를 바꾸면 전체 실험에 영향을 준다.
SEED = 42  # 같은 프로그램을 다시 실행할 때 재현성을 높이기 위한 숫자다.
np.random.seed(SEED)  # NumPy의 전역 난수 생성 상태를 고정한다.
torch.manual_seed(SEED)  # PyTorch의 가중치 초기화 등 난수 생성 상태를 고정한다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA GPU가 있으면 사용하고, 없으면 CPU를 사용한다.
DATA_DIR = Path(".")  # 현재 작업 폴더에서 train.csv, val.csv, test.csv를 찾는다.
OUTPUT_DIR = Path("problem3_outputs")  # 그림과 실험표를 저장할 폴더 이름이다.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 결과 폴더가 없으면 생성하고, 이미 있으면 넘어간다.
RKS_D_VALUES = [20, 40, 80, 160, 320]  # RKS의 총 sin/cos 특성 개수 D=2h 후보를 정한다.
RKS_SIGMA = 1.0  # 랜덤 주파수의 표준편차를 1/sigma로 지정하는 스케일이다.
RKS_LAMBDA = 0.01  # RKS의 선형회귀 계수에 추가할 Ridge L2 벌점 강도다.
MLP_WIDTH_VALUES = [16, 32, 64, 128]  # MLP의 은닉층 노드 수 후보를 정한다.
MLP_LR = 0.002  # Adam이 가중치를 업데이트할 때 사용하는 학습률이다.
MLP_BATCH_SIZE = 64  # 한 번의 가중치 업데이트에 사용할 데이터 개수다.
MLP_MAX_EPOCHS = 600  # MLP가 전체 학습 데이터를 최대 몇 번 볼지 정한다.
MLP_PATIENCE = 45  # 검증 MSE 개선이 없는 Epoch가 이만큼 쌓이면 멈춘다.
print(f"계산 장치: {DEVICE}")  # 실제 선택된 CPU/GPU를 터미널에서 확인한다.

# %% 2. 주어진 CSV 세 개를 읽고, 모양과 내용에 문제가 없는지 확인한다.
train_df = pd.read_csv(DATA_DIR / "train.csv")  # 정답 y가 있는 훈련 데이터 600행을 읽는다.
val_df = pd.read_csv(DATA_DIR / "val.csv")  # 정답 y가 있는 검증 데이터 200행을 읽는다.
test_df = pd.read_csv(DATA_DIR / "test.csv")  # 정답 y가 없는 테스트 데이터 200행을 읽는다.
assert list(train_df.columns) == ["x1", "x2", "y"], "train.csv 열은 x1,x2,y여야 해요"  # 훈련 열 이름과 순서를 검증한다.
assert list(val_df.columns) == ["x1", "x2", "y"], "val.csv 열은 x1,x2,y여야 해요"  # 검증 열 이름과 순서를 검증한다.
assert list(test_df.columns) == ["x1", "x2"], "test.csv 열은 x1,x2여야 해요"  # 테스트에는 y열이 없어야 한다.
X_train = train_df[["x1", "x2"]].to_numpy(dtype=np.float64)  # 훈련의 두 입력 열을 (n_train,2) 배열로 만든다.
y_train = train_df["y"].to_numpy(dtype=np.float64)  # 훈련의 실제 답을 길이 n_train 벡터로 만든다.
X_val = val_df[["x1", "x2"]].to_numpy(dtype=np.float64)  # 검증의 두 입력 열을 (n_val,2) 배열로 만든다.
y_val = val_df["y"].to_numpy(dtype=np.float64)  # 검증의 실제 답을 길이 n_val 벡터로 만든다.
X_test = test_df[["x1", "x2"]].to_numpy(dtype=np.float64)  # 테스트에서는 입력 x1,x2만 사용한다.
for array in (X_train, y_train, X_val, y_val, X_test):  # 모든 데이터 배열을 차례로 점검한다.
    assert np.isfinite(array).all(), "데이터에 NaN/무한대가 있어요"  # 계산 불가능한 숫자가 있는지 검사한다.
assert X_train.ndim == 2 and X_train.shape[1] == 2  # 훈련 데이터의 입력 차원이 정확히 2인지 확인한다.
assert X_val.ndim == 2 and X_val.shape[1] == 2  # 검증 데이터의 입력 차원이 정확히 2인지 확인한다.
assert X_test.ndim == 2 and X_test.shape[1] == 2  # 테스트 데이터의 입력 차원이 정확히 2인지 확인한다.
print(f"데이터: train={len(X_train)} / val={len(X_val)} / test={len(X_test)}")  # 각 분할의 관측치 개수를 보여 준다.

# %% 3. 수학의 MSE를 코드로 직접 정의한다.
def mse(y_true, y_pred):  # 실제값 배열과 예측값 배열을 받아 MSE 하나를 계산한다.
    y_true = np.asarray(y_true).reshape(-1)  # 실제값을 모양 (n,)의 1차원 배열로 바꾼다.
    y_pred = np.asarray(y_pred).reshape(-1)  # 예측값도 (n,) 모양으로 통일한다.
    assert y_true.shape == y_pred.shape  # 같은 행끼리 비교할 수 있도록 길이를 확인한다.
    return float(np.mean((y_true - y_pred) ** 2))  # (실제-예측)^2을 평균 낸 값을 반환한다.

# %% 4. [3(a)] 훈련 데이터를 평면에 점으로 그리고 색으로 정답 y를 표현한다.
fig, ax = plt.subplots(figsize=(7, 5.8))  # PNG로 저장할 그래프와 좌표축을 만든다.
scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="viridis", s=28, alpha=0.85)  # 위치=(x1,x2), 점 색깔=y다.
ax.set_xlabel("x1")  # 가로축에 첫 번째 입력의 이름을 붙인다.
ax.set_ylabel("x2")  # 세로축에 두 번째 입력의 이름을 붙인다.
ax.set_title("3(a) Training points colored by y")  # 과제 소문항과 그래프의 의미를 제목에 적는다.
ax.set_xlim(-3, 3)  # 문제에 주어진 x1 범위를 사용한다.
ax.set_ylim(-3, 3)  # 문제에 주어진 x2 범위를 사용한다.
ax.set_aspect("equal", adjustable="box")  # 두 방향의 거리 척도를 같게 만들어 공간 패턴을 왜곡하지 않는다.
fig.colorbar(scatter, ax=ax, label="observed y")  # 색깔이 어떤 y값을 나타내는지 범례를 붙인다.
fig.tight_layout()  # 글씨와 그래프가 잘리지 않도록 여백을 맞춘다.
fig.savefig(OUTPUT_DIR / "3a_training_scatter.png", dpi=180)  # 3(a)의 제출용 그림 파일을 저장한다.
plt.close(fig)  # 불필요한 그림 메모리를 해제한다.

# %% 5. [3(b)] RKS 모델: random sin/cos 특성 + Ridge Regression을 직접 구현한다.
class RandomKitchenSinks:  # RKS 모델의 생성, 변환, 학습, 예측을 하나의 클래스로 묶는다.
    def __init__(self, D=80, sigma=1.0, ridge_lambda=0.01, seed=42):  # 학습 전에 정할 하이퍼파라미터를 받는다.
        if D <= 0 or D % 2 != 0:  # sin/cos는 쌍으로 만들므로 D는 양의 짝수여야 한다.
            raise ValueError("D must be a positive even integer")  # 잘못된 D가 입력되면 즉시 알려 준다.
        if sigma <= 0 or ridge_lambda <= 0:  # σ와 λ를 모두 양수로 제한한다.
            raise ValueError("sigma and ridge_lambda must be positive")  # 불가능한 설정을 걸러 낸다.
        self.D = int(D)  # 전체 변환 특성의 개수 D를 저장한다.
        self.h = self.D // 2  # sin/cos 쌍의 개수 h는 D/2이다.
        self.sigma = float(sigma)  # 랜덤 주파수의 스케일을 저장한다.
        self.ridge_lambda = float(ridge_lambda)  # Ridge 규제 강도를 저장한다.
        rng_w = np.random.default_rng(seed)  # 랜덤 주파수 w를 뽑는 독립적인 난수 생성기다.
        rng_b = np.random.default_rng(seed + 1)  # 랜덤 위상 b를 뽑는 별도의 난수 생성기다.
        self.W = rng_w.normal(0.0, 1.0 / self.sigma, size=(self.h, 2)).T  # w_j 두 좌표를 N(0,1/sigma^2)에서 뽑아 (2,h)로 만든다.
        self.b = rng_b.uniform(0.0, 2.0 * np.pi, size=self.h)  # 각 사인파의 시작 위치 b_j를 [0,2π]에서 뽑는다.
        self.coef_ = None  # 아직 학습하지 않았으므로 최종 선형회귀 계수가 없다.

    def _features(self, X):  # 입력 x1,x2를 여러 개의 sin/cos 숫자로 바꾸는 함수다.
        angle = X @ self.W + self.b  # 모든 데이터에 대해 w_j^T x_i + b_j를 계산한다; 결과는 (n,h)다.
        phi = np.empty((len(X), self.D), dtype=np.float64)  # 출력 sin/cos 특징을 담을 (n,D) 빈 행렬을 만든다.
        phi[:, 0::2] = np.sin(angle)  # 짝수 열(0,2,4,...)마다 사인값을 채워 넣는다.
        phi[:, 1::2] = np.cos(angle)  # 홀수 열(1,3,5,...)마다 코사인값을 채워 넣는다.
        return phi  # 한 데이터마다 D개 특징을 가진 행렬을 반환한다.

    def _design(self, X):  # 절편 b0를 포함하는 선형회귀 디자인 행렬을 만든다.
        ones = np.ones((len(X), 1), dtype=np.float64)  # 절편 계수를 곱할 '1' 열을 만든다.
        return np.column_stack((ones, self._features(X)))  # 첫 열 1 + D개 특징을 이어 (n,D+1)을 반환한다.

    def fit(self, X, y):  # 주어진 입력과 정답으로 Ridge의 계수를 학습한다.
        A = self._design(X)  # 모든 데이터에 대한 실제 디자인 행렬 A를 만든다.
        penalty = np.eye(self.D + 1, dtype=np.float64)  # L2 벌점을 주기 위한 (D+1)×(D+1) 단위행렬을 만든다.
        penalty[0, 0] = 0.0  # 첫 계수는 절편이므로 Ridge 벌점을 받지 않게 만든다.
        left = A.T @ A + self.ridge_lambda * penalty  # 정규방정식 왼쪽 A^T A + λP를 계산한다.
        right = A.T @ y  # 정규방정식 오른쪽 A^T y를 계산한다.
        self.coef_ = np.linalg.solve(left, right)  # 역행렬을 직접 계산하지 않고 연립방정식을 풀어 계수를 찾는다.
        return self  # 학습된 객체 자체를 반환해 .fit(...).predict(...)처럼 쓸 수 있게 한다.

    def predict(self, X):  # 학습된 계수로 새 데이터에 대한 y값을 예측한다.
        if self.coef_ is None:  # 아직 fit을 한 적 없는 모델인지 확인한다.
            raise RuntimeError("Call fit() before predict()")  # 학습 전 예측 시 명확한 오류를 낸다.
        return self._design(X) @ self.coef_  # 새로운 디자인 행렬과 계수 벡터를 곱해 예측값을 반환한다.

# %% 6. [3(b)] D를 바꿔 보고, Train MSE와 Validation MSE를 모두 기록한다.
rks_rows = []  # 여러 RKS 후보의 평가 기록을 담을 빈 리스트다.
for D in RKS_D_VALUES:  # 미리 정한 D=20,40,80,160,320을 순서대로 시도한다.
    candidate = RandomKitchenSinks(D=D, sigma=RKS_SIGMA, ridge_lambda=RKS_LAMBDA, seed=SEED)  # D 외의 조건을 고정한 후보 모델이다.
    candidate.fit(X_train, y_train)  # 오직 훈련 데이터로 각 후보의 Ridge 계수를 학습한다.
    train_mse = mse(y_train, candidate.predict(X_train))  # 학습 데이터에서의 예측 오차를 기록한다.
    val_mse = mse(y_val, candidate.predict(X_val))  # 보지 않은 검증 데이터의 예측 오차를 기록한다.
    rks_rows.append({"D": D, "train_mse": train_mse, "val_mse": val_mse})  # D와 두 손실을 하나의 기록으로 저장한다.
    print(f"RKS D={D:3d} | train MSE={train_mse:.6f} | val MSE={val_mse:.6f}")  # 각 후보 성능을 즉시 출력한다.
rks_table = pd.DataFrame(rks_rows)  # 위 기록을 CSV로 저장할 수 있는 표로 바꾼다.
rks_table.to_csv(OUTPUT_DIR / "3b_rks_sweep.csv", index=False)  # 각 D의 손실을 실험 데이터 파일로 저장한다.
best_rks_idx = int(rks_table["val_mse"].idxmin())  # 검증 MSE가 최소인 행 번호를 찾는다.
best_D = int(rks_table.loc[best_rks_idx, "D"])  # 그 행에 기록된 최적 D를 선택한다.
print(f"선택한 최적 RKS D: {best_D}")  # 검증 기준으로 선택한 설정을 보여 준다.

# %% 7. Hyperparameter sweep 그래프는 train과 validation을 구분해 두 개로 그린다.
def plot_sweep(table, x_name, y_name, title, filename):  # 표의 한 손실 열을 선 그래프로 저장하는 재사용 함수다.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))  # 그림과 좌표축을 생성한다.
    ax.plot(table[x_name], table[y_name], marker="o", linewidth=2)  # 가로축=하이퍼파라미터, 세로축=MSE를 그린다.
    ax.set_xlabel(x_name)  # 가로축에 D 또는 hidden_width를 적는다.
    ax.set_ylabel("Mean squared error (MSE)")  # 세로축에 평가 지표를 적는다.
    ax.set_title(title)  # 무엇을 비교한 그래프인지 제목을 붙인다.
    ax.grid(True, alpha=0.3)  # 값을 눈으로 비교하기 쉽도록 연한 격자를 그린다.
    fig.tight_layout()  # 글씨가 잘리지 않도록 여백을 조정한다.
    fig.savefig(OUTPUT_DIR / filename, dpi=180)  # PNG 이미지로 저장한다.
    plt.close(fig)  # 메모리를 해제한다.

plot_sweep(rks_table, "D", "train_mse", "3(b) RKS training MSE vs D", "3b_rks_training_mse.png")  # 요구된 RKS 학습 손실 그래프다.
plot_sweep(rks_table, "D", "val_mse", "3(b) RKS validation MSE vs D", "3b_rks_validation_mse.png")  # 요구된 RKS 검증 손실 그래프다.

# %% 8. [3(c)] 정확히 은닉층이 하나인 MLP 신경망 클래스를 작성한다.
class OneHiddenMLP(nn.Module):  # PyTorch의 모든 신경망이 상속하는 nn.Module을 부모 클래스로 사용한다.
    def __init__(self, width):  # width는 중간의 은닉 노드를 몇 개 둘지 정하는 숫자다.
        super().__init__()  # PyTorch가 내부 가중치를 정상적으로 추적하도록 부모 생성자를 실행한다.
        self.net = nn.Sequential(  # 층을 적은 순서대로 연결해 하나의 신경망으로 구성한다.
            nn.Linear(2, width),  # (x1,x2)의 두 숫자를 width개의 가중합으로 바꾼다.
            nn.ReLU(),  # 음수는 0, 양수는 그대로 통과시켜 비선형성을 추가한다.
            nn.Linear(width, 1),  # 은닉 표현 width개를 최종 예측값 하나로 바꾼다.
        )  # Sequential 정의를 마친다.

    def forward(self, x):  # 모델에 입력 x가 들어왔을 때 계산할 순서를 정한다.
        return self.net(x).squeeze(1)  # (batch,1) 형태의 출력을 (batch,)로 바꿔 정답 모양과 맞춘다.

# %% 9. MLP를 직접 훈련하고 검증 MSE가 가장 좋을 때의 가중치를 보존한다.
def fit_mlp(width):  # 은닉층 너비 하나를 받아 훈련된 모델과 평가기록을 반환한다.
    torch.manual_seed(SEED)  # 후보마다 비교 가능한 초기화를 얻도록 같은 난수 시드를 사용한다.
    model = OneHiddenMLP(width).to(DEVICE)  # 모델을 만들고 CPU 또는 GPU로 이동시킨다.
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # 훈련 입력을 PyTorch 실수 Tensor로 변환한다.
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # 훈련 정답을 PyTorch 실수 Tensor로 변환한다.
    dataset = TensorDataset(x_train_tensor, y_train_tensor)  # 같은 행의 입력과 정답을 하나의 샘플로 묶는다.
    loader = DataLoader(dataset, batch_size=MLP_BATCH_SIZE, shuffle=True)  # 미니배치를 만들고 Epoch마다 순서를 섞는다.
    full_train_x = x_train_tensor.to(DEVICE)  # Epoch 마지막에 전체 train MSE를 정확히 계산할 입력을 준비한다.
    full_train_y = y_train_tensor.to(DEVICE)  # Epoch 마지막에 전체 train MSE를 계산할 정답을 준비한다.
    val_x_tensor = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)  # 검증 입력을 같은 계산 장치에 만든다.
    val_y_tensor = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)  # 검증 정답을 같은 계산 장치에 만든다.
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR)  # 모든 학습 가능한 가중치의 업데이트를 Adam이 담당한다.
    loss_fn = nn.MSELoss()  # 훈련에서 (예측-정답)^2 평균을 최소화한다.
    best_val_mse = float("inf")  # 아직 검증 성능이 없으므로 최고 점수를 무한대로 시작한다.
    best_epoch = 0  # 검증 성능이 가장 좋았던 Epoch 번호를 기록한다.
    best_state = copy.deepcopy(model.state_dict())  # 좋은 결과가 나왔을 때 되돌릴 가중치 복사본이다.
    epochs_without_improve = 0  # 검증 손실이 개선되지 않은 연속 Epoch 수다.
    history = []  # 매 Epoch train/val MSE를 담아 학습 과정을 시각화할 기록이다.
    for epoch in range(1, MLP_MAX_EPOCHS + 1):  # 1부터 최대 Epoch 수까지 학습을 반복한다.
        model.train()  # 신경망을 훈련 모드로 전환한다.
        for batch_x, batch_y in loader:  # 훈련 데이터 중 한 미니배치를 가져온다.
            batch_x = batch_x.to(DEVICE)  # 입력 배치를 모델과 같은 장치로 옮긴다.
            batch_y = batch_y.to(DEVICE)  # 정답 배치도 같은 장치로 옮긴다.
            optimizer.zero_grad()  # 지난 배치에서 남은 기울기를 지운다.
            batch_pred = model(batch_x)  # 신경망으로 이번 배치의 y를 예측한다.
            batch_loss = loss_fn(batch_pred, batch_y)  # 이번 배치의 제곱오차 평균을 계산한다.
            batch_loss.backward()  # 연쇄법칙/역전파로 모든 가중치의 기울기를 계산한다.
            optimizer.step()  # 계산된 기울기를 사용해 Adam이 가중치를 갱신한다.
        model.eval()  # 검증 계산을 위해 평가 모드로 전환한다.
        with torch.no_grad():  # 여기서는 평가만 하므로 불필요한 미분 그래프를 만들지 않는다.
            epoch_train = loss_fn(model(full_train_x), full_train_y).item()  # 현 Epoch의 전체 훈련 MSE를 숫자로 얻는다.
            epoch_val = loss_fn(model(val_x_tensor), val_y_tensor).item()  # 현 Epoch의 전체 검증 MSE를 숫자로 얻는다.
        history.append({"epoch": epoch, "train_mse": epoch_train, "val_mse": epoch_val})  # 학습 과정의 한 행을 기록한다.
        if epoch_val < best_val_mse - 1e-8:  # 이전 최고 검증 MSE보다 충분히 개선되었는지 검사한다.
            best_val_mse = epoch_val  # 이번 Epoch을 최고 점수로 등록한다.
            best_epoch = epoch  # 최고 점수가 나온 Epoch 번호를 저장한다.
            best_state = copy.deepcopy(model.state_dict())  # 모델의 최고 가중치 전체를 복사해 보관한다.
            epochs_without_improve = 0  # 성능이 개선됐으므로 대기 카운터를 초기화한다.
        else:  # 검증 MSE가 개선되지 않았다면 실행되는 부분이다.
            epochs_without_improve += 1  # 더 좋아지지 않은 Epoch 수를 하나 늘린다.
        if epochs_without_improve >= MLP_PATIENCE:  # 연속으로 너무 오래 좋아지지 않았는지 확인한다.
            break  # 과적합과 불필요한 계산을 피하기 위해 훈련 반복을 종료한다.
    model.load_state_dict(best_state)  # 마지막 Epoch이 아닌 '검증 MSE가 가장 작았던' 가중치로 복원한다.
    model.eval()  # 최종 예측 시에는 평가 모드로 둔다.
    history_table = pd.DataFrame(history)  # Epoch별 기록을 데이터프레임으로 변환한다.
    final_train_mse = mse(y_train, predict_mlp(model, X_train))  # 복원한 최적 가중치의 train MSE를 계산한다.
    final_val_mse = mse(y_val, predict_mlp(model, X_val))  # 복원한 최적 가중치의 val MSE를 계산한다.
    return model, best_epoch, final_train_mse, final_val_mse, history_table  # 학습된 모델과 평가 결과를 모두 돌려준다.

# %% 10. PyTorch 모델로 예측하고 NumPy 배열로 되돌리는 함수를 만든다.
def predict_mlp(model, X, chunk_size=4096):  # 데이터가 아주 많을 때도 나눠 예측할 수 있는 함수다.
    model.eval()  # Dropout 등 학습/평가 동작이 다를 수 있는 층을 평가 모드로 둔다.
    pieces = []  # 여러 예측 배치의 결과를 쌓을 빈 리스트다.
    with torch.no_grad():  # 예측 시에는 역전파가 필요 없으므로 미분 기록을 끈다.
        for start in range(0, len(X), chunk_size):  # chunk_size만큼 입력 행을 나눠 처리한다.
            batch = X[start:start + chunk_size]  # 입력 배열에서 현재 예측 배치만 꺼낸다.
            tensor = torch.tensor(batch, dtype=torch.float32, device=DEVICE)  # 배치를 신경망과 같은 장치의 Tensor로 만든다.
            predicted = model(tensor).detach().cpu().numpy()  # 예측 Tensor를 CPU NumPy 배열로 바꾼다.
            pieces.append(predicted)  # 이번 배치 결과를 순서대로 보관한다.
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)  # 모든 예측을 원래 행 순서대로 합친다.

# %% 11. [3(c)] 은닉층 width를 바꾸고 학습/검증 MSE를 표로 기록한다.
mlp_rows = []  # width별 평가 결과를 저장할 리스트를 만든다.
for width in MLP_WIDTH_VALUES:  # 16,32,64,128개 노드인 모델을 차례대로 비교한다.
    candidate, epoch, train_mse, val_mse, history_table = fit_mlp(width)  # 하나의 모델을 학습하고 조기종료 결과를 받는다.
    mlp_rows.append({"hidden_width": width, "best_epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})  # 선택 기준을 기록한다.
    history_table.to_csv(OUTPUT_DIR / f"3c_mlp_history_width_{width}.csv", index=False)  # 해당 width의 Epoch별 학습곡선을 보존한다.
    print(f"MLP width={width:3d} | best epoch={epoch:3d} | train MSE={train_mse:.6f} | val MSE={val_mse:.6f}")  # 후보 성능을 출력한다.
mlp_table = pd.DataFrame(mlp_rows)  # 결과 리스트를 표로 만든다.
mlp_table.to_csv(OUTPUT_DIR / "3c_mlp_sweep.csv", index=False)  # 전체 width의 결과를 CSV에 저장한다.
best_mlp_idx = int(mlp_table["val_mse"].idxmin())  # 검증 손실이 가장 낮은 행을 찾는다.
best_width = int(mlp_table.loc[best_mlp_idx, "hidden_width"])  # 해당 행의 최적 width를 선택한다.
print(f"선택한 최적 MLP hidden width: {best_width}")  # MLP의 최적 설정을 출력한다.
plot_sweep(mlp_table, "hidden_width", "train_mse", "3(c) MLP training MSE vs width", "3c_mlp_training_mse.png")  # width별 Train MSE 그래프를 저장한다.
plot_sweep(mlp_table, "hidden_width", "val_mse", "3(c) MLP validation MSE vs width", "3c_mlp_validation_mse.png")  # width별 Validation MSE 그래프를 저장한다.

# %% 12. [3(d)] 선택한 설정으로 두 모델을 다시 학습하고 검증 손실을 비교한다.
final_rks = RandomKitchenSinks(D=best_D, sigma=RKS_SIGMA, ridge_lambda=RKS_LAMBDA, seed=SEED)  # 검증으로 선택한 D를 사용해 RKS를 새로 만든다.
final_rks.fit(X_train, y_train)  # 오직 훈련 데이터만으로 RKS의 최종 계수를 다시 계산한다.
final_mlp, final_epoch, mlp_train_mse, mlp_val_mse, final_history = fit_mlp(best_width)  # 선택한 width의 MLP를 초기화부터 다시 학습한다.
rks_train_mse = mse(y_train, final_rks.predict(X_train))  # 최종 RKS의 훈련 MSE를 계산한다.
rks_val_mse = mse(y_val, final_rks.predict(X_val))  # 최종 RKS의 검증 MSE를 계산한다.
comparison = pd.DataFrame([  # 두 모델의 성능을 하나의 표로 만든다.
    {"model": "RKS", "selected_hyperparameter": f"D={best_D}", "train_mse": rks_train_mse, "val_mse": rks_val_mse},  # RKS 결과 한 행이다.
    {"model": "MLP", "selected_hyperparameter": f"hidden_width={best_width}", "train_mse": mlp_train_mse, "val_mse": mlp_val_mse},  # MLP 결과 한 행이다.
])  # DataFrame 생성이 끝난다.
comparison.to_csv(OUTPUT_DIR / "3d_model_comparison.csv", index=False)  # 제출 보고서의 수치표를 CSV로 저장한다.
final_history.to_csv(OUTPUT_DIR / "3d_final_mlp_learning_curve.csv", index=False)  # 최종 MLP 학습곡선도 저장한다.
print("\n[3(d)] 최종 검증 성능\n" + comparison.to_string(index=False))  # 비교 표를 터미널에 출력한다.

# %% 13. Test는 정답이 없으므로 예측값만 만들고 행 순서 그대로 CSV로 내보낸다.
rks_test_pred = final_rks.predict(X_test)  # 최종 RKS로 test.csv의 각 행을 예측한다.
mlp_test_pred = predict_mlp(final_mlp, X_test)  # 최종 MLP로 같은 test.csv 행들을 예측한다.
assert len(rks_test_pred) == len(X_test)  # RKS 예측 행 수가 테스트 데이터 행 수와 같은지 검사한다.
assert len(mlp_test_pred) == len(X_test)  # MLP 예측 행 수도 원본과 같은지 검사한다.
assert np.isfinite(rks_test_pred).all() and np.isfinite(mlp_test_pred).all()  # NaN 또는 무한대 예측값이 있는지 검사한다.
np.savetxt("rks_test_predictions.csv", rks_test_pred, fmt="%.10f", delimiter=",")  # 헤더 없이 행마다 예측 숫자 하나인 RKS 제출 CSV를 저장한다.
np.savetxt("mlp_test_predictions.csv", mlp_test_pred, fmt="%.10f", delimiter=",")  # 헤더 없이 행마다 예측 숫자 하나인 MLP 제출 CSV를 저장한다.
print("테스트 예측 저장 완료: rks_test_predictions.csv / mlp_test_predictions.csv")  # 실제 파일명을 알려 준다.

# %% 14. [3(d)] [-3,3]×[-3,3] 전체를 촘촘한 격자로 만들어 두 모델의 곡면을 그린다.
axis_values = np.linspace(-3.0, 3.0, 160)  # -3부터 3까지 같은 간격으로 160개의 좌표를 만든다.
mesh_x1, mesh_x2 = np.meshgrid(axis_values, axis_values)  # 모든 (x1,x2) 격자 조합을 만든다.
grid_inputs = np.column_stack((mesh_x1.ravel(), mesh_x2.ravel()))  # (160^2,2) 모양의 예측 입력 행렬로 바꾼다.
rks_grid_pred = final_rks.predict(grid_inputs).reshape(mesh_x1.shape)  # RKS 예측을 160×160 격자 모양으로 재배열한다.
mlp_grid_pred = predict_mlp(final_mlp, grid_inputs).reshape(mesh_x1.shape)  # MLP 예측을 동일한 격자로 재배열한다.
color_low = float(min(y_train.min(), rks_grid_pred.min(), mlp_grid_pred.min()))  # 세 그림에서 색깔 기준의 최솟값을 통일한다.
color_high = float(max(y_train.max(), rks_grid_pred.max(), mlp_grid_pred.max()))  # 세 그림에서 색깔 기준의 최댓값을 통일한다.

# %% 15. 공통 색 범위를 사용해 각 예측 곡면의 독립된 그림을 저장한다.
def save_surface(Z, title, filename):  # 예측 격자 Z와 파일명을 받아 2D 등고 색지도를 저장한다.
    fig, ax = plt.subplots(figsize=(7, 5.8))  # 그림과 좌표축을 만든다.
    contour = ax.contourf(mesh_x1, mesh_x2, Z, levels=35, cmap="viridis", vmin=color_low, vmax=color_high)  # 격자에서 예측값의 공간 분포를 색깔로 표현한다.
    ax.set_xlabel("x1")  # x1 좌표축 이름을 표시한다.
    ax.set_ylabel("x2")  # x2 좌표축 이름을 표시한다.
    ax.set_title(title)  # RKS 또는 MLP 표면인지 제목을 표시한다.
    ax.set_xlim(-3, 3)  # 문제의 입력 범위와 동일하게 가로축 범위를 정한다.
    ax.set_ylim(-3, 3)  # 문제의 입력 범위와 동일하게 세로축 범위를 정한다.
    ax.set_aspect("equal", adjustable="box")  # 가로/세로 축의 척도를 같게 한다.
    fig.colorbar(contour, ax=ax, label="predicted y")  # 색상에 대응하는 예측 y 크기를 표시한다.
    fig.tight_layout()  # 라벨 잘림을 방지한다.
    fig.savefig(OUTPUT_DIR / filename, dpi=180)  # 해당 예측 표면을 PNG로 저장한다.
    plt.close(fig)  # 그림 객체를 닫는다.

save_surface(rks_grid_pred, f"3(d) RKS fitted surface, D={best_D}", "3d_rks_surface.png")  # RKS 최종 예측 공간 그림을 저장한다.
save_surface(mlp_grid_pred, f"3(d) MLP fitted surface, width={best_width}", "3d_mlp_surface.png")  # MLP 최종 예측 공간 그림을 저장한다.

# %% 16. 훈련 자료의 패턴과 RKS/MLP의 예측을 같은 눈금으로 나란히 비교한다.
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)  # 왼쪽부터 실제 데이터, RKS, MLP 세 패널을 만든다.
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="viridis", vmin=color_low, vmax=color_high, s=12)  # 훈련 데이터의 실제 색을 표시한다.
axes[0].set_title("Observed training y")  # 첫 패널의 의미를 적는다.
axes[1].contourf(mesh_x1, mesh_x2, rks_grid_pred, levels=35, cmap="viridis", vmin=color_low, vmax=color_high)  # RKS의 예상 공간 패턴을 표시한다.
axes[1].set_title(f"RKS, D={best_D}")  # 두 번째 패널의 모델 및 설정을 적는다.
axes[2].contourf(mesh_x1, mesh_x2, mlp_grid_pred, levels=35, cmap="viridis", vmin=color_low, vmax=color_high)  # MLP의 예상 공간 패턴을 표시한다.
axes[2].set_title(f"MLP, width={best_width}")  # 세 번째 패널의 모델 및 설정을 적는다.
for ax in axes:  # 세 좌표축에 동일한 표시 범위를 적용한다.
    ax.set_xlabel("x1")  # 가로축 이름을 통일한다.
    ax.set_ylabel("x2")  # 세로축 이름을 통일한다.
    ax.set_xlim(-3, 3)  # 같은 x1 범위를 사용한다.
    ax.set_ylim(-3, 3)  # 같은 x2 범위를 사용한다.
    ax.set_aspect("equal", adjustable="box")  # 세 패널의 공간 크기 비율을 같게 한다.
fig.tight_layout()  # 세 개의 그림 간격을 알맞게 배치한다.
fig.savefig(OUTPUT_DIR / "3d_three_panel_comparison.png", dpi=180)  # (d)에서 가장 핵심인 비교 그림을 저장한다.
plt.close(fig)  # 불필요한 그래프 메모리를 정리한다.

# %% 17. 최종 모델 MLP가 어떻게 학습되었는지 검증/훈련 손실을 같은 그림에 기록한다.
fig, ax = plt.subplots(figsize=(7, 4.5))  # Early Stopping 학습곡선 그래프를 만든다.
ax.plot(final_history["epoch"], final_history["train_mse"], label="Train MSE")  # Epoch별 훈련 오차를 그린다.
ax.plot(final_history["epoch"], final_history["val_mse"], label="Validation MSE")  # Epoch별 검증 오차를 그린다.
ax.axvline(final_epoch, linestyle="--", label=f"Best epoch={final_epoch}")  # 최적 검증 오차를 얻은 Epoch을 표시한다.
ax.set_xlabel("Epoch")  # 가로축의 단위를 명시한다.
ax.set_ylabel("MSE")  # 세로축의 손실 지표를 명시한다.
ax.set_title("3(d) Final MLP learning curves")  # 그래프의 제목을 붙인다.
ax.legend()  # 선들의 이름을 표시한다.
ax.grid(True, alpha=0.3)  # 약한 격자를 넣어 숫자 추세를 보기 쉽게 한다.
fig.tight_layout()  # 그림이 잘리지 않도록 정리한다.
fig.savefig(OUTPUT_DIR / "3d_mlp_learning_curve.png", dpi=180)  # 학습과 검증이 달라지는 과정을 PNG로 저장한다.
plt.close(fig)  # 그림 객체를 닫는다.

# %% 18. 영문 제출 초안을 실제 실행 수치로 자동 채운다 (해석은 그래프를 보고 검토할 것).
winner = "RKS" if rks_val_mse < mlp_val_mse else "MLP"  # 더 작은 검증 MSE를 낸 모델명을 정한다.
english_summary = f"""Problem 3: Random Kitchen Sinks versus One-Hidden-Layer MLP

(a) Data visualization
We plotted the 600 labeled training examples as a 2D scatter plot, with x1 and x2 on the axes and observed y represented by color. See 3a_training_scatter.png.

(b) Random Kitchen Sinks with ridge regression
We transformed each two-dimensional input into D sinusoidal features: sin(w_j^T x+b_j) and cos(w_j^T x+b_j), where w_j ~ N(0,I_2/sigma^2), b_j ~ Uniform(0,2pi), h=D/2, sigma={RKS_SIGMA}, and the random seed was fixed at {SEED}. We fit an L2-regularized least-squares model with lambda={RKS_LAMBDA}, excluding the intercept from the penalty. We swept D over {RKS_D_VALUES} and chose D={best_D} by validation MSE. The refitted model achieved train MSE={rks_train_mse:.6f} and validation MSE={rks_val_mse:.6f}. See 3b_rks_training_mse.png and 3b_rks_validation_mse.png.

(c) One-hidden-layer MLP
We trained a neural network with two input units, one fully connected ReLU hidden layer, and one linear output. We swept hidden layer width over {MLP_WIDTH_VALUES}, fixing Adam learning rate at {MLP_LR}, batch size at {MLP_BATCH_SIZE}, and early-stopping patience at {MLP_PATIENCE}. Using the validation MSE criterion, we selected width={best_width}. See 3c_mlp_training_mse.png and 3c_mlp_validation_mse.png.

(d) Comparison and visualization
We refitted each selected model on the training split and evaluated both against the labeled validation split. RKS validation MSE={rks_val_mse:.6f}; MLP validation MSE={mlp_val_mse:.6f}. The lower validation MSE was achieved by {winner}. Both predictions on the unlabeled 200-row test split were exported, preserving the original row order, to rks_test_predictions.csv and mlp_test_predictions.csv (headerless, one scalar per row). See 3d_three_panel_comparison.png, 3d_rks_surface.png, and 3d_mlp_surface.png.

Interpretation: RKS uses a fixed, randomly constructed smooth sinusoidal basis and fits only the linear output weights with ridge regression; the MLP learns both its hidden features and its output weights through backpropagation. In the three-panel figure both surfaces recover the trough near the left-center and the higher values toward the top and bottom; the RKS surface is smoother, while the ReLU MLP surface is slightly more piecewise-linear. No test MSE is reported because test.csv has no ground-truth y.
"""  # test에는 정답 y가 없으므로 위 문자열에는 검증 결과만 숫자로 적는다.
(OUTPUT_DIR / "3_submittable_english_draft.txt").write_text(english_summary, encoding="utf-8")  # 수치가 채워진 영문 초안을 텍스트 파일로 저장한다.
print("\n3번 전체 완료: problem3_outputs 폴더, 두 test 예측 CSV, 영문 제출 초안을 확인하세요.")  # 모든 작업이 완료되었음을 알려 준다.
