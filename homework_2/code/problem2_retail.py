# %% [markdown]
# Homework 2, Problem 2(b)~(e) | 한 줄씩 배우는 실행용 Python 코드
# 실행: python problem2_retail.py retail.csv  (homework_2 폴더에서는 python run_all.py --problem 2)
# 필요한 패키지: python -m pip install numpy pandas scikit-learn torch
# 데이터 파일에는 StockCode, CustomerID, Quantity 열이 있어야 합니다.
# ※ 각 코드 바로 위의 한국어 주석은 그 줄의 목적과 데이터의 변화를 설명합니다.

# %% 0. 라이브러리: 데이터 표, 행렬, 신경망, 전처리 도구 가져오기
import sys  # 명령줄에 쓴 'retail.csv' 파일 경로를 읽을 때 사용한다.
import copy  # 가장 성능 좋은 신경망의 가중치를 '복사'해서 저장한다.
import numpy as np  # 행렬 A, 벡터 beta, 평균, 제곱 등 수치계산을 수행한다.
import pandas as pd  # CSV를 표(DataFrame)로 읽고 제품/고객별 평균을 계산한다.
import torch  # PyTorch Tensor, CPU/GPU 계산과 자동미분에 사용한다.
from torch import nn  # nn.Embedding, nn.Linear, nn.ReLU 등 신경망 부품을 가져온다.
from torch.utils.data import TensorDataset, DataLoader  # 데이터를 작은 묶음(batch)으로 공급한다.
from sklearn.model_selection import KFold  # Target Encoding의 자기 정답 누수를 줄이는 5-fold 분할.
from sklearn.preprocessing import StandardScaler  # 평균을 빼고 표준편차로 나누어 입력을 표준화한다.
from sklearn.linear_model import Ridge  # 우리가 손으로 만든 Ridge와 정답이 같은지 확인한다.

# %% 1. 실험 환경과 재현 가능한 난수 고정
SEED = 42  # '무작위'를 완전히 같게 재현하기 위한 출발 번호다.
np.random.seed(SEED)  # NumPy의 행 섞기 결과를 실행할 때마다 같게 만든다.
torch.manual_seed(SEED)  # PyTorch의 초기 Embedding/가중치 난수를 고정한다.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA GPU가 있으면 GPU, 없으면 CPU.
COLUMNS = ["StockCode", "CustomerID"]  # Quantity를 예측하기 위해 사용할 두 범주형 입력 변수.
print("계산 장치:", DEVICE)  # 'cuda' 또는 'cpu'를 화면에 표시해 사용 장치를 확인한다.

# %% 2. CSV 읽기 → 깨진 행 검사 → 모든 모델에 공통으로 사용할 데이터 분할
file_path = sys.argv[1] if len(sys.argv) > 1 else "retail.csv"  # 실행 인자가 없으면 현재 폴더의 retail.csv.
df = pd.read_csv(file_path, dtype={c: "string" for c in COLUMNS})  # 제품/고객 ID의 앞자리 0을 보존하려고 문자열로 읽는다.
df = df.dropna(subset=COLUMNS + ["Quantity"]).reset_index(drop=True)  # 입력이나 정답이 없는 행을 빼고 행 번호를 다시 매긴다.
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="raise")  # 수량을 숫자로 바꾼다; 'abc'가 있으면 조용히 넘어가지 않고 오류를 낸다.

# 조건 검사는 실패한 데이터를 몰래 학습하지 않도록, 문제를 바로 알리기 위함이다.
if not np.isfinite(df["Quantity"].to_numpy(dtype=float)).all():  # 수량에 무한대가 섞여 있는가?
    raise ValueError("Quantity는 유한한 숫자여야 합니다.")  # 그렇다면 학습을 진행하지 않고 원인을 알려준다.
if (df["Quantity"] <= 0).any():  # 과제에는 Quantity가 '양의 정수'라고 주어졌다. 0 이하가 있는가?
    raise ValueError("이 과제에서 Quantity는 양수여야 합니다.")  # 전제와 다른 데이터는 우선 확인한다.
if len(df) < 10:  # 5-fold를 만들 수 없을 만큼 파일이 너무 작은지 확인한다.
    raise ValueError("데이터가 너무 적습니다. 적어도 10행 이상을 사용해 주세요.")

# (a) 분할 전에 전체 데이터의 범주 수와 Quantity 꼬리를 확인한다. 난수를 쓰지 않으므로 분할 결과는 그대로다.
n_cat = {c: df[c].nunique() for c in COLUMNS}  # 열마다 서로 다른 ID의 개수.
print("(a) 전체 행 수:", len(df), "/ 고유 StockCode:", n_cat["StockCode"], "/ 고유 CustomerID:", n_cat["CustomerID"])
print("(a) dense one-hot float32 크기(GB):", round(len(df) * sum(n_cat.values()) * 4 / 1e9, 2))  # 행 수 × 범주 수 × 4바이트.
print("(a) Quantity 분위수 (50/90/99/99.9%, 최대):", df["Quantity"].quantile([.5, .9, .99, .999]).tolist(), df["Quantity"].max())

idx = np.random.permutation(len(df))  # 행 번호 0~N-1을 무작위로 섞는다; 행 자체의 순서를 바꾸는 건 아직 아니다.
cut_train = int(0.70 * len(df))  # 전체의 70% 지점: 학습용 데이터의 끝 위치.
cut_valid = int(0.85 * len(df))  # 전체의 85% 지점: 검증용 데이터의 끝 위치.
tr_idx, va_idx, te_idx = np.split(idx, [cut_train, cut_valid])  # 70%, 15%, 15%의 행 번호 묶음으로 나눈다.
train = df.iloc[tr_idx].reset_index(drop=True)  # 행 번호를 사용해 학습 데이터만 꺼낸다.
valid = df.iloc[va_idx].reset_index(drop=True)  # 검증 데이터: alpha나 Early Stopping을 고른다.
test = df.iloc[te_idx].reset_index(drop=True)  # 테스트 데이터: 모든 모델 선택이 끝나야 평가한다.
print("분할된 행 수 (train/valid/test):", len(train), len(valid), len(test))  # 데이터 분할이 의도한 크기인지 확인한다.

# %% 3. (b) Target Encoding 함수: 제품/고객 코드 → '해당 범주의 평균 Quantity'
def target_encode(reference, query):  # reference는 평균을 계산할 표; query는 숫자로 바꿀 표다.
    fallback = reference["Quantity"].mean()  # 학습에 없던 새 고객은 전체 학습 평균으로 대체한다.
    features = np.empty((len(query), len(COLUMNS)), dtype=np.float64)  # query 행 수 × 2개 입력 열의 빈 실수 행렬.
    for j, col in enumerate(COLUMNS):  # j=0일 땐 StockCode, j=1일 땐 CustomerID를 하나씩 처리한다.
        means = reference.groupby(col)["Quantity"].mean()  # 예: 제품 A의 수량이 2,4,6이면 A→4.
        mapped = query[col].map(means)  # query의 범주 이름을 위에서 계산한 평균값으로 교체한다.
        mapped = mapped.fillna(fallback)  # reference에 없던 범주/NA 인코딩값을 전체 평균으로 채운다.
        features[:, j] = mapped.to_numpy(dtype=float)  # [:, j]는 '모든 행의 j번째 열'에 저장하라는 뜻.
    return features  # 모양 (query의 행 수, 2)의 인코딩된 행렬을 반환한다.

# %% 4. 5-Fold Out-of-Fold Target Encoding: 자기 자신의 정답으로 인코딩하지 않기
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)  # train을 무작위 5조각으로 나눈다.
te_train = np.empty((len(train), 2), dtype=float)  # 각 학습 행의 인코딩값을 채울 빈 2열 행렬.
for fitting, heldout in kf.split(train):  # 매 반복에서 4조각은 평균 계산, 1조각은 인코딩 대상.
    ref_rows = train.iloc[fitting]  # 정답 평균을 계산하는 4조각(현재 heldout 제외).
    query_rows = train.iloc[heldout]  # 자기 정답을 평균 계산에 포함하면 안 되는 1조각.
    te_train[heldout] = target_encode(ref_rows, query_rows)  # 원래 행 위치에 인코딩 결과를 정확히 저장한다.
te_valid = target_encode(train, valid)  # 검증 데이터 정답은 평균 계산에 일절 사용하지 않는다.
te_test = target_encode(train, test)  # 테스트 데이터 정답 역시 평균 계산에 일절 사용하지 않는다.
print("Target Encoding 행렬 크기:", te_train.shape, te_valid.shape, te_test.shape)

# %% 5. Heavy-tail 수량과 입력 스케일 처리
# 예: Quantity=1 → log(2)≈0.693, 1000 → log(1001)≈6.909.
# log1p(x)는 log(1+x)이다. 로그 변환 뒤 MSE를 적용하면 큰 수량 몇 개의 지배를 완화한다.
y_train = np.log1p(train["Quantity"].to_numpy(dtype=float))  # 학습 정답: log(1+Quantity).
y_valid = np.log1p(valid["Quantity"].to_numpy(dtype=float))  # 검증 정답: 같은 로그 스케일.
y_test = np.log1p(test["Quantity"].to_numpy(dtype=float))  # 테스트 정답: 마지막 평가에만 사용.
true_qty = test["Quantity"].to_numpy(dtype=float)  # MAE/RMSE 계산을 위해 원래 수량 단위도 보존한다.

scaler = StandardScaler()  # 각 입력 열을 (값-학습 평균)/학습 표준편차로 만들 도구.
X_train = scaler.fit_transform(np.log1p(te_train))  # OOF 인코딩 → log1p → 학습 평균·표준편차 계산 후 표준화.
X_valid = scaler.transform(np.log1p(te_valid))  # 이미 계산된 학습 스케일로만 검증 입력을 변환한다.
X_test = scaler.transform(np.log1p(te_test))  # 테스트 분포의 평균·표준편차로 다시 학습하면 안 된다.
print("표준화한 두 입력의 train 평균:", X_train.mean(axis=0))  # train 각 열의 평균은 0에 가깝다.
print("표준화한 두 입력의 train 표준편차:", X_train.std(axis=0))  # train 각 열의 표준편차는 대략 1이다.

# %% 6. Ridge 수식 그대로 구현: (A^T A + alpha D) beta = A^T t
# 모델: 예측 log수량 = b + w1*(제품 인코딩 값) + w2*(고객 인코딩 값).
# 목적함수: sum_i (t_i - 예측_i)^2 + alpha*(w1^2 + w2^2). 절편 b에는 벌점이 없다.
def manual_ridge(X, y, alpha):  # X는 (N,2), y는 (N,), alpha는 Ridge 벌점 강도다.
    ones = np.ones(len(X))  # 절편 b에 곱해줄 1을 N개 만든다.
    A = np.column_stack([ones, X])  # X 앞에 1열을 붙여 A=[1,x1,x2]인 (N,3) 행렬을 만든다.
    D = np.diag([0.0] + [1.0] * X.shape[1])  # D=diag(0,1,1): 절편 b에는 벌점을 주지 않는다.
    left = A.T @ A + alpha * D  # 정규방정식의 왼쪽 3×3 행렬: A^T A + alpha D.
    right = A.T @ y  # 정규방정식의 오른쪽 3개 숫자: A^T y.
    beta = np.linalg.solve(left, right)  # left @ beta = right를 풀어서 [b,w1,w2]를 구한다.
    return beta  # 찾은 세 계수를 반환한다.

# %% 7. Validation으로 alpha 고르기: Test를 '선택'에 사용하면 안 됨
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]  # 약한 벌점부터 강한 벌점까지 비교한다.
best_alpha = None  # 아직 최적 alpha를 고르지 않았다.
best_val_mse = float("inf")  # 초기 최고 점수는 무한대; 첫 후보가 반드시 더 좋다.
alpha_rows = []  # 그림(보고서 Figure 3)에 쓸 alpha별 검증 점수 기록.
for alpha in alphas:  # 후보 alpha마다 반복한다.
    beta_candidate = manual_ridge(X_train, y_train, alpha)  # 오직 train으로 계수 학습.
    A_valid = np.column_stack([np.ones(len(X_valid)), X_valid])  # 검증 입력에도 절편용 1열을 붙인다.
    pred_valid = A_valid @ beta_candidate  # (N_valid,3)×(3,)→(N_valid,)의 로그 예측 벡터.
    val_mse = np.mean((y_valid - pred_valid) ** 2)  # validation에서의 평균 로그 제곱오차.
    print(f"alpha={alpha:g} / validation log-MSE={val_mse:.6f}")  # 각 후보의 성능을 출력한다.
    alpha_rows.append({"alpha": alpha, "validation_log_MSE": val_mse})  # 소수점을 자르지 않고 저장한다.
    if val_mse < best_val_mse:  # 이번 후보의 검증 오차가 지금까지보다 작으면,
        best_alpha = alpha  # 현재 alpha를 새로운 최적 후보로 기억한다.
        best_val_mse = val_mse  # 현재 오차도 함께 기억해 다음 후보와 비교한다.
pd.DataFrame(alpha_rows).to_csv("hw2_ridge_alpha_sweep.csv", index=False, float_format="%.10f")  # Figure 3의 입력 파일.

beta_ridge = manual_ridge(X_train, y_train, best_alpha)  # 선택된 alpha로 train 데이터에서 다시 계수를 구한다.
print("선택된 alpha:", best_alpha)  # Test를 아직 안 본 상태에서 확정한 최적 벌점을 표시한다.
print("직접 구현한 [b, w1, w2]:", beta_ridge)  # 정규방정식을 풀어 얻은 계수를 출력한다.

# %% 8. sklearn Ridge로 구현 정확성 검증하기
sk_model = Ridge(alpha=best_alpha)  # scikit-learn에 내장된 Ridge 모델을 만든다.
sk_model.fit(X_train, y_train)  # 같은 train 데이터와 같은 alpha로 학습한다.
sk_beta = np.r_[sk_model.intercept_, sk_model.coef_]  # 절편과 2개 계수를 [b,w1,w2]로 이어 붙인다.
np.testing.assert_allclose(beta_ridge, sk_beta, rtol=1e-5, atol=1e-6)  # 작은 부동소수점 오차 안에서 같은지 확인한다.
print("NumPy 수작업 Ridge == sklearn Ridge: PASS")  # 검사에서 예외가 나지 않았으면 성공이다.

# %% 9. PyTorch로도 같은 선형시스템을 풀어 CPU/GPU Tensor 검증하기
A_np = np.column_stack([np.ones(len(X_train)), X_train])  # NumPy의 (N,3) 학습 입력 A를 만든다.
A_torch = torch.tensor(A_np, dtype=torch.float64, device=DEVICE)  # 같은 A를 double Tensor로 변환해 장치로 이동.
y_torch = torch.tensor(y_train, dtype=torch.float64, device=DEVICE)  # 로그 정답 y도 같은 장치에 둔다.
d_values = torch.tensor([0., 1., 1.], dtype=torch.float64, device=DEVICE)  # 벌점 대각원소: [0,1,1].
D_torch = torch.diag(d_values)  # 3×3 대각행렬 D를 구성한다.
left_torch = A_torch.T @ A_torch + best_alpha * D_torch  # PyTorch 수식도 A^T A + alpha D.
right_torch = A_torch.T @ y_torch  # 오른쪽 A^T y 벡터를 계산한다.
beta_torch = torch.linalg.solve(left_torch, right_torch)  # CPU 또는 GPU에서 같은 연립방정식을 푼다.
np.testing.assert_allclose(beta_torch.cpu().numpy(), beta_ridge, rtol=1e-5, atol=1e-6)  # Tensor 값을 CPU NumPy로 옮겨 비교.
print("NumPy Ridge == PyTorch Ridge: PASS")  # 같은 정규방정식이 같은 계수를 만드는지 확인한다.

# %% 10. (c) 고유한 범주 이름을 1,2,3,...의 정수 ID로 바꾸기
# Embedding은 문자('A123') 대신 정수 ID(1,2,3)를 입력받는다.
# 0번은 train에서 보지 못한 '새로운 제품/고객'을 위해 예약한다.
def make_ids(col):  # col은 'StockCode' 또는 'CustomerID' 문자열이다.
    vocabulary = {item: i + 1 for i, item in enumerate(train[col].unique())}  # train에 있는 이름으로만 ID 사전 작성.
    def encode(series):  # 이 내부 함수는 주어진 열의 범주 이름을 ID로 바꾼다.
        mapped = series.map(vocabulary)  # 사전에 있는 이름이면 1,2,3,...로 변환한다.
        filled = mapped.fillna(0)  # 사전에 없는 새로운 이름은 0으로 표시한다.
        return filled.to_numpy(dtype=np.int64)  # nn.Embedding 인덱스에 맞는 64-bit 정수 배열로 바꾼다.
    return (  # train/valid/test 각각의 ID 배열과 Embedding 테이블 크기를 함께 전달한다.
        encode(train[col]),  # 학습 범주의 정수 ID 배열.
        encode(valid[col]),  # 검증 범주의 정수 ID 배열.
        encode(test[col]),  # 테스트 범주의 정수 ID 배열.
        len(vocabulary) + 1,  # 0번(unknown)을 포함한 총 ID 개수.
    )

stock_tr, stock_va, stock_te, n_stock = make_ids("StockCode")  # 제품 코드를 정수 ID로 바꾼다.
cust_tr, cust_va, cust_te, n_cust = make_ids("CustomerID")  # 고객 코드를 정수 ID로 바꾼다.
cat_train = np.column_stack([stock_tr, cust_tr])  # 각 학습 행이 [제품ID, 고객ID] 형태가 된다.
cat_valid = np.column_stack([stock_va, cust_va])  # 같은 두 열로 검증 입력을 만든다.
cat_test = np.column_stack([stock_te, cust_te])  # 같은 두 열로 테스트 입력을 만든다.
print("train에 있는 고유 제품 수(+unknown):", n_stock, "/ 고유 고객 수(+unknown):", n_cust)  # 임베딩 표 크기; 전체 범주 수는 위 (a) 출력.

# %% 11. Embedding 모델 및 Hybrid 모델에 공통으로 쓰는 신경망 설계
class RetailNet(nn.Module):  # PyTorch 신경망의 기본 클래스 nn.Module을 상속받는다.
    def __init__(self, n_stock, n_cust, hybrid=False, emb_dim=8):  # 생성자: 테이블 크기, 결합 여부, Embedding 차원.
        super().__init__()  # 부모 nn.Module이 필요한 내부 초기화를 수행하게 한다.
        self.hybrid = hybrid  # True이면 (d), False이면 (c)로 같은 설계를 재사용한다.
        self.stock_emb = nn.Embedding(n_stock, emb_dim, padding_idx=0)  # 제품마다 학습할 8개 수; unknown 0은 영벡터.
        self.cust_emb = nn.Embedding(n_cust, emb_dim, padding_idx=0)  # 고객마다 학습할 8개 수; unknown 0은 영벡터.
        n_in = 2 * emb_dim + (2 if hybrid else 0)  # (c)는 8+8=16개; (d)는 8+8+2=18개 입력.
        self.net = nn.Sequential(  # 여러 신경망 연산을 순서대로 묶는다.
            nn.Linear(n_in, 32),  # 입력 16/18개를 학습 가능한 32개 은닉값으로 변환한다.
            nn.ReLU(),  # 음수는 0, 양수는 그대로 둬 비선형 관계를 학습하게 한다.
            nn.Linear(32, 1),  # 32개 은닉값을 하나의 '예측 로그 수량'으로 바꾼다.
        )

    def forward(self, cat, numeric):  # 모델을 호출할 때 실제 데이터가 통과하는 계산 경로.
        stock_ids = cat[:, 0]  # 모든 행의 첫 번째 열: 제품 정수 ID.
        customer_ids = cat[:, 1]  # 모든 행의 두 번째 열: 고객 정수 ID.
        stock_vector = self.stock_emb(stock_ids)  # 제품 ID마다 8개로 된 학습 가능 벡터를 조회한다.
        customer_vector = self.cust_emb(customer_ids)  # 고객 ID마다 8개로 된 벡터를 조회한다.
        parts = [stock_vector, customer_vector]  # 두 Embedding을 연결할 목록을 만든다.
        if self.hybrid:  # (d)의 Hybrid 모델인지 검사한다.
            parts.append(numeric)  # Hybrid면 표준화된 Target Encoding 2개도 목록에 추가한다.
        x = torch.cat(parts, dim=1)  # 같은 행의 입력을 가로로 연결하여 16개 또는 18개 특성을 만든다.
        output = self.net(x)  # Linear→ReLU→Linear로 로그 수량을 예측한다.
        return output.squeeze(1)  # (batch,1)에서 불필요한 열 차원을 제거해 (batch,)로 만든다.

# %% 12. TensorDataset과 DataLoader: 모델에 1,024행씩 공급하기
def to_loader(cat, numeric, y, shuffle):  # 범주형 ID, 숫자 입력, 정답을 동일한 행 순서로 묶는다.
    cat_tensor = torch.tensor(cat, dtype=torch.long)  # Embedding이 받아야 할 정수 ID Tensor.
    numeric_tensor = torch.tensor(numeric, dtype=torch.float32)  # Target Encoding 특성은 소수 실수 Tensor.
    y_tensor = torch.tensor(y, dtype=torch.float32)  # 학습 목표인 로그 수량도 실수 Tensor.
    dataset = TensorDataset(cat_tensor, numeric_tensor, y_tensor)  # 동일 행의 세 정보를 하나의 사례로 묶는다.
    loader = DataLoader(dataset, batch_size=1024, shuffle=shuffle)  # 매번 최대 1024개 사례를 꺼내 준다.
    return loader  # 학습 또는 평가 반복문에서 사용할 묶음 공급기 반환.

train_loader = to_loader(cat_train, X_train, y_train, shuffle=True)  # 학습은 매 epoch 행을 무작위로 섞는다.
valid_loader = to_loader(cat_valid, X_valid, y_valid, shuffle=False)  # 검증은 섞을 이유가 없다.

# %% 13. 학습 함수: 예측→MSE→역전파→Adam 업데이트→검증→Early Stopping
def fit_net(hybrid, max_epochs=300, patience=10):  # hybrid 선택, 최대 300 epoch, 개선 없는 10 epoch에서 중단.
    torch.manual_seed(SEED)  # (c)와 (d)를 비교할 때 초기화 난수를 일정하게 한다.
    net = RetailNet(n_stock, n_cust, hybrid=hybrid).to(DEVICE)  # 신경망을 만들고 GPU/CPU로 보낸다.
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)  # Adam, 학습률 0.001, 작은 L2 패널티.
    best_loss = float("inf")  # 가장 작은 검증 log-MSE를 저장할 변수.
    best_epoch = 0  # 가장 좋았던 epoch 번호(1부터 셈).
    best_state = None  # 최고 성능 시점의 모델 파라미터 사본.
    stale = 0  # 검증 성능이 개선되지 않은 연속 epoch 수.

    for epoch in range(max_epochs):  # 데이터 전체를 다시 한 번 보는 과정을 최대 300번 반복한다(보통 그 전에 조기 종료).
        net.train()  # 모델을 학습 모드로 설정한다.
        for cat, numeric, target in train_loader:  # 한 번에 최대 1024행으로 구성된 배치를 가져온다.
            cat = cat.to(DEVICE)  # 제품/고객 ID Tensor를 모델과 동일한 장치로 이동한다.
            numeric = numeric.to(DEVICE)  # 두 개의 숫자 입력도 같은 장치로 이동한다.
            target = target.to(DEVICE)  # 로그 수량 정답도 같은 장치로 이동한다.
            optimizer.zero_grad()  # 직전 배치에서 쌓인 기울기를 초기화한다.
            prediction = net(cat, numeric)  # 현재 Embedding과 가중치로 로그 수량을 예측한다.
            loss = nn.functional.mse_loss(prediction, target)  # 배치에서 (예측-정답)^2의 평균을 계산한다.
            loss.backward()  # 연쇄법칙으로 모든 학습 가능한 파라미터에 대해 기울기를 구한다.
            optimizer.step()  # Adam이 그 기울기를 이용해 신경망 파라미터를 수정한다.

        net.eval()  # 지금부터는 학습 없이 검증 성능만 측정할 모드로 바꾼다.
        total_squared_error = 0.0  # 검증 데이터 전체의 제곱오차 총합을 담을 변수.
        count = 0  # 전체 검증 사례 수를 세어 정확한 평균을 구한다.
        with torch.no_grad():  # 검증에서는 역전파가 불필요하므로 기울기 계산을 끈다.
            for cat, numeric, target in valid_loader:  # 검증 데이터를 배치 단위로 순회한다.
                cat = cat.to(DEVICE)  # 검증 제품/고객 ID를 장치로 이동한다.
                numeric = numeric.to(DEVICE)  # 검증 Target Encoding 입력을 장치로 이동한다.
                target = target.to(DEVICE)  # 검증 로그 정답도 장치로 이동한다.
                pred = net(cat, numeric)  # 학습에서 보지 않은 검증 데이터에 대해 예측한다.
                batch_sum = nn.functional.mse_loss(pred, target, reduction="sum")  # 평균이 아닌 '배치의 합'을 계산한다.
                total_squared_error += batch_sum.item()  # 배치별 오차 합을 누적해 큰 배치에 알맞은 비중을 준다.
                count += len(target)  # 이번 배치의 데이터 수를 누적한다.
        val_loss = total_squared_error / count  # 전체 합을 전체 개수로 나눈 정확한 validation MSE.
        if val_loss < best_loss - 1e-6:  # 검증 MSE가 이전 최저보다 1e-6 이상 개선됐는가?
            best_loss = val_loss  # 새로 기록한 최저 MSE를 저장한다.
            best_epoch = epoch + 1  # 그때의 epoch 번호도 기록한다.
            best_state = copy.deepcopy(net.state_dict())  # 현재 모든 가중치의 '독립적인 사본'을 보관한다.
            stale = 0  # 개선됐으므로 연속 미개선 횟수를 0으로 돌린다.
        else:  # 개선이 없었다면,
            stale += 1  # 연속 미개선 횟수를 하나 늘린다.
        if stale >= patience:  # 연속 미개선 횟수가 10에 도달했는가?
            break  # 더 학습하지 않고 종료한다: Early Stopping.

    net.load_state_dict(best_state)  # 마지막 가중치가 아니라 '검증 성능이 최고였던 가중치'를 복원한다.
    print(f"hybrid={hybrid}, epochs={epoch+1}, best epoch={best_epoch}, best validation log-MSE={best_loss:.6f}")  # 학습 요약 출력.
    return net  # 학습이 끝난 최적 모델을 반환한다.

# %% 14. 학습된 신경망으로 Test 또는 다른 데이터 예측하기
@torch.no_grad()  # 아래 함수 전체에서 기울기를 계산하지 않는다. 학습이 아니라 예측 전용이다.
def predict_net(net, cat, numeric):  # 학습된 신경망, 정수ID, 숫자 입력을 받아 로그 수량 예측을 만든다.
    net.eval()  # 평가 모드로 바꾼다.
    predictions = []  # 배치별 예측 결과를 차례로 담을 빈 목록.
    dummy_y = np.zeros(len(cat), dtype=np.float32)  # DataLoader 형식을 맞추기 위한 가짜 정답; 예측에는 사용 안 함.
    loader = to_loader(cat, numeric, dummy_y, shuffle=False)  # 예측할 데이터를 1024행씩 나눠 공급한다.
    for cat_batch, num_batch, _ in loader:  # 각 배치의 ID/숫자 입력만 사용하고 가짜 정답은 버린다.
        cat_batch = cat_batch.to(DEVICE)  # ID를 모델 장치로 이동한다.
        num_batch = num_batch.to(DEVICE)  # 숫자 입력도 모델 장치로 이동한다.
        batch_prediction = net(cat_batch, num_batch)  # 한 배치의 로그 수량을 예측한다.
        predictions.append(batch_prediction.cpu().numpy())  # 결과를 CPU의 NumPy 배열로 저장한다.
    return np.concatenate(predictions)  # 배치별 배열을 원래 행 순서의 긴 1차원 배열로 합친다.

# %% 15. (c)와 (d) 모델을 같은 조건으로 학습하기
embedding_model = fit_net(hybrid=False)  # (c) 8D 제품+8D 고객 Embedding만 사용한 MLP를 학습한다.
hybrid_model = fit_net(hybrid=True)  # (d) 위 Embedding에 Target Encoding 2개도 더한 MLP를 학습한다.

# %% 16. (e) 테스트 성능: 로그 단위와 원래 Quantity 단위를 둘 다 보고하기
# 주의: 이 함수를 호출할 때까지 Test를 이용하여 alpha, epoch, 네트워크를 고르지 않았다.
def evaluate(name, pred_log):  # name은 모델 이름, pred_log는 log(1+수량) 예측이다.
    pred_log = np.maximum(pred_log, 0.0)  # 수량이 음수가 되지 않도록 로그 예측도 0 이상으로 자른다.
    pred_qty = np.expm1(pred_log)  # log(1+y)의 역함수 exp(t)-1로 원래 수량 단위로 복원한다.
    log_mse = np.mean((y_test - pred_log) ** 2)  # 로그 스케일 평균제곱오차: 학습 기준과 연결된다.
    rmsle = np.sqrt(log_mse)  # 로그 MSE의 제곱근: 작은 값이 좋다.
    mae = np.mean(np.abs(true_qty - pred_qty))  # 원래 Quantity 단위에서 평균 몇 개를 틀리는가?
    rmse = np.sqrt(np.mean((true_qty - pred_qty) ** 2))  # 큰 수량 오차에 더 큰 벌점을 주는 RMSE.
    return {"model": name, "log_MSE": log_mse, "RMSLE": rmsle, "MAE": mae, "RMSE": rmse}  # 표의 한 행.

A_test = np.column_stack([np.ones(len(X_test)), X_test])  # Ridge의 Test 입력에도 절편용 1열을 붙인다.
ridge_pred = A_test @ beta_ridge  # (b) 손으로 구현한 Ridge의 테스트 로그 예측값.
embed_pred = predict_net(embedding_model, cat_test, X_test)  # (c) Embedding 전용 MLP의 테스트 예측값.
hybrid_pred = predict_net(hybrid_model, cat_test, X_test)  # (d) Hybrid MLP의 테스트 예측값.

rows = [  # 모델별 평가 결과를 한 표로 모을 목록.
    evaluate("constant baseline", np.full(len(test), y_train.mean())),  # 비교 기준: 학습 로그 정답 평균만 항상 예측.
    evaluate("target encoding + ridge", ridge_pred),  # (b) Target Encoding + Ridge.
    evaluate("two embeddings + MLP", embed_pred),  # (c) Embedding + MLP.
    evaluate("target encoding + embeddings + MLP", hybrid_pred),  # (d) Hybrid MLP.
]
results = pd.DataFrame(rows)  # 각 모델의 log-MSE, RMSLE, MAE, RMSE를 pandas 표로 바꾼다.
print("\nTEST RESULTS (모든 지표는 작을수록 좋음)")  # 결과 표를 읽는 방법을 안내한다.
print(results.to_string(index=False, float_format=lambda v: f"{v:.6f}"))  # 모델별 지표를 소수점 6자리로 출력한다.
results.to_csv("hw2_retail_results.csv", index=False)  # 제출 결과를 채울 수 있도록 CSV 파일에 저장한다.
print("결과를 hw2_retail_results.csv 에 저장했습니다.")  # 파일 생성 사실을 화면에 알린다.
base_sq = (true_qty - np.expm1(max(y_train.mean(), 0.0))) ** 2  # 기준모델의 행별 원래 단위 제곱오차.
i_max = int(np.argmax(true_qty))  # Test에서 수량이 가장 큰 행.
print("(e) 가장 큰 test Quantity:", int(true_qty[i_max]), "/ 기준모델 제곱오차 중 비중:", f"{base_sq[i_max] / base_sq.sum():.1%}")  # RMSE가 한 행에 좌우되는 정도.

# %% 17. (e) 직관을 Test에서 확인: 학습에서 자주 본 ID일수록 Embedding이 더 유리한가?
# Embedding은 ID마다 학습되는 8개 숫자(룩업 테이블)이므로, 학습에 많이 나온 ID일수록 잘 추정되고
# 처음 보는 ID(0번 unknown)는 정보가 없다. Test 행을 '그 ID가 train에 나온 횟수'로 나눠 log-MSE를 비교한다.
freq_rows = []  # 빈도 구간별 결과.
for col in COLUMNS:  # StockCode와 CustomerID 각각에 대해.
    seen = test[col].map(train[col].value_counts()).fillna(0).to_numpy()  # 각 test 행의 ID가 train에 나온 횟수.
    for name, lo, hi in [("unseen", 0, 0), ("1-9", 1, 9), ("10-99", 10, 99), ("100+", 100, np.inf)]:
        m = (seen >= lo) & (seen <= hi)  # 이 빈도 구간에 속하는 test 행.
        row = {"feature": col, "train_count": name, "test_rows": int(m.sum())}
        for label, pred in [("ridge", ridge_pred), ("embeddings", embed_pred), ("hybrid", hybrid_pred)]:
            row[f"log_MSE_{label}"] = float(np.mean((y_test[m] - np.maximum(pred[m], 0.0)) ** 2)) if m.any() else np.nan
        freq_rows.append(row)
by_freq = pd.DataFrame(freq_rows)  # 구간별 표.
by_freq.to_csv("hw2_retail_by_frequency.csv", index=False)  # 보고서 2(e) 표의 입력 파일.
print("\n(e) test log-MSE by how often the ID appears in train")
print(by_freq.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
