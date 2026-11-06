import sys  # 터미널에서 받은 데이터 파일 이름을 읽는다.
import math  # 정규분포의 상수 pi와 로그를 사용한다.
import numpy as np  # 재현 가능한 데이터 분할을 만든다.
import pandas as pd  # 열 이름이 있는 택시 표를 읽는다.
import torch  # 배열 계산과 자동미분을 담당한다.
from torch import nn  # 신경망 층들을 가져온다.
from torch.nn.functional import softplus  # 실수를 양수인 표준편차로 바꾼다.

torch.manual_seed(7)  # 같은 실행 조건에서 초기 가중치와 배치 순서를 재현한다.
torch.set_num_threads(2)  # CPU 계산에 두 스레드를 사용한다.
df = pd.read_parquet(sys.argv[1])  # 한 행이 한 여행인 표를 연다.
y = torch.tensor(df.pop("trip_duration").to_numpy(), dtype=torch.float32)  # 정답을 분리하여 입력 누출을 막는다.
X = torch.tensor(df.to_numpy(), dtype=torch.float32)  # 남은 일곱 입력을 N행 7열 텐서로 만든다.
idx = np.random.default_rng(42).permutation(len(y))  # 모든 행 번호를 무작위 순서로 섞는다.
tr, va, te = np.split(idx, [int(.70*len(y)), int(.85*len(y))])  # 훈련, 검증, 테스트에 서로 다른 행을 배정한다.
print("split train/validation/test", len(tr), len(va), len(te))  # 분할 크기를 확인한다.
print("test trips >= 45 min", int((y[te] >= 45).sum()))  # 테스트에서 45분 이상 걸린 여행 수다.
net = nn.Sequential(nn.Linear(7,64), nn.ReLU(), nn.Linear(64,2))  # 여행마다 은닉 특징 64개로 두 수를 계산한다.
opt = torch.optim.Adam(net.parameters(), lr=.002)  # 가중치별 갱신 크기를 조절하는 Adam을 만든다.
def predict(x):  # 여러 여행의 조건부 평균과 표준편차를 돌려준다.
    raw = net(x)  # B개 여행에서 B행 2열의 출력을 얻는다.
    return raw[:,0], softplus(raw[:,1]) + .001  # 첫 열은 분 단위 중심, 둘째 열은 양수로 바꾼 폭이다.
def nll(yb, mu, sd):  # 정규분포로 실제 시간들을 얼마나 잘 설명하는지 계산한다.
    return (.5*math.log(2*math.pi)+sd.log()+.5*((yb-mu)/sd)**2).mean()  # 폭 비용과 표준화 잔차 비용을 평균낸다.
best, wait = math.inf, 0  # 처음에는 모든 유한한 검증 점수가 개선이다; wait는 개선 없이 지난 회차 수다.
for epoch in range(300):  # 훈련 자료를 최대 300번 순회한다; 검증 NLL이 더 줄지 않으면 일찍 멈춘다.
    net.train()  # 모델을 학습 모드로 둔다.
    order = torch.as_tensor(tr)[torch.randperm(len(tr))]  # 훈련 행만 다시 섞는다.
    for batch in order.split(2048):  # 마지막 작은 묶음까지 처리한다.
        opt.zero_grad()  # 앞 배치의 기울기를 지운다.
        loss = nll(y[batch], *predict(X[batch]))  # 별표는 평균과 표준편차를 각각 인자로 전달한다.
        loss.backward()  # 손실에서 각 가중치까지 미분한다.
        opt.step()  # 계산한 기울기로 가중치를 한 번 갱신한다.
    net.eval()  # 검증 모드로 전환한다.
    with torch.no_grad():  # 검증에서는 미분을 기록할 필요가 없다.
        score = nll(y[va], *predict(X[va])).item()  # 검증 손실을 파이썬 숫자로 얻는다.
    print("epoch", epoch+1, "validation NLL", round(score, 6))  # 검증 곡선을 로그에 남긴다.
    if score < best:  # 검증 NLL이 줄었는지 비교한다.
        best, chosen_epoch = score, epoch+1  # 가장 좋은 점수와 회차를 기록한다.
        state = {k:v.detach().clone() for k,v in net.state_dict().items()}  # 다음 갱신과 독립적인 가중치 사본을 만든다.
        wait = 0  # 개선되었으므로 대기 횟수를 0으로 되돌린다.
    else:  # 이번 회차에는 검증 NLL이 줄지 않았다.
        wait += 1  # 개선 없는 회차를 하나 센다.
        if wait >= 30:  # 30회 연속 개선이 없으면 더 학습해도 얻을 것이 없다고 본다.
            break  # 조기 종료(early stopping).
print("epochs run", epoch+1)  # 실제로 학습한 회차 수를 출력한다.
net.load_state_dict(state)  # 검증으로 선택한 모델을 복원한다.
def report(name, mu, sd):  # 같은 테스트 행에서 각 모델을 평가한다.
    z = y[te] < 45  # 45분 미만이면 참, 정확히 45분이면 거짓이다.
    p = torch.special.ndtr(((45-mu)/sd).double())  # 표준화된 경계 왼쪽 면적을 구한다.
    decision = p >= .5  # 사건을 예측할 확률 기준을 적용한다.
    brier = ((p-z.double())**2).mean()  # 확률과 실제 0 또는 1 사이 제곱거리를 평균낸다.
    acc = (decision == z).double().mean()  # 맞힌 여행의 비율을 구한다.
    late_hits = ((~decision) & (~z)).sum()  # 실제 45분 이상 중 모델도 그렇게 판단한 건수다.
    print(name, float(nll(y[te],mu,sd)), brier.item(), acc.item(), late_hits.item())  # 모델별 평가 결과를 출력한다.
print("model test_NLL Brier accuracy late_hits")  # 아래 두 줄의 열 이름이다.
with torch.no_grad():  # 최종 테스트는 학습을 끝낸 뒤 수행한다.
    report("constant", y[tr].mean(), y[tr].std(unbiased=False))  # 훈련 평균과 모집단식 표준편차를 쓰는 기준모델이다.
    report("MLP", *predict(X[te]))  # 여행별로 다른 평균과 폭을 사용한다.
print("epoch, validation NLL", chosen_epoch, best)  # 모델 선택의 근거를 출력한다.
