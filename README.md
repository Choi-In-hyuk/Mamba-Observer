# Mamba with Observer Variants on CIFAR-10

This repository explores the effect of integrating Luenberger-style observers into the Mamba SSM architecture.  
We compare three setups: vanilla Mamba, Mamba with a **layer-level observer**, and Mamba with an **inner-state observer**.

---

## Results

| Model Variant             | # Parameters | CIFAR-10 Test Accuracy |
|---------------------------|-------------:|-----------------------:|
| **Mamba (baseline)**      |     469,002  | 67.36% |
| **Mamba + Layer Observer**|     630,861  | 68.28% |
| **Mamba + Inner Observer**|     518,218  | 69.32% |

---

# Mamba + Observer (Diagonal Approx) 정리

## 1. 기본 Mamba (Standard SSM)
$$
x_{t+1} = A x_t + B u_t
$$
$$
y_t = C x_t + D u_t
$$
- \( A = \text{diag}(a) \) (대각 구조)  
- \( B, C, D \)는 입력 \(u_t\)에 따라 토큰별로 달라짐  
- 병렬 selective\_scan 가능  

---

## 2. 정석 Luenberger Observer (병렬 불가)
$$
\hat{x}_{t+1} = (A - L C)\hat{x}_t + (B - L D) u_t + L y_t
$$
$$
\hat{y}_t = C \hat{x}_t + D u_t
$$
- \( L \): 옵저버 이득 행렬  
- \( LC \) 때문에 상태 간 결합이 생겨 병렬 스캔 불가능  

---

## 3. 대각화 근사 Observer 

### (1) 원래 분기 (Original branch)
$$
x_{t+1} = A x_t + B u_t
$$
$$
y^{(1)}_t = C x_t + D u_t
$$

### (2) 옵저버 분기 (Observer branch, diagonal approx)
$$
\hat{x}_{t+1} = (A - \Gamma)\hat{x}_t + \big(B + \Gamma D_{\text{approx}}\big) u_t
$$
$$
y^{(2)}_t = C \hat{x}_t + D u_t
$$
- \(\Gamma = \text{diag}(\gamma)\), \(\gamma \ge 0\) (학습 파라미터)  
- \(D_{\text{approx}}\): \(D\)를 상태 차원으로 맞춘 근사치  

### (3) 최종 출력 (Blending)
$$
y_t = (1-\alpha)\, y^{(1)}_t + \alpha \, y^{(2)}_t
$$
- \(\alpha \in [0,1]\): 블렌딩 비율 (코드에선 0.1로 고정)  

---

# CIFAR-10 Training Results (Mamba with Luenberger Observer, α=0.1)

| Epoch | Train Acc (%) | Test Acc (%) | Best Acc (%) | Train Loss | Test Loss | LR      |
|-------|---------------|--------------|--------------|------------|-----------|---------|
| 1     | 35.89         | 39.97        | 39.97        | 1.7500     | 1.6485    | 0.000994|
| 2     | 47.23         | 46.69        | 46.69        | 1.4672     | 1.4791    | 0.000976|
| 3     | 53.30         | 52.62        | 52.62        | 1.3164     | 1.3408    | 0.000946|
| 4     | 57.27         | 54.39        | 54.39        | 1.2023     | 1.2882    | 0.000905|
| 5     | 60.62         | 57.35        | 57.35        | 1.1164     | 1.1910    | 0.000854|
| 6     | 63.48         | 60.54        | 60.54        | 1.0376     | 1.1170    | 0.000794|
| 7     | 65.64         | 62.56        | 62.56        | 0.9712     | 1.0800    | 0.000727|
| 8     | 67.65         | 63.64        | 63.64        | 0.9192     | 1.0255    | 0.000655|
| 9     | 69.48         | 63.72        | 63.72        | 0.8666     | 1.0366    | 0.000578|
| 10    | 70.98         | 66.17        | 66.17        | 0.8250     | 0.9692    | 0.000500|
| 11    | 72.45         | 66.76        | 66.76        | 0.7827     | 0.9403    | 0.000422|
| 12    | 74.05         | 66.72        | 66.76        | 0.7451     | 0.9459    | 0.000345|
| 13    | 75.21         | 67.61        | 67.61        | 0.7063     | 0.9126    | 0.000273|
| 14    | 76.49         | 68.82        | 68.82        | 0.6677     | 0.8985    | 0.000206|
| 15    | 77.86         | 68.53        | 68.82        | 0.6316     | 0.9032    | 0.000146|
| 16    | 78.94         | 68.84        | 68.84        | 0.6038     | 0.9040    | 0.000095|
| 17    | 79.87         | 69.11        | 69.11        | 0.5773     | 0.8999    | 0.000054|
| 18    | 80.49         | 69.15        | 69.15        | 0.5561     | 0.8969    | 0.000024|
| 19    | 81.17         | 69.27        | 69.27        | 0.5409     | 0.8984    | 0.000006|
| 20    | 81.41         | 69.32        | 69.32        | 0.5345     | 0.8984    | 0.000000|

**Final Best Accuracy**: **69.32%**  
**Observer α**: 0.1


# Mamba SSM 파라미터 정리 (with Observer)

## 1. 기본 SSM (Mamba 구조)

상태 갱신:
$$
x_{t+1} = A x_t + B u_t
$$

출력:
$$
y_t = C x_t + D u_t
$$

- \( A \in \mathbb{R}^{d_{\text{inner}} \times d_{\text{state}}} \) : 상태 전이 행렬 (HiPPO 기반 초기화, 학습 가능)  
- \( B \in \mathbb{R}^{B \times d_{\text{state}} \times L} \) : 입력 의존 행렬 (토큰별로 달라짐)  
- \( C \in \mathbb{R}^{B \times d_{\text{state}} \times L} \) : 출력 의존 행렬 (토큰별로 달라짐)  
- \( D \in \mathbb{R}^{d_{\text{inner}}} \) : skip connection 게인  
- \( \Delta t \in \mathbb{R}^{B \times d_{\text{inner}} \times L} \) : 토큰별 시간 스케일 (softplus로 양수 제약)  

---

## 2. 정석 Luenberger Observer (참고용)

$$
\hat{x}_{t+1} = (A - L C)\hat{x}_t + (B - L D) u_t + L y_t
$$

$$
\hat{y}_t = C \hat{x}_t + D u_t
$$

- \( L \): 옵저버 이득 행렬 (full matrix, selective\_scan 불가능)  

---

## 3. 대각화 근사 Observer (네 코드 방식)

### 증강된 상태 차원
- 옵저버를 켜면 \( d_{\text{state}} \to 2 \times d_{\text{state}} \)  
- selective\_scan은 아래 증강된 파라미터들을 사용  

### Augmented 행렬
$$
A_{\text{aug}} =
\begin{bmatrix}
A & 0 \\
0 & A - \Gamma
\end{bmatrix}
\quad \in \mathbb{R}^{d_{\text{inner}} \times (2 d_{\text{state}})}
$$

$$
B_{\text{aug}} =
\begin{bmatrix}
B \\
B + \Gamma D_{\text{approx}}
\end{bmatrix}
\quad \in \mathbb{R}^{B \times (2 d_{\text{state}}) \times L}
$$

$$
C_{\text{aug}} =
\begin{bmatrix}
(1-\alpha) C \\
\alpha C
\end{bmatrix}
\quad \in \mathbb{R}^{B \times (2 d_{\text{state}}) \times L}
$$

- \(\Gamma = \text{diag}(\gamma)\), \(\gamma \ge 0\) : 학습 가능한 옵저버 게인  
- \(D_{\text{approx}}\) : \(D\)를 상태 차원으로 사상(projection)한 근사치  
- \(\alpha \in [0,1]\) : 출력 블렌딩 비율  

### 최종 출력
$$
y_t = (1-\alpha)\,(C x_t + D u_t) + \alpha\,(C \hat{x}_t + D u_t)
$$

---

# Mamba with Observer (Between Layers)

## 1. 구조 (큰 흐름)

입력 (B, L, D)  
↓  
[Mamba Layer 1]  
↓  
[Observer 1] ← (예상 출력 vs 실제 출력 비교, 보정)  
↓  
[Mamba Layer 2]  
↓  
[Observer 2]  
↓  
...  
↓  
[Mamba Layer N]  
↓  
출력 (B, L, D)

- Mamba 레이어는 **Selective Scan 기반 SSM**  
- 각 Mamba 사이사이에 **Observer 모듈**이 들어가서 출력 보정 수행  

---

## 2. Observer 모듈 내부

### 입력:  
Mamba 출력 \( h_t \in \mathbb{R}^{B \times L \times D} \)

### 단계:

1. **Global state 추정 (State Estimator)**  
   - 입력 평균 풀링 (sequence mean)  
   $$
   g = \frac{1}{L}\sum_{t=1}^{L} h_t
   $$
   - Linear → SiLU → Dropout → Linear  
   $$
   \hat{s} = f_{\text{est}}(g) \in \mathbb{R}^{B \times d_{\text{state}}}
   $$

2. **다음 출력 예측 (Predictor)**  
   - 전역 상태 \(\hat{s}\)로 다음 출력을 예측  
   $$
   \hat{y} = f_{\text{pred}}(\hat{s}) \in \mathbb{R}^{B \times D}
   $$
   - 길이 \(L\)에 복제  
   $$
   \hat{y}_t = \hat{y}, \quad \forall t \in [1,L]
   $$

3. **예측 오차 (Prediction Error)**  
   $$
   e_t = h_t - \hat{y}_t
   $$

4. **보정 생성 (Corrector)**  
   $$
   c_t = f_{\text{corr}}(e_t)
   $$

5. **Attention으로 보정 가중치 산출**  
   $$
   \alpha_t = \sigma(W e_t) \in (0,1)
   $$

6. **최종 보정된 출력**  
   $$
   h'_t = h_t + \alpha_t \cdot c_t
   $$

---

## 3. 요약 (의미적 해석)

- **Mamba 레이어**: 원래 SSM 동작 (sequence modeling)  
- **Observer**: 각 레이어 사이에서  
  - "예상 출력" (\(\hat{y}\))을 만들어보고,  
  - "실제 출력" (\(h\))과 비교해 잔차를 계산,  
  - 잔차 기반으로 보정 신호를 생성,  
  - Attention 가중치로 보정 크기를 조절해 다음 레이어에 전달  

즉, **"레이어 간 예상 vs 실제 출력 차이를 줄이는 보조 경로"**로 작동하는 옵저버임.  
정석 Luenberger 옵저버처럼 \(y-\hat{y}\) 잔차를 쓰되, selective\_scan 구조를 깨지 않도록 **보정 신호만 추가**하는 형태라고 볼 수 있어.



## mamba with observer
<img width="1917" height="962" alt="image" src="https://github.com/user-attachments/assets/d49fa321-2069-4c5a-8f44-27198dfff43c" />
