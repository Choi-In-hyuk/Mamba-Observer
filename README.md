# Mamba with Observer Variants on CIFAR-10

This repository explores the effect of integrating Luenberger-style observers into the Mamba SSM architecture.  
We compare three setups: vanilla Mamba, Mamba with a **layer-level observer**, and Mamba with an **inner-state observer**.

---

## Results

| Model Variant             | # Parameters | CIFAR-10 Test Accuracy |
|---------------------------|-------------:|-----------------------:|
| **Mamba (baseline)**      |     469,002  | 67.36% |
| **Mamba + Layer Observer**|     630,861  | 68.28% |
| **Mamba + Inner Observer**|     518,218  | 70.00% |

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

## 4. 정리 포인트
- 옵저버를 쓰면 \(A, B, C\)의 **상태 차원이 2배** (증강 상태)  
- \(A\): 원래 분기와 \((A-\Gamma)\) 분기를 블록 대각으로 합침  
- \(B, C\): 각 분기마다 다른 형태로 증강  
- \(D\): 그대로 유지  
- \(\Delta t, B, C\)는 여전히 입력마다 동적으로 결정됨  

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
Mamba 출력 \(h_t \in \mathbb{R}^{B \times L \times D}\)

### 단계:

1. **Global state 추정 (State Estimator)**  
   - 입력 평균 풀링 (sequence mean)  
   \[
   g = \frac{1}{L}\sum_{t=1}^{L} h_t
   \]
   - Linear → SiLU → Dropout → Linear  
   \[
   \hat{s} = f_\text{est}(g) \in \mathbb{R}^{B \times d_\text{state}}
   \]

2. **다음 출력 예측 (Predictor)**  
   - 전역 상태 \(\hat{s}\)로 다음 출력을 예측  
   \[
   \hat{y} = f_\text{pred}(\hat{s}) \in \mathbb{R}^{B \times D}
   \]
   - 길이 L에 복제  
   \[
   \hat{y}_t = \hat{y}, \quad \forall t \in [1,L]
   \]

3. **예측 오차 (Prediction Error)**  
   \[
   e_t = h_t - \hat{y}_t
   \]

4. **보정 생성 (Corrector)**  
   \[
   c_t = f_\text{corr}(e_t)
   \]

5. **Attention으로 보정 가중치 산출**  
   \[
   \alpha_t = \sigma(W e_t) \in (0,1)
   \]

6. **최종 보정된 출력**  
   \[
   h'_t = h_t + \alpha_t \cdot c_t
   \]

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
