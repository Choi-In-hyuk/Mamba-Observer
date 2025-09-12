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

## 3. 대각화 근사 Observer (네 코드 방식)

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




## mamba with observer
<img width="1917" height="962" alt="image" src="https://github.com/user-attachments/assets/d49fa321-2069-4c5a-8f44-27198dfff43c" />
