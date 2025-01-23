# 파킨슨병 환자 데이터 분석 프로젝트

## 개요
이 프로젝트는 파킨슨병 원격 모니터링 데이터셋을 활용한 머신러닝 분석 프로젝트입니다. `total_UPDRS` 점수 예측을 목표로 합니다.

## 목차
1. [데이터셋 소개](#데이터셋-소개)
2. [프로젝트 구조](#프로젝트-구조)
3. [환경 구성](#환경-구성)
4. [분석 방법](#분석-방법)
5. [실행 방법](#실행-방법)
6. [분석 결과](#분석-결과)

## 데이터셋 소개
- **파일명**: `telemonitoring_parkinsons_updrs.data.csv`
- **주요 컬럼**:
  ```
  subject#, age, sex, test_time, motor_UPDRS, total_UPDRS,
  Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP,
  Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11,
  Shimmer:DDA, NHR, HNR, RPDE, DFA, PPE
  ```
- **타겟 변수**: `total_UPDRS`

## 프로젝트 구조
```
.
├── README.md                     # 프로젝트 설명
├── main.py        # 메인 분석 코드
├── telemonitoring_parkinsons_updrs.data.csv   # 데이터셋
```

## 환경 구성
### 가상환경 설정
```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 가상환경 활성화
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. 필요 패키지 설치
pip install -r requirements.txt
```

### 필수 라이브러리
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
xgboost>=1.4.0
pygam>=0.8.0
shap>=0.39.0
scipy>=1.7.0
```

## 분석 방법
### 1. 지도학습 (Supervised Learning)
- **회귀 모델**:
  - Linear Regression
  - Ridge Regression (alpha 튜닝)
  - Lasso Regression (alpha 튜닝)
  - GAM (Generalized Additive Model)
  - Decision Tree
  - Gradient Boosting
  - XGBoost
  - SVM (Support Vector Machine)

- **모델 평가 지표**:
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score
  - Cross-validation Score

### 2. 비지도학습 (Unsupervised Learning)
- **차원 축소**:
  - PCA (Principal Component Analysis)
    - 주성분 분석을 통한 차원 축소
    - 누적 설명된 분산 비율 분석
    - 2D 시각화

- **군집화**:
  - K-means Clustering
    - Elbow Method로 최적 군집 수 결정
    - Silhouette Score 분석
    - 군집별 특성 분석

## 실행 방법
1. 저장소 클론
```bash
git clone [repository-url]
cd Parkinson-s-Disease
```

2. 가상환경 설정
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. 프로그램 실행
```bash
python main.py
```

## 분석 결과
- **최적 모델**: [XGBoost]
  - MSE: [5.2439]
  - R² Score: [0.9534]
  - Cross-validation Score: [0.9428]

- **군집 분석 결과**:
  - 최적 군집 수: 2
  - 군집별 특성 차이 발견
