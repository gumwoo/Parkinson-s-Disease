import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 필요한 패키지 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 머신러닝 관련 패키지
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# pygam 및 shap 패키지 임포트 시도
try:
    from pygam import LinearGAM, s
    import shap
except ImportError:
    print("pygam 또는 shap 패키지가 설치되어 있지 않습니다.")
    print("pip install pygam shap 명령어로 설치할 수 있습니다.")
    sys.exit(1)

def load_data(file_path):
    """데이터 로드 함수"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        
        columns = ["subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS",
                  "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
                  "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
                  "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"]
        
        data = pd.read_csv(file_path, names=columns, skiprows=1)
        return data
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {str(e)}")
        sys.exit(1)

def main():
    # 데이터 파일 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'telemonitoring_parkinsons_updrs.data.csv')
    
    # 데이터 로드
    data = load_data(data_file)
    
    # 결측치 확인
    print("Missing values:\n", data.isnull().sum())

    # 데이터 기본 정보 출력
    print("\nDataset Info:")
    print(data.info())
    print("\nDataset Description:")
    print(data.describe())

    # 이상치 처리 함수 (IQR 방식)
    def remove_outliers(df, columns):
        df_clean = df.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        return df_clean

    # 타겟 변수 제외한 수치형 컬럼에 대해 이상치 처리
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns.drop(['motor_UPDRS', 'total_UPDRS', 'subject#'])
    data_clean = remove_outliers(data, numeric_columns)

    # 특성과 타겟 분리
    X = data_clean.drop(["motor_UPDRS", "total_UPDRS", "subject#"], axis=1)
    y = data_clean["total_UPDRS"]

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # 2. 확장된 EDA
    # 상관관계 히트맵
    plt.figure(figsize=(15, 10))
    sns.heatmap(data_clean.corr(), cmap='RdBu', center=0, annot=True, fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # 각 변수의 분포 시각화
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(X.columns, 1):
        plt.subplot(5, 4, i)
        sns.histplot(data_clean[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

    # 주요 특성과 목표변수 관계
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(X.columns[:6], 1):  # 상위 6개 특성만 표시
        plt.subplot(2, 3, i)
        plt.scatter(data_clean[col], y, alpha=0.5)
        plt.xlabel(col)
        plt.ylabel('total_UPDRS')
    plt.tight_layout()
    plt.show()

    # 3. 데이터 분할
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, do_cv=True):
        # 예측
        pred = model.predict(X_test)

        # 평가 지표
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)

        # Cross-validation 점수
        if do_cv:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            cv_mean = np.nan
            cv_std = np.nan

        # 예측값 vs 실제값 플롯
        plt.figure(figsize=(20, 5))

        # 1. 예측값 vs 실제값 플롯
        plt.subplot(1, 5, 1)
        plt.scatter(y_test, pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Predicted vs Actual')

        # 2. 잔차 플롯
        residuals = y_test - pred
        plt.subplot(1, 5, 2)
        plt.scatter(pred, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name}: Residuals Plot')

        # 3. 잔차 분포 히스토그램
        plt.subplot(1, 5, 3)
        sns.histplot(residuals, kde=True, bins=30, color='orange')
        plt.xlabel('Residuals')
        plt.title(f'{model_name}: Residuals Distribution')

        # 4. 예측값과 실제값의 밀도 등고선 플롯
        plt.subplot(1, 5, 4)
        sns.kdeplot(x=y_test, y=pred, cmap='Blues', fill=True, thresh=0.05)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Prediction Density Contour')

        # 5. 3D 잔차 플롯
        ax = plt.subplot(1, 5, 5, projection='3d')
        feature_index = 0  # 원하는 특성 인덱스로 변경 (예: 'Jitter(%)'의 인덱스)

        # X_test가 DataFrame인지 numpy 배열인지 확인
        if isinstance(X_test, pd.DataFrame):
            feature_name = X_test.columns[feature_index]
            feature_values = X_test.iloc[:, feature_index]
        elif isinstance(X_test, np.ndarray):
            feature_name = f'Feature {feature_index}'
            feature_values = X_test[:, feature_index]
        else:
            feature_name = f'Feature {feature_index}'
            feature_values = X_test[:, feature_index]

        ax.scatter(feature_values, pred, residuals, alpha=0.5)
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Predicted Values')
        ax.set_zlabel('Residuals')
        ax.set_title(f'{model_name}: 3D Residuals ({feature_name})')

        plt.tight_layout()
        plt.show()

        # SHAP 값 계산 및 시각화
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_') or isinstance(model, LinearGAM):
            try:
                if isinstance(model, LinearGAM):
                    explainer = shap.LinearExplainer(model, X_train)
                else:
                    explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)

                # SHAP 요약 플롯
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                plt.title(f'{model_name}: SHAP Feature Importance')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"SHAP 값 계산 중 오류 발생 for {model_name}: {e}")
        else:
            print(f"{model_name}에는 SHAP 값을 계산할 수 있는 속성이 없습니다.")

        return mse, mae, rmse, r2, cv_mean, cv_std

    """# 5. 모델링"""

    results = {}

    # (1) Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    mse_lr, mae_lr, rmse_lr, r2_lr, cv_lr_mean, cv_lr_std = evaluate_model(lr, X_train, X_test, y_train, y_test, "Linear Regression")
    results['Linear Regression'] = {'mse': mse_lr, 'mae': mae_lr, 'rmse': rmse_lr, 'r2': r2_lr, 'cv_mean': cv_lr_mean, 'cv_std': cv_lr_std}

    # (2) Ridge
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    ridge = Ridge()
    ridge_gs = GridSearchCV(ridge, param_grid={'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
    ridge_gs.fit(X_train, y_train)
    mse_ridge, mae_ridge, rmse_ridge, r2_ridge, cv_ridge_mean, cv_ridge_std = evaluate_model(ridge_gs.best_estimator_, X_train, X_test, y_train, y_test, "Ridge")
    results['Ridge'] = {'mse': mse_ridge, 'mae': mae_ridge, 'rmse': rmse_ridge, 'r2': r2_ridge, 'cv_mean': cv_ridge_mean, 'cv_std': cv_ridge_std}
    print(f"Best Ridge alpha: {ridge_gs.best_params_}")

    # (3) Lasso
    lasso = Lasso(max_iter=10000)
    lasso_gs = GridSearchCV(lasso, param_grid={'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
    lasso_gs.fit(X_train, y_train)
    mse_lasso, mae_lasso, rmse_lasso, r2_lasso, cv_lasso_mean, cv_lasso_std = evaluate_model(lasso_gs.best_estimator_, X_train, X_test, y_train, y_test, "Lasso")
    results['Lasso'] = {'mse': mse_lasso, 'mae': mae_lasso, 'rmse': rmse_lasso, 'r2': r2_lasso, 'cv_mean': cv_lasso_mean, 'cv_std': cv_lasso_std}
    print(f"Best Lasso alpha: {lasso_gs.best_params_}")

    # (4) GAM
    gam = LinearGAM(
        s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) +
        s(9) + s(10) + s(11) + s(12) + s(13) + s(14) + s(15) + s(16) + s(17)
    )
    gam.gridsearch(X_train.values, y_train)  # numpy 배열 전달

    # evaluate_model에 numpy 배열 전달
    mse_gam, mae_gam, rmse_gam, r2_gam, cv_gam_mean, cv_gam_std = evaluate_model(
        gam, X_train.values, X_test.values, y_train, y_test, "GAM", do_cv=False
    )
    results['GAM'] = {
        'mse': mse_gam,
        'mae': mae_gam,
        'rmse': rmse_gam,
        'r2': r2_gam,
        'cv_mean': cv_gam_mean,
        'cv_std': cv_gam_std
    }

    # (5) Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    params_dt = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8]
    }
    dt_gs = GridSearchCV(dt, param_grid=params_dt, cv=5, scoring='neg_mean_squared_error')
    dt_gs.fit(X_train, y_train)
    mse_dt, mae_dt, rmse_dt, r2_dt, cv_dt_mean, cv_dt_std = evaluate_model(dt_gs.best_estimator_, X_train, X_test, y_train, y_test, "Decision Tree")
    results['Decision Tree'] = {'mse': mse_dt, 'mae': mae_dt, 'rmse': rmse_dt, 'r2': r2_dt, 'cv_mean': cv_dt_mean, 'cv_std': cv_dt_std}
    print(f"Best Decision Tree parameters: {dt_gs.best_params_}")

    # (6) Gradient Boosting
    gb = GradientBoostingRegressor(random_state=42)
    params_gb = {
        'n_estimators': [100, 300],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }
    gb_gs = GridSearchCV(gb, param_grid=params_gb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    gb_gs.fit(X_train, y_train)
    mse_gb, mae_gb, rmse_gb, r2_gb, cv_gb_mean, cv_gb_std = evaluate_model(gb_gs.best_estimator_, X_train, X_test, y_train, y_test, "Gradient Boosting")
    results['Gradient Boosting'] = {'mse': mse_gb, 'mae': mae_gb, 'rmse': rmse_gb, 'r2': r2_gb, 'cv_mean': cv_gb_mean, 'cv_std': cv_gb_std}
    print(f"Best Gradient Boosting parameters: {gb_gs.best_params_}")

    # (7) XGBoost
    xgb = XGBRegressor(random_state=42)
    params_xgb = {
        'n_estimators': [100, 300],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.7, 1.0]
    }
    xgb_gs = GridSearchCV(xgb, param_grid=params_xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    xgb_gs.fit(X_train, y_train)
    mse_xgb, mae_xgb, rmse_xgb, r2_xgb, cv_xgb_mean, cv_xgb_std = evaluate_model(xgb_gs.best_estimator_, X_train, X_test, y_train, y_test, "XGBoost")
    results['XGBoost'] = {'mse': mse_xgb, 'mae': mae_xgb, 'rmse': rmse_xgb, 'r2': r2_xgb, 'cv_mean': cv_xgb_mean, 'cv_std': cv_xgb_std}
    print(f"Best XGBoost parameters: {xgb_gs.best_params_}")

    # (8) SVM
    svm = SVR()
    params_svm = {
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    svm_gs = GridSearchCV(svm, param_grid=params_svm, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    svm_gs.fit(X_train, y_train)
    mse_svm, mae_svm, rmse_svm, r2_svm, cv_svm_mean, cv_svm_std = evaluate_model(svm_gs.best_estimator_, X_train, X_test, y_train, y_test, "SVM")
    results['SVM'] = {'mse': mse_svm, 'mae': mae_svm, 'rmse': rmse_svm, 'r2': r2_svm, 'cv_mean': cv_svm_mean, 'cv_std': cv_svm_std}
    print(f"Best SVM parameters: {svm_gs.best_params_}")

    # 6. 피처 중요도 시각화 (Gradient Boosting 및 XGBoost)
    # (6.1) Gradient Boosting
    if hasattr(gb_gs.best_estimator_, 'feature_importances_'):
        feature_importance_gb = pd.DataFrame({
            'feature': X.columns,
            'importance': gb_gs.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_gb, x='importance', y='feature')
        plt.title('Feature Importance (Gradient Boosting)')
        plt.tight_layout()
        plt.savefig('feature_importance_gb.png', dpi=300, bbox_inches='tight')
        plt.show()

    # (6.2) XGBoost
    if hasattr(xgb_gs.best_estimator_, 'feature_importances_'):
        feature_importance_xgb = pd.DataFrame({
            'feature': X.columns,
            'importance': xgb_gs.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_xgb, x='importance', y='feature')
        plt.title('Feature Importance (XGBoost)')
        plt.tight_layout()
        plt.savefig('feature_importance_xgb.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 모델 매핑 딕셔너리 생성 (Linear Regression과 GAM 포함)
    model_mapping = {
        'Linear Regression': lr,
        'Ridge': ridge_gs,
        'Lasso': lasso_gs,
        'GAM': gam,
        'Decision Tree': dt_gs,
        'Gradient Boosting': gb_gs,
        'XGBoost': xgb_gs,
        'SVM': svm_gs
    }

    # 6. 비지도학습 (PCA)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # 설명된 분산 비율 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Analysis')
    plt.grid(True)
    plt.show()

    # 첫 2개 주성분으로 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Visualization with Target Variable')
    plt.colorbar(label='total_UPDRS')
    plt.show()

    # 7. 비지도학습 추가 (K-Means Clustering)

    # Elbow Method
    def plot_elbow_method(X, max_clusters=10):
        sse = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            sse.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), sse, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('SSE')
        plt.title('Elbow Method For Optimal k')
        plt.xticks(range(1, max_clusters + 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Silhouette Score
    def plot_silhouette_scores(X, max_clusters=10):
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)

        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', color='orange')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores For Optimal k')
        plt.xticks(range(2, max_clusters + 1))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 최적의 클러스터 수 시각화
    plot_elbow_method(X_scaled, max_clusters=10)
    plot_silhouette_scores(X_scaled, max_clusters=10)

    # 최적의 클러스터 수 결정
    optimal_k = 2
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X_scaled)
    cluster_labels = kmeans.labels_

    # 클러스터 라벨을 데이터프레임에 추가
    data_clean['Cluster'] = cluster_labels

    # 클러스터별 통계량 확인
    print(data_clean.groupby('Cluster').mean())

    # PCA를 통한 2차원 시각화
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=cluster_labels, palette='viridis', alpha=0.6)
    plt.title(f'K-Means Clustering (k={optimal_k}) with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()

    # 클러스터별 타겟 변수 비교
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='total_UPDRS', data=data_clean, palette='viridis')
    plt.title('Distribution of total_UPDRS Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('total_UPDRS')
    plt.tight_layout()
    plt.show()

    # 7. 결과 시각화
    model_names = list(results.keys())
    mse_values = [results[model]['mse'] for model in model_names]
    r2_values = [results[model]['r2'] for model in model_names]

    # MSE 비교
    plt.figure(figsize=(12, 6))
    sns.barplot(x=model_names, y=mse_values)
    plt.title("Model Comparison - MSE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # R² 비교
    plt.figure(figsize=(12, 6))
    sns.barplot(x=model_names, y=r2_values)
    plt.title("Model Comparison - R²")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # 모델별 MSE와 CV Score를 하나의 그래프에 시각화하기
    models = list(results.keys())
    mse_values = [results[m]['mse'] for m in models]
    cv_means = [results[m]['cv_mean'] for m in models]
    cv_stds = [results[m]['cv_std'] for m in models]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    bar_width = 0.35

    # Test MSE 막대
    plt.bar(x - bar_width/2, mse_values, bar_width, label='Test MSE', color='lightcoral', alpha=0.7)
    # CV MSE 막대
    plt.bar(x + bar_width/2, cv_means, bar_width, label='CV MSE', color='lightblue', alpha=0.7)

    plt.xlabel('Models')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Performance Comparison (MSE & CV Score)')
    plt.xticks(x, models, rotation=45)
    plt.legend()

    # 막대 위에 값 표시
    for i, model in enumerate(models):
        plt.text(i - bar_width/2, mse_values[i], f'{mse_values[i]:.2f}', ha='center', va='bottom')
        plt.text(i + bar_width/2, cv_means[i], f'{cv_means[i]:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # 최적 모델 식별
    best_model_name = min(results, key=lambda x: results[x]['mse'])
    best_model = model_mapping[best_model_name]
    print("\nBest model:", best_model_name)
    print(f"MSE: {results[best_model_name]['mse']:.3f}")
    print(f"R²: {results[best_model_name]['r2']:.3f}")
    print(f"Cross-validation score: {results[best_model_name]['cv_mean']:.3f}")

    # 결과를 데이터프레임으로 변환하여 자세히 출력
    results_df = pd.DataFrame({
        'Model': model_names,
        'MSE': [results[model]['mse'] for model in model_names],
        'MAE': [results[model]['mae'] for model in model_names],
        'RMSE': [results[model]['rmse'] for model in model_names],
        'R²': [results[model]['r2'] for model in model_names],
        'CV Score': [results[model]['cv_mean'] for model in model_names]
    }).sort_values('MSE')

    # 결과 출력
    print("\n=== Detailed Model Evaluation Results ===")
    print("\nSorted by MSE (lower is better):")
    print(results_df.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x)))
    print("\n=== 모델별 성능 평가 결과 ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Test MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"평균 CV Score: {metrics['cv_mean']:.4f}")
        if model_name in model_mapping:
            model = model_mapping[model_name]
            if isinstance(model, GridSearchCV):
                print(f"최적 파라미터: {model.best_params_}")
            elif model_name == 'Linear Regression':
                print("최적 파라미터: 없음 (기본 설정)")
            elif model_name == 'GAM':
                print("최적 파라미터: GAM이 기본 설정으로 적합됨")
            else:
                print("최적 파라미터 정보가 없습니다.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {str(e)}")
        sys.exit(1)