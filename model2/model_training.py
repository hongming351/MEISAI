import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def train_xgboost_model(X, y, params=None):
    """
    训练XGBoost回归模型
    """
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'random_state': 42
        }

    model = xgb.XGBRegressor(**params)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return model, X_train, X_test, y_train, y_test, y_pred, mse, rmse, r2


def cross_validate_model(X, y, params=None, cv=10):
    """
    交叉验证模型
    """
    if params is None:
        params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'random_state': 42
        }

    model = xgb.XGBRegressor(**params)

    # 10折交叉验证
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    mse_scores = cross_val_score(
        model, X, y, scoring='neg_mean_squared_error', cv=kf
    )
    r2_scores = cross_val_score(
        model, X, y, scoring='r2', cv=kf
    )

    # 计算平均指标
    avg_mse = -np.mean(mse_scores)
    avg_rmse = np.sqrt(avg_mse)
    avg_r2 = np.mean(r2_scores)

    return avg_mse, avg_rmse, avg_r2, mse_scores, r2_scores


def optimize_model_hyperparameters(X, y):
    """
    优化模型超参数（简单网格搜索）
    """
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    best_score = float('-inf')
    best_params = {}

    for depth in param_grid['max_depth']:
        for lr in param_grid['learning_rate']:
            for n_est in param_grid['n_estimators']:
                for subsample in param_grid['subsample']:
                    for colsample in param_grid['colsample_bytree']:
                        params = {
                            'max_depth': depth,
                            'learning_rate': lr,
                            'n_estimators': n_est,
                            'subsample': subsample,
                            'colsample_bytree': colsample,
                            'objective': 'reg:squarederror',
                            'random_state': 42
                        }

                        # 交叉验证
                        _, _, r2, _, _ = cross_validate_model(X, y, params)

                        if r2 > best_score:
                            best_score = r2
                            best_params = params

    return best_params, best_score