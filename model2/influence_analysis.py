import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import load_and_preprocess_data, prepare_features, create_interaction_features
from model_training import train_xgboost_model, cross_validate_model, optimize_model_hyperparameters
from shap_analysis import (calculate_shap_values, plot_summary_plot, plot_shap_dependence,
                          plot_force_plot, get_feature_importance, analyze_interactions,
                          plot_interaction_heatmap, find_controversial_cases)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 使用微软雅黑或黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def main():
    print("=" * 50)
    print("Dancing with the Stars 影响因素分析")
    print("=" * 50)

    # 1. 数据准备
    print("\n步骤1: 数据准备与特征工程...")
    data = load_and_preprocess_data()
    X, y_judge, y_fan, feature_names, preprocessor = prepare_features(data)
    X, feature_names = create_interaction_features(X, feature_names)

    print(f"数据量: {len(data)} 条")
    print(f"特征数量: {len(feature_names)} 个")

    # 2. 模型训练 - 评委得分模型
    print("\n步骤2: 训练评委得分模型...")
    judge_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'objective': 'reg:squarederror',
        'random_state': 42
    }
    judge_model, X_train, X_test, y_train, y_test, y_pred, mse, rmse, r2 = train_xgboost_model(
        X, y_judge, judge_params
    )
    print(f"评委得分模型 - MSE: {mse:.3f}, RMSE: {rmse:.3f}, R^2: {r2:.3f}")

    # 交叉验证
    avg_mse, avg_rmse, avg_r2, mse_scores, r2_scores = cross_validate_model(X, y_judge, judge_params)
    print(f"交叉验证 - 平均R^2: {avg_r2:.3f}")

    # 3. 模型训练 - 粉丝投票模型
    print("\n步骤3: 训练粉丝投票模型...")
    fan_params = {
        'n_estimators': 250,
        'max_depth': 6,
        'learning_rate': 0.08,
        'objective': 'reg:squarederror',
        'random_state': 42
    }
    fan_model, X_train_fan, X_test_fan, y_train_fan, y_test_fan, y_pred_fan, mse_fan, rmse_fan, r2_fan = train_xgboost_model(
        X, y_fan, fan_params
    )
    print(f"粉丝投票模型 - MSE: {mse_fan:.3f}, RMSE: {rmse_fan:.3f}, R^2: {r2_fan:.3f}")

    # 交叉验证
    avg_mse_fan, avg_rmse_fan, avg_r2_fan, mse_scores_fan, r2_scores_fan = cross_validate_model(X, y_fan, fan_params)
    print(f"交叉验证 - 平均R^2: {avg_r2_fan:.3f}")

    # 4. SHAP分析
    print("\n步骤4: SHAP值分析...")

    # 评委得分模型SHAP分析
    judge_explainer, judge_shap_values = calculate_shap_values(judge_model, X, feature_names)
    judge_importance = get_feature_importance(judge_shap_values, feature_names)
    print("\n评委得分模型特征重要性:")
    print(judge_importance.head(10))

    # 粉丝投票模型SHAP分析
    fan_explainer, fan_shap_values = calculate_shap_values(fan_model, X, feature_names)
    fan_importance = get_feature_importance(fan_shap_values, feature_names)
    print("\n粉丝投票模型特征重要性:")
    print(fan_importance.head(10))

    # 5. 可视化
    print("\n步骤5: 生成可视化结果...")
    
    # 确保visualizations目录存在
    vis_dir = 'visualizations'
    import os
    os.makedirs(vis_dir, exist_ok=True)

    # 特征重要性对比
    judge_top = judge_importance.head(10)
    fan_top = fan_importance.head(10)
    combined = pd.merge(judge_top, fan_top, on='feature', how='outer', suffixes=('_judge', '_fan'))
    combined = combined.sort_values(['importance_judge', 'importance_fan'], ascending=False)

    # 准备绘图数据
    plot_data = combined[['feature', 'importance_percent_judge', 'importance_percent_fan']].copy()
    plot_data.set_index('feature', inplace=True)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_data.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_title('特征重要性对比 (Top 10)')
    ax.set_ylabel('重要性百分比 (%)')
    ax.set_xlabel('特征')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_importance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 绘制SHAP summary plots
    plot_summary_plot(judge_shap_values, feature_names, '评委得分模型特征重要性',
                     os.path.join(vis_dir, 'judge_feature_importance.png'))
    plot_summary_plot(fan_shap_values, feature_names, '粉丝投票模型特征重要性',
                     os.path.join(vis_dir, 'fan_feature_importance.png'))

    # 绘制关键特征依赖图
    key_features = ['celebrity_age_during_season', 'pro_experience']
    for feature in key_features:
        if feature in feature_names:
            idx = feature_names.index(feature)
            plot_shap_dependence(judge_shap_values, X, feature_names, idx,
                                os.path.join(vis_dir, f'judge_{feature}_dependence.png'))
            plot_shap_dependence(fan_shap_values, X, feature_names, idx,
                                os.path.join(vis_dir, f'fan_{feature}_dependence.png'))

    # 分析交互效应
    print("\n步骤6: 分析特征交互效应...")
    judge_interactions = analyze_interactions(judge_model, X, feature_names)
    fan_interactions = analyze_interactions(fan_model, X, feature_names)

    print("\n评委得分模型强交互效应:")
    print(judge_interactions)
    print("\n粉丝投票模型强交互效应:")
    print(fan_interactions)

    # 绘制交互效应热力图
    plot_interaction_heatmap(judge_model, X, feature_names,
                            os.path.join(vis_dir, 'judge_interaction_heatmap.png'))
    plot_interaction_heatmap(fan_model, X, feature_names,
                            os.path.join(vis_dir, 'fan_interaction_heatmap.png'))

    # 查找争议案例
    print("\n步骤7: 查找争议案例...")
    controversial_cases = find_controversial_cases(data, y_judge, y_fan)
    print("\n争议案例:")
    print(controversial_cases)

    # 分析特定案例（如 Bristol Palin）
    print("\n步骤8: 分析特定案例...")
    bristol_idx = data[(data['celebrity_name'] == 'Bristol Palin') & (data['season'] == 11)].index
    if len(bristol_idx) > 0:
        idx = bristol_idx[0]
        print(f"\nBristol Palin (Season 11) - 评委得分: {y_judge[idx]:.2f}, 粉丝投票: {y_fan[idx]:.2f}")
        print("注：由于SHAP库兼容性问题，force plot已跳过")

        # 创建简化版的分析结果
        print("\nBristol Palin简化分析:")
        print(f"- 年龄: {data.iloc[idx]['celebrity_age_during_season']}")
        print(f"- 行业: {data.iloc[idx]['celebrity_industry']}")
        print(f"- 专业经验: {data.iloc[idx]['pro_experience']}")

    # 保存分析结果
    judge_importance.to_csv('judge_feature_importance.csv', index=False)
    fan_importance.to_csv('fan_feature_importance.csv', index=False)
    judge_interactions.to_csv('judge_interactions.csv', index=False)
    fan_interactions.to_csv('fan_interactions.csv', index=False)
    controversial_cases.to_csv('controversial_cases.csv', index=False)

    print("\n" + "=" * 50)
    print("分析完成！结果已保存到当前目录")
    print("=" * 50)


if __name__ == "__main__":
    main()