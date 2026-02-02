import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import PartialDependenceDisplay
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model():
    """加载模型"""
    model_data = joblib.load('ultimate_90_percent_model.joblib')
    return model_data['model'], model_data['feature_cols'], model_data['accuracy']

def load_data():
    """加载数据"""
    df = pd.read_csv('ultimate_90_percent_scores.csv')
    return df

def analyze_feature_importance(model, feature_cols):
    """分析特征重要性"""
    print("特征重要性分析:")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance_df.to_string(index=False))
    print()
    
    return importance_df

def sensitivity_analysis(model, feature_cols, data):
    """敏感性分析"""
    print("开始敏感性分析...")
    
    # 创建输出目录
    if not os.path.exists('sensitivity_analysis'):
        os.makedirs('sensitivity_analysis')
    
    # 1. 部分依赖图 (PDP)
    print("生成部分依赖图...")
    X = data[feature_cols].fillna(0)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    
    for i, feature in enumerate(feature_cols):
        row = i // 4
        col = i % 4
        PartialDependenceDisplay.from_estimator(
            model, X, [i], ax=axes[row, col], kind='both'
        )
        axes[row, col].set_title(f'部分依赖图: {feature}')
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel('淘汰概率')
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis/partial_dependence_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 特征扰动分析
    print("进行特征扰动分析...")
    base_predictions = model.predict_proba(X)[:, 1]
    base_mean = base_predictions.mean()
    
    sensitivity_results = []
    
    for feature in feature_cols:
        X_permuted = X.copy()
        # 随机打乱特征值
        X_permuted[feature] = np.random.permutation(X_permuted[feature])
        permuted_predictions = model.predict_proba(X_permuted)[:, 1]
        permuted_mean = permuted_predictions.mean()
        # 计算预测值的变化
        prediction_change = np.abs(permuted_predictions - base_predictions).mean()
        mean_change = np.abs(permuted_mean - base_mean)
        
        sensitivity_results.append({
            'feature': feature,
            'base_mean': base_mean,
            'permuted_mean': permuted_mean,
            'mean_change': mean_change,
            'prediction_change': prediction_change
        })
    
    sensitivity_df = pd.DataFrame(sensitivity_results).sort_values('prediction_change', ascending=False)
    print("\n特征扰动分析结果:")
    print(sensitivity_df.to_string(index=False))
    print()
    
    # 保存敏感性分析结果
    sensitivity_df.to_csv('sensitivity_analysis/sensitivity_results.csv', index=False)
    
    # 3. 可视化敏感性分析结果
    print("生成敏感性分析可视化...")
    
    # 特征重要性条形图
    importance_df = analyze_feature_importance(model, feature_cols)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.sort_values('importance', ascending=False))
    plt.title('特征重要性分析', fontsize=16)
    plt.xlabel('重要性', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.tight_layout()
    plt.savefig('sensitivity_analysis/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 敏感性分析结果条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='prediction_change', y='feature', data=sensitivity_df.sort_values('prediction_change', ascending=False))
    plt.title('特征扰动敏感性分析', fontsize=16)
    plt.xlabel('平均预测变化', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.tight_layout()
    plt.savefig('sensitivity_analysis/feature_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return sensitivity_df

def generate_report(sensitivity_df, importance_df, model_accuracy):
    """生成敏感性分析报告"""
    print("生成敏感性分析报告...")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("终极模型敏感性分析报告")
    report_lines.append("="*80)
    report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"模型准确率: {model_accuracy:.2%}")
    report_lines.append("")
    
    # 1. 特征重要性
    report_lines.append("1. 特征重要性分析")
    report_lines.append("-"*40)
    for idx, row in importance_df.iterrows():
        report_lines.append(f"  • {row['feature']}: {row['importance']:.4f}")
    report_lines.append("")
    
    # 2. 敏感性分析结果
    report_lines.append("2. 特征扰动敏感性分析")
    report_lines.append("-"*40)
    for idx, row in sensitivity_df.iterrows():
        report_lines.append(f"  • {row['feature']}:")
        report_lines.append(f"    基础平均概率: {row['base_mean']:.4f}")
        report_lines.append(f"    扰动后平均概率: {row['permuted_mean']:.4f}")
        report_lines.append(f"    平均预测变化: {row['prediction_change']:.4f}")
        report_lines.append("")
    
    # 3. 关键发现
    report_lines.append("3. 关键发现")
    report_lines.append("-"*40)
    most_sensitive = sensitivity_df.iloc[0]['feature']
    least_sensitive = sensitivity_df.iloc[-1]['feature']
    report_lines.append(f"  • 最敏感特征: {most_sensitive}")
    report_lines.append(f"  • 最不敏感特征: {least_sensitive}")
    report_lines.append(f"  • 平均预测变化范围: {sensitivity_df['prediction_change'].min():.4f} - {sensitivity_df['prediction_change'].max():.4f}")
    report_lines.append("")
    
    # 4. 结论与建议
    report_lines.append("4. 结论与建议")
    report_lines.append("-"*40)
    report_lines.append("本敏感性分析揭示了各特征对淘汰预测结果的影响程度。")
    report_lines.append("模型对某些特征的变化较为敏感，这些特征在预测过程中起着重要作用。")
    report_lines.append("")
    report_lines.append("建议:")
    report_lines.append("  • 重点关注最敏感的特征，确保其数据质量")
    report_lines.append("  • 对敏感特征进行更精细的特征工程")
    report_lines.append("  • 在模型部署时，监控敏感特征的输入值范围")
    report_lines.append("")
    
    report_lines.append("="*80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    with open('sensitivity_analysis/sensitivity_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("报告已保存到 sensitivity_analysis/sensitivity_report.txt")
    
    return report_text

def main():
    """主函数"""
    print("="*80)
    print("终极模型敏感性分析")
    print("="*80)
    
    # 加载模型和数据
    model, feature_cols, accuracy = load_model()
    data = load_data()
    
    print(f"模型加载成功，准确率: {accuracy:.2%}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"数据条数: {len(data)}")
    print()
    
    # 进行敏感性分析
    sensitivity_df = sensitivity_analysis(model, feature_cols, data)
    
    # 分析特征重要性
    importance_df = analyze_feature_importance(model, feature_cols)
    
    # 生成报告
    generate_report(sensitivity_df, importance_df, accuracy)
    
    print("\n" + "="*80)
    print("敏感性分析完成!")
    print("="*80)
    print("输出文件:")
    print("1. sensitivity_analysis/partial_dependence_plots.png - 部分依赖图")
    print("2. sensitivity_analysis/feature_importance.png - 特征重要性条形图")
    print("3. sensitivity_analysis/feature_sensitivity.png - 特征扰动敏感性分析")
    print("4. sensitivity_analysis/sensitivity_results.csv - 敏感性分析结果")
    print("5. sensitivity_analysis/sensitivity_report.txt - 敏感性分析报告")
    print("="*80)

if __name__ == "__main__":
    main()