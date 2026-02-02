import pandas as pd
import numpy as np
import joblib
import shap

# 加载模型和特征列
model_data = joblib.load('ultimate_90_percent_model_pure.joblib')
model = model_data['model']
feature_cols = model_data['feature_cols']
accuracy = model_data['accuracy']

# 加载训练数据
shap_df = pd.read_csv('sensitivity_analysis_improved/shap_values.csv')
X = shap_df[feature_cols]

# 加载原始数据以获取season和week信息
score_df = pd.read_csv('enhanced_comprehensive_scores.csv')
fan_df = pd.read_csv('../task1/fan_vote_predictions_enhanced.csv')

# 创建数据集时需要识别Bottom Two周
# 从dwts_rank_bottom_two_processed.csv文件中读取Bottom Two周信息
bottom_two_df = pd.read_csv('../task1/dwts_rank_bottom_two_processed.csv')

# 识别Bottom Two周的选手和周数
bottom_two_weeks = set()
for index, row in bottom_two_df.iterrows():
    if row['is_bottom_two']:
        # 转换为整数以匹配enhanced_comprehensive_scores.csv中的周数
        bottom_two_weeks.add((row['season'], int(row['eliminated_week'])))

# 在训练数据中添加is_bottom_two_week字段
# 首先需要重构训练数据的season和week信息
temp_ml = []
for (season, week), week_group in score_df.groupby(['season', 'week']):
    if week < score_df['week'].max():
        next_week_players = set(fan_df[(fan_df['season'] == season) & (fan_df['week'] == week + 1)]['contestant'])
        
        for _, row in week_group.iterrows():
            player_name = row['celebrity_name']
            is_eliminated = 1 if player_name not in next_week_players else 0
            
            temp_ml.append({
                'season': season,
                'week': week,
                'player': player_name,
                'is_bottom_two_week': (season, week) in bottom_two_weeks
            })

temp_ml_df = pd.DataFrame(temp_ml)

# 现在我们需要将这个字段与shap_df合并
# 由于我们没有直接的选手标识符，我们需要通过其他方式匹配
# 这里我们假设数据是按顺序排列的
shap_df['is_bottom_two_week'] = temp_ml_df['is_bottom_two_week']

# 检查匹配是否成功
print(f'总记录数: {len(shap_df)}')
print(f'Bottom Two周记录数: {shap_df["is_bottom_two_week"].sum()}')

# 分层SHAP对比
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 对于分类模型，shap_values返回二维数组，取类别1的SHAP值
if isinstance(shap_values, list) and len(shap_values) == 2:
    shap_values = shap_values[1]

# 常规周SHAP分析
X_regular = X[shap_df['is_bottom_two_week'] == False]
shap_regular = shap_values[shap_df['is_bottom_two_week'] == False]

# Bottom Two周SHAP分析
X_bt = X[shap_df['is_bottom_two_week'] == True]
shap_bt = shap_values[shap_df['is_bottom_two_week'] == True]

# 计算平均绝对值SHAP
shap_regular_mean = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_regular).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

shap_bt_mean = pd.DataFrame({
    'feature': feature_cols,
    'mean_abs_shap': np.abs(shap_bt).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)

# 计算judge_score在Bottom Two周的|SHAP|均值 / 常规周的比值
judge_regular = shap_regular_mean[shap_regular_mean['feature'] == 'judge_score']['mean_abs_shap'].values[0]
judge_bt = shap_bt_mean[shap_bt_mean['feature'] == 'judge_score']['mean_abs_shap'].values[0]
judge_ratio = judge_bt / judge_regular

# 计算fan_vote在Bottom Two周的|SHAP|均值 / 常规周的比值
fan_regular = shap_regular_mean[shap_regular_mean['feature'] == 'fan_vote']['mean_abs_shap'].values[0]
fan_bt = shap_bt_mean[shap_bt_mean['feature'] == 'fan_vote']['mean_abs_shap'].values[0]
fan_ratio = fan_bt / fan_regular

print(f'\njudge_score在Bottom Two周的平均绝对值SHAP值: {judge_bt:.4f}')
print(f'judge_score在常规周的平均绝对值SHAP值: {judge_regular:.4f}')
print(f'judge_score在Bottom Two周的SHAP值是常规周的 {judge_ratio:.2f} 倍')
print()
print(f'fan_vote在Bottom Two周的平均绝对值SHAP值: {fan_bt:.4f}')
print(f'fan_vote在常规周的平均绝对值SHAP值: {fan_regular:.4f}')
print(f'fan_vote在Bottom Two周的SHAP值是常规周的 {fan_ratio:.2f} 倍')
print()
print('Bottom Two周SHAP特征重要性排名:')
print(shap_bt_mean.to_string(index=False))

# 保存分析结果
results = {
    'judge_score': {
        'bt': judge_bt,
        'regular': judge_regular,
        'ratio': judge_ratio
    },
    'fan_vote': {
        'bt': fan_bt,
        'regular': fan_regular,
        'ratio': fan_ratio
    },
    'shap_bt_mean': shap_bt_mean,
    'shap_regular_mean': shap_regular_mean
}

# 可视化Bottom Two周与常规周的SHAP对比
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建对比图
comparison_df = pd.DataFrame({
    'feature': feature_cols,
    'bottom_two': np.abs(shap_bt).mean(axis=0),
    'regular': np.abs(shap_regular).mean(axis=0)
})

comparison_df = comparison_df.melt(id_vars='feature', var_name='week_type', value_name='mean_abs_shap')

plt.figure(figsize=(12, 6))
sns.barplot(x='mean_abs_shap', y='feature', hue='week_type', data=comparison_df.sort_values('mean_abs_shap', ascending=False))
plt.title('Bottom Two周与常规周的平均绝对值SHAP值对比')
plt.xlabel('平均绝对值SHAP值')
plt.ylabel('特征')
plt.legend(title='周类型')
plt.tight_layout()
plt.savefig('sensitivity_analysis_improved/shap_comparison_bottom_two_vs_regular.png', dpi=300, bbox_inches='tight')

# 保存数据到CSV
shap_bt_mean.to_csv('sensitivity_analysis_improved/shap_bt_mean.csv', index=False)
shap_regular_mean.to_csv('sensitivity_analysis_improved/shap_regular_mean.csv', index=False)