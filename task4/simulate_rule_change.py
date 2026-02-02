import pandas as pd
import numpy as np
import joblib

# 加载模型和特征列
model_data = joblib.load('ultimate_90_percent_model_pure.joblib')
model = model_data['model']
feature_cols = model_data['feature_cols']
accuracy = model_data['accuracy']

# 加载原始数据
score_df = pd.read_csv('enhanced_comprehensive_scores.csv')
fan_df = pd.read_csv('../task1/fan_vote_predictions_enhanced.csv')

# 检查是否包含Bobby Bones的数据
bobby_data = score_df[score_df['celebrity_name'].str.contains('Bobby', case=False)]
if bobby_data.empty:
    print('未找到Bobby Bones的数据')
    # 尝试查找其他名字变体
    print('包含Bobby的数据:')
    print(score_df[score_df['celebrity_name'].str.contains('Bobby', case=False, na=False)]['celebrity_name'].unique())
else:
    print('找到Bobby Bones的数据:')
    print(bobby_data[['season', 'week', 'celebrity_name', 'comprehensive_score', 'rank']].head())

# 查找赛季27的数据（根据反馈内容）
season_27_data = score_df[score_df['season'] == 27]
print(f'\n赛季27的数据数量: {len(season_27_data)}')
print(f'赛季27的选手: {season_27_data["celebrity_name"].unique()}')

# 查看赛季27的投票方法（百分比制）
print(f'\n赛季27的投票方法:')
print(season_27_data[['week', 'judge_score', 'fan_vote', 'comprehensive_score', 'rank']].groupby('week').head(1))

# 模拟规则变更：将百分比制改为排名制
# 对于排名制，综合评分需要重新计算
def simulate_rank_based_scoring(df):
    # 按周分组，计算每个选手的综合评分排名
    df['simulated_comprehensive_score'] = df.groupby('week')['comprehensive_score'].rank(ascending=False)
    # 标准化为0-1范围
    df['simulated_comprehensive_score'] = 1 - (df['simulated_comprehensive_score'] - 1) / (df.groupby('week')['comprehensive_score'].transform('count') - 1)
    return df

# 对赛季27数据进行规则变更模拟
season_27_simulated = simulate_rank_based_scoring(season_27_data.copy())

# 查看模拟结果
print(f'\n规则变更模拟结果（赛季27）:')
print(season_27_simulated[['week', 'celebrity_name', 'comprehensive_score', 'simulated_comprehensive_score']].sort_values(['week', 'celebrity_name']))

# 为模型准备数据
def prepare_model_data(df):
    # 只保留模型使用的特征
    # 注意：我们需要检查feature_cols中的特征是否存在于df中
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].fillna(0)
    # 如果需要添加缺失的特征（用0填充）
    for col in feature_cols:
        if col not in available_features:
            X[col] = 0
    # 确保特征顺序与模型一致
    X = X[feature_cols]
    return X

# 预测淘汰结果
season_27_X = prepare_model_data(season_27_simulated)
season_27_data['elimination_probability'] = model.predict_proba(season_27_X)[:, 1]

# 查看Bobby Bones的淘汰概率
bobby_bones = season_27_data[season_27_data['celebrity_name'].str.contains('Bobby', case=False)]
if not bobby_bones.empty:
    print(f'\nBobby Bones的淘汰概率（周级别）:')
    print(bobby_bones[['week', 'elimination_probability']].sort_values('week'))
    
    # 查找淘汰周数
    bobby_bones['is_eliminated'] = bobby_bones['rank'] == bobby_bones.groupby('week')['rank'].transform('max')
    print(f'\nBobby Bones的淘汰周数:')
    print(bobby_bones[['week', 'is_eliminated']][bobby_bones['is_eliminated']])

# 查看所有选手的淘汰概率
print(f'\n赛季27各选手的平均淘汰概率:')
print(season_27_data.groupby('celebrity_name')['elimination_probability'].mean().sort_values(ascending=False))

# 查找排名前几的淘汰候选选手
top_elimination_candidates = season_27_data.groupby('celebrity_name')['elimination_probability'].mean().sort_values(ascending=False).head(5)
print(f'\n排名前5的淘汰候选选手:')
print(top_elimination_candidates)