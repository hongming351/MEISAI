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

# 查找赛季27的数据（根据反馈内容）
season_27_data = score_df[score_df['season'] == 27].copy()

# 定义排名制综合评分计算方法
def calculate_rank_based_score(df):
    # 对于排名制，我们需要使用judge_score和fan_vote的排名来计算综合评分
    # 这里我们假设judge_score和fan_vote的权重都是0.5
    
    # 计算评委排名（越高越好）
    df['judge_rank'] = df.groupby('week')['judge_score'].rank(ascending=False)
    # 计算粉丝排名（越高越好）
    df['fan_rank'] = df.groupby('week')['fan_vote'].rank(ascending=False)
    
    # 标准化排名为0-1范围（1表示最好）
    df['judge_score_norm'] = 1 - (df['judge_rank'] - 1) / (df.groupby('week')['judge_score'].transform('count') - 1)
    df['fan_score_norm'] = 1 - (df['fan_rank'] - 1) / (df.groupby('week')['fan_vote'].transform('count') - 1)
    
    # 计算综合评分
    df['rank_based_comprehensive_score'] = (df['judge_score_norm'] + df['fan_score_norm']) / 2
    return df

# 对赛季27数据进行排名制综合评分计算
season_27_rank_based = calculate_rank_based_score(season_27_data)

# 查看计算结果
print('赛季27排名制综合评分计算结果:')
print(season_27_rank_based[['week', 'celebrity_name', 'judge_score', 'fan_vote', 
                          'judge_rank', 'fan_rank', 'judge_score_norm', 'fan_score_norm', 
                          'rank_based_comprehensive_score']].sort_values(['week', 'rank_based_comprehensive_score'], ascending=[True, False]))

# 保存原始综合评分，用于比较
season_27_rank_based['original_comprehensive_score'] = season_27_rank_based['comprehensive_score']

# 为模型准备数据
def prepare_model_data(df):
    # 只保留模型使用的特征
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].fillna(0)
    
    # 如果模型使用comprehensive_score特征，我们需要替换为排名制综合评分
    if 'comprehensive_score' in feature_cols:
        X['comprehensive_score'] = df['rank_based_comprehensive_score']
    
    # 添加缺失的特征（用0填充）
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    
    # 确保特征顺序与模型一致
    X = X[feature_cols]
    return X

# 预测淘汰结果
season_27_X = prepare_model_data(season_27_rank_based)
season_27_rank_based['elimination_probability'] = model.predict_proba(season_27_X)[:, 1]

# 查看Bobby Bones的淘汰概率
bobby_bones = season_27_rank_based[season_27_rank_based['celebrity_name'] == 'Bobby Bones']
if not bobby_bones.empty:
    print(f'\nBobby Bones的淘汰概率（排名制）:')
    print(bobby_bones[['week', 'elimination_probability']].sort_values('week'))
    
    # 查找淘汰周数（使用原始排名判断）
    bobby_bones['is_eliminated'] = bobby_bones['rank'] == bobby_bones.groupby('week')['rank'].transform('max')
    print(f'\nBobby Bones的淘汰周数（原始排名）:')
    print(bobby_bones[['week', 'is_eliminated']][bobby_bones['is_eliminated']])

# 查看DeMarcus Ware的淘汰概率
demarcus_ware = season_27_rank_based[season_27_rank_based['celebrity_name'] == 'DeMarcus Ware']
if not demarcus_ware.empty:
    print(f'\nDeMarcus Ware的淘汰概率（排名制）:')
    print(demarcus_ware[['week', 'elimination_probability']].sort_values('week'))

# 查看所有选手的平均淘汰概率
print(f'\n赛季27各选手的平均淘汰概率（排名制）:')
print(season_27_rank_based.groupby('celebrity_name')['elimination_probability'].mean().sort_values(ascending=False))

# 查找排名前几的淘汰候选选手
top_elimination_candidates = season_27_rank_based.groupby('celebrity_name')['elimination_probability'].mean().sort_values(ascending=False).head(5)
print(f'\n排名前5的淘汰候选选手（排名制）:')
print(top_elimination_candidates)

# 比较原始百分比制和排名制的淘汰概率差异
print(f'\n原始百分比制和排名制的淘汰概率差异:')
original_probs = score_df[score_df['season'] == 27].groupby('celebrity_name')['comprehensive_score'].mean()
rank_based_probs = season_27_rank_based.groupby('celebrity_name')['elimination_probability'].mean()
prob_diff = rank_based_probs - original_probs
print(prob_diff.sort_values(ascending=False))

# 查找命运逆转的选手
print(f'\n命运逆转的选手（原始百分比制存活，但排名制可能被淘汰的选手）:')
reversed_fate = prob_diff.sort_values(ascending=True).head(3)
print(reversed_fate)

# 保存模拟结果
season_27_rank_based.to_csv('sensitivity_analysis_improved/season_27_rank_based_simulation.csv', index=False)