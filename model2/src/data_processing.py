import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_and_preprocess_data():
    """
    加载并预处理数据（使用 task1 目录下的预处理文件）
    """
    print("正在加载 task1 目录下的预处理数据...")
    
    # 获取当前文件所在目录的绝对路径，确保路径正确
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))  # 到D:\MEISAI目录
    
    # 加载三个预处理数据文件
    rank_regular_path = os.path.join(base_dir, 'task1', 'dwts_rank_regular_processed.csv')
    percentage_regular_path = os.path.join(base_dir, 'task1', 'dwts_percentage_regular_processed.csv')
    rank_bottom_two_path = os.path.join(base_dir, 'task1', 'dwts_rank_bottom_two_processed.csv')
    
    # 验证文件是否存在
    for file_path in [rank_regular_path, percentage_regular_path, rank_bottom_two_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    print(f"正在读取: {rank_regular_path}")
    print(f"正在读取: {percentage_regular_path}")
    print(f"正在读取: {rank_bottom_two_path}")
    
    # 使用 cp1252 编码读取以正确处理重音字符（É, û 等）
    df_rank_regular = pd.read_csv(rank_regular_path, encoding='cp1252')
    df_percentage_regular = pd.read_csv(percentage_regular_path, encoding='cp1252')
    df_rank_bottom_two = pd.read_csv(rank_bottom_two_path, encoding='cp1252')
    
    print(f"Rank-based Regular (S1-2): {len(df_rank_regular)} 条记录")
    print(f"Percentage-based Regular (S3-27): {len(df_percentage_regular)} 条记录")
    print(f"Rank-based Bottom Two (S28-34): {len(df_rank_bottom_two)} 条记录")
    
    # 为每个数据集添加阶段标记
    df_rank_regular['voting_phase'] = 'rank_regular'
    df_percentage_regular['voting_phase'] = 'percentage_regular'
    df_rank_bottom_two['voting_phase'] = 'rank_bottom_two'
    
    # 合并数据
    data = pd.concat([df_rank_regular, df_percentage_regular, df_rank_bottom_two], ignore_index=True)
    
    print(f"合并后总数据: {len(data)} 条记录，{data['season'].nunique()} 个赛季")
    
    # 基本数据清理
    data = data.dropna(subset=['celebrity_age_during_season', 'celebrity_industry',
                              'celebrity_homecountry/region', 'ballroom_partner'])
    
    # 计算平均评委得分
    judge_columns = [col for col in data.columns if 'week' in col.lower() and 'judge' in col.lower() and 'score' in col.lower()]
    data['avg_judge_score'] = data[judge_columns].replace(0, np.nan).mean(axis=1)
    data = data.dropna(subset=['avg_judge_score'])
    
    # 加载粉丝投票预测数据
    fan_vote_path = os.path.join(base_dir, 'task1', 'fan_vote_predictions_enhanced.csv')
    if not os.path.exists(fan_vote_path):
        raise FileNotFoundError(f"粉丝投票预测数据文件不存在: {fan_vote_path}")
    
    print(f"正在读取: {fan_vote_path}")
    fan_vote_data = pd.read_csv(fan_vote_path)
    data = pd.merge(data, fan_vote_data[['contestant', 'season', 'fan_vote_raw']],
                   left_on=['celebrity_name', 'season'],
                   right_on=['contestant', 'season'],
                   how='left')
    # 重命名列
    data.rename(columns={'fan_vote_raw': 'predicted_fan_votes'}, inplace=True)
    # 删除重复列
    data.drop(['contestant'], axis=1, inplace=True)
    
    # 计算职业舞者经验（历史决赛/获胜次数）
    pro_dancer_experience = data.groupby('ballroom_partner')['placement'].apply(
        lambda x: sum((x <= 3) & (x > 0))
    ).reset_index()
    pro_dancer_experience.columns = ['ballroom_partner', 'pro_experience']
    data = pd.merge(data, pro_dancer_experience, on='ballroom_partner', how='left')
    data['pro_experience'] = data['pro_experience'].fillna(0)
    
    # 地域特征处理
    data['is_american'] = data['celebrity_homecountry/region'] == 'United States'
    
    # 行业类别处理
    industry_mapping = {
        'Athlete': 'Athlete',
        'Actor/Actress': 'Actor',
        'Singer': 'Singer',
        'Television Personality': 'TV Personality',
        'Model': 'Model',
        'Comedian': 'Comedian',
        'Dancer': 'Dancer',
        'Musician': 'Musician',
        'Writer': 'Writer',
        'Politician': 'Politician'
    }
    data['industry_simplified'] = data['celebrity_industry'].map(
        lambda x: next((v for k, v in industry_mapping.items() if k in str(x)), 'Other')
    )
    
    return data


def prepare_features(data):
    """
    准备特征和目标变量
    """
    # 复制数据以避免修改原始数据
    data_clean = data.copy()
    
    # 删除目标变量中的缺失值
    mask = data_clean['predicted_fan_votes'].notna()
    data_clean = data_clean[mask]
    
    # 目标变量
    y_judge = data_clean['avg_judge_score']
    y_fan = data_clean['predicted_fan_votes']

    # 特征变量
    numeric_features = ['celebrity_age_during_season', 'pro_experience']
    categorical_features = ['industry_simplified', 'is_american']

    # 特征预处理
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 转换特征
    X = preprocessor.fit_transform(data_clean)

    # 获取特征名称
    feature_names = numeric_features
    ohe_categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_
    for category, values in zip(categorical_features, ohe_categories):
        for value in values[1:]:  # 排除第一个类别（参考类别）
            feature_names.append(f"{category}_{value}")

    # 清理特征名称中的重音字符和非ASCII字符
    import unicodedata
    cleaned_feature_names = []
    for name in feature_names:
        # 将Unicode字符规范化并移除重音
        normalized = unicodedata.normalize('NFKD', str(name))
        ascii_name = normalized.encode('ascii', 'ignore').decode('ascii')
        # 如果完全为空，保留原始名称（但移除非ASCII字符）
        if not ascii_name.strip():
            ascii_name = str(name).encode('ascii', 'ignore').decode('ascii')
        cleaned_feature_names.append(ascii_name)
    
    # 确保没有重复的名称（万一清理后出现重复）
    seen = {}
    final_feature_names = []
    for name in cleaned_feature_names:
        original_name = name
        counter = 1
        while name in seen:
            name = f"{original_name}_{counter}"
            counter += 1
        seen[name] = True
        final_feature_names.append(name)

    return X, y_judge, y_fan, final_feature_names, preprocessor


def create_interaction_features(X, feature_names):
    """
    创建交互特征
    """
    df = pd.DataFrame(X, columns=feature_names)
    
    # 年龄与行业交互
    age = df['celebrity_age_during_season']
    industry_cols = [col for col in feature_names if 'industry_simplified' in col]
    for col in industry_cols:
        df[f'age_{col}'] = age * df[col]
    
    # 经验与行业交互
    experience = df['pro_experience']
    for col in industry_cols:
        df[f'experience_{col}'] = experience * df[col]
    
    return df.values, list(df.columns)