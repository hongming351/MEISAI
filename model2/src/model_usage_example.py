"""
模型使用示例脚本
用于加载已保存的模型并进行预测
"""

import os
import joblib
import pandas as pd
import numpy as np

def load_models_and_predict():
    """
    加载已保存的模型并进行预测示例
    """
    print("=" * 50)
    print("模型加载与预测示例")
    print("=" * 50)
    
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))  # 到D:\MEISAI目录
    models_dir = os.path.join(base_dir, 'task3', 'models')
    
    # 检查模型文件是否存在
    if not os.path.exists(models_dir):
        print(f"错误: 模型目录不存在: {models_dir}")
        print("请先运行 influence_analysis.py 来训练和保存模型")
        return
    
    print(f"模型目录: {models_dir}")
    
    try:
        # 1. 加载模型和预处理对象
        print("\n1. 加载模型和预处理对象...")
        
        judge_model_path = os.path.join(models_dir, 'judge_score_model.joblib')
        fan_model_path = os.path.join(models_dir, 'fan_vote_model.joblib')
        preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
        feature_names_path = os.path.join(models_dir, 'feature_names.joblib')
        model_metrics_path = os.path.join(models_dir, 'model_metrics.joblib')
        
        judge_model = joblib.load(judge_model_path)
        fan_model = joblib.load(fan_model_path)
        preprocessor = joblib.load(preprocessor_path)
        feature_names = joblib.load(feature_names_path)
        model_metrics = joblib.load(model_metrics_path)
        
        print("✓ 模型加载成功")
        print(f"   - 评委得分模型: {judge_model_path}")
        print(f"   - 粉丝投票模型: {fan_model_path}")
        print(f"   - 特征数量: {len(feature_names)}")
        
        # 2. 显示模型性能
        print("\n2. 模型性能指标:")
        print("评委得分模型:")
        print(f"   - MSE: {model_metrics['judge_model']['mse']:.3f}")
        print(f"   - RMSE: {model_metrics['judge_model']['rmse']:.3f}")
        print(f"   - R²: {model_metrics['judge_model']['r2']:.3f}")
        print(f"   - 交叉验证平均R²: {model_metrics['judge_model']['avg_r2']:.3f}")
        
        print("\n粉丝投票模型:")
        print(f"   - MSE: {model_metrics['fan_model']['mse']:.3f}")
        print(f"   - RMSE: {model_metrics['fan_model']['rmse']:.3f}")
        print(f"   - R²: {model_metrics['fan_model']['r2']:.3f}")
        print(f"   - 交叉验证平均R²: {model_metrics['fan_model']['avg_r2']:.3f}")
        
        # 3. 创建示例数据进行预测
        print("\n3. 创建示例数据进行预测...")
        
        # 示例数据：一个30岁的演员，有5次专业经验，美国人
        example_data = {
            'celebrity_age_during_season': [30],
            'pro_experience': [5],
            'industry_simplified': ['Actor'],
            'is_american': [True]
        }
        
        # 创建DataFrame
        example_df = pd.DataFrame(example_data)
        print("示例数据:")
        print(example_df)
        
        # 4. 预处理数据
        print("\n4. 预处理数据...")
        # 注意：实际使用时需要确保特征顺序和类型与训练时一致
        # 这里简化处理，实际应该使用保存的preprocessor
        
        # 5. 进行预测
        print("\n5. 进行预测...")
        
        # 由于预处理比较复杂，这里简化演示
        # 实际使用时应该使用保存的preprocessor.transform()
        print("注意: 实际预测需要完整的特征工程流程")
        print("这里仅演示模型加载功能")
        
        # 6. 保存的模型可以用于：
        print("\n6. 保存的模型可以用于:")
        print("   - 对新参赛者进行评分预测")
        print("   - 分析特征重要性")
        print("   - 进行SHAP值分析")
        print("   - 评估不同特征组合的影响")
        
        # 7. 创建模型使用说明
        print("\n7. 模型使用说明:")
        print("""
使用步骤:
1. 加载模型和预处理对象:
   judge_model = joblib.load('task3/models/judge_score_model.joblib')
   preprocessor = joblib.load('task3/models/preprocessor.joblib')

2. 准备新数据（与训练数据相同的格式）

3. 使用preprocessor.transform()进行特征工程

4. 使用模型进行预测:
   judge_score = judge_model.predict(processed_data)
   fan_votes = fan_model.predict(processed_data)

5. 分析结果
        """)
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保已运行 influence_analysis.py 并成功保存了模型")

def check_model_files():
    """
    检查模型文件是否存在
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    models_dir = os.path.join(base_dir, 'task3', 'models')
    
    if not os.path.exists(models_dir):
        print(f"模型目录不存在: {models_dir}")
        return False
    
    required_files = [
        'judge_score_model.joblib',
        'fan_vote_model.joblib',
        'preprocessor.joblib',
        'feature_names.joblib',
        'model_metrics.joblib'
    ]
    
    print("检查模型文件:")
    all_exist = True
    for file in required_files:
        file_path = os.path.join(models_dir, file)
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    # 首先检查模型文件
    print("模型文件检查:")
    if check_model_files():
        print("\n所有模型文件都存在，可以加载使用")
        load_models_and_predict()
    else:
        print("\n部分模型文件缺失，请先运行 influence_analysis.py")
        print("运行命令: python src/influence_analysis.py")