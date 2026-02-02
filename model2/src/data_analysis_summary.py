#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Dancing with the Stars 数据分析总结
# 基于修复后的影响分析脚本结果
# """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_analysis_results():
    """加载分析结果文件"""
    print("加载分析结果文件...")
    
    # 加载特征重要性数据
    judge_importance = pd.read_csv('../data/outputs/judge_feature_importance.csv')
    fan_importance = pd.read_csv('../data/outputs/fan_feature_importance.csv')
    
    # 加载争议案例
    controversial_cases = pd.read_csv('../data/outputs/controversial_cases.csv')
    
    # 加载交互效应数据
    try:
        judge_interactions = pd.read_csv('../data/outputs/judge_interactions.csv')
        fan_interactions = pd.read_csv('../data/outputs/fan_interactions.csv')
    except:
        judge_interactions = pd.DataFrame()
        fan_interactions = pd.DataFrame()
    
    return judge_importance, fan_importance, controversial_cases, judge_interactions, fan_interactions

def analyze_top_features(judge_importance, fan_importance):
    """分析最重要的特征"""
    print("\n" + "="*60)
    print("特征重要性分析")
    print("="*60)
    
    # 评委模型前5特征
    print("\n1. 评委得分模型 - 最重要的5个特征:")
    judge_top5 = judge_importance.head(5)
    for idx, row in judge_top5.iterrows():
        print(f"   {idx+1}. {row['feature']}: {row['importance_percent']:.2f}%")
    
    # 粉丝模型前5特征
    print("\n2. 粉丝投票模型 - 最重要的5个特征:")
    fan_top5 = fan_importance.head(5)
    for idx, row in fan_top5.iterrows():
        print(f"   {idx+1}. {row['feature']}: {row['importance_percent']:.2f}%")
    
    # 对比分析
    print("\n3. 评委与粉丝模型对比:")
    print("   - 共同重要特征: 行业类别相关特征")
    print("   - 评委更关注: 职业舞者经验与行业的交互")
    print("   - 粉丝更关注: 年龄与行业的交互")
    
    return judge_top5, fan_top5

def analyze_differences(judge_importance, fan_importance):
    """分析评委与粉丝模型差异"""
    print("\n" + "="*60)
    print("评委与粉丝模型差异分析")
    print("="*60)
    
    # 创建特征名称映射以进行比较
    judge_features = set(judge_importance['feature'])
    fan_features = set(fan_importance['feature'])
    common_features = judge_features.intersection(fan_features)
    
    # 对于共有特征，比较重要性排名
    comparison = []
    for feature in common_features:
        judge_rank = judge_importance[judge_importance['feature'] == feature].index[0] + 1
        fan_rank = fan_importance[fan_importance['feature'] == feature].index[0] + 1
        rank_diff = judge_rank - fan_rank
        
        judge_imp = judge_importance.loc[judge_importance['feature'] == feature, 'importance_percent'].values[0]
        fan_imp = fan_importance.loc[fan_importance['feature'] == feature, 'importance_percent'].values[0]
        imp_diff = judge_imp - fan_imp
        
        comparison.append({
            'feature': feature,
            'judge_rank': judge_rank,
            'fan_rank': fan_rank,
            'rank_diff': rank_diff,
            'judge_imp%': judge_imp,
            'fan_imp%': fan_imp,
            'imp_diff%': imp_diff
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # 找出差异最大的特征
    print("\n差异最大的特征（评委vs粉丝重要性差异）:")
    comparison_df['abs_imp_diff'] = comparison_df['imp_diff%'].abs()
    top_diffs = comparison_df.sort_values('abs_imp_diff', ascending=False).head(5)
    
    for idx, row in top_diffs.iterrows():
        if row['imp_diff%'] > 0:
            print(f"   - {row['feature']}: 评委({row['judge_imp%']:.2f}%) > 粉丝({row['fan_imp%']:.2f}%)")
        else:
            print(f"   - {row['feature']}: 粉丝({row['fan_imp%']:.2f}%) > 评委({row['judge_imp%']:.2f}%)")
    
    return comparison_df

def analyze_controversial_cases(controversial_cases):
    """分析争议案例"""
    print("\n" + "="*60)
    print("争议案例分析")
    print("="*60)
    
    if len(controversial_cases) == 0:
        print("没有找到争议案例")
        return
    
    print(f"共发现 {len(controversial_cases)} 个争议案例")
    
    # 计算评委得分与粉丝投票的差异
    controversial_cases = controversial_cases.copy()
    controversial_cases['score_diff'] = controversial_cases['avg_judge_score'] - controversial_cases['predicted_fan_votes'] * 10
    controversial_cases['abs_diff'] = controversial_cases['score_diff'].abs()
    
    # 找出最大的分歧案例
    top_divergence = controversial_cases.sort_values('abs_diff', ascending=False).head(3)
    
    print("\n评委与粉丝分歧最大的案例:")
    for idx, row in top_divergence.iterrows():
        print(f"\n   {row['celebrity_name']} (第{row['season']}季):")
        print(f"    - 评委得分: {row['avg_judge_score']:.2f}")
        print(f"    - 粉丝投票: {row['predicted_fan_votes']:.2f}")
        print(f"    - 差异程度: {row['abs_diff']:.2f}")
        if row['score_diff'] > 0:
            print("    - 类型: 评委更喜欢，但粉丝支持度低")
        else:
            print("    - 类型: 粉丝更喜欢，但评委评分低")
    
    # 统计行业分布
    print("\n争议案例的特征分析:")
    print("   - 这些案例揭示了评委专业标准与粉丝偏好的差异")
    print("   - 高评委分低粉丝票: 专业表现好但缺乏观众缘")
    print("   - 低评委分高粉丝票: 观众喜爱但技术表现一般")
    
    return controversial_cases

def generate_key_insights(judge_top5, fan_top5, comparison_df, controversial_cases):
    """生成关键见解"""
    print("\n" + "="*60)
    print("关键见解与建议")
    print("="*60)
    
    # 基于特征重要性的见解
    print("\n1. 模型表现见解:")
    print("   - 评委得分模型R²=0.815，解释能力较强")
    print("   - 粉丝投票模型R²=0.052，预测难度较大，粉丝投票受更多不可控因素影响")
    
    # 特征重要性见解
    print("\n2. 关键影响因素:")
    print("   - 对评委最重要的: 职业舞者经验与特定行业的交互")
    print("   - 对粉丝最重要的: 参赛者年龄与行业的交互")
    print("   - 行业类别是共同重要因素，但交互方式不同")
    
    # 争议案例见解
    if len(controversial_cases) > 0:
        print("\n3. 评委与粉丝分歧:")
        print("   - 存在系统性的评委专业标准与粉丝偏好的差异")
        print("   - 某些参赛者获得评委高分但粉丝支持度低（技术好但缺乏观众缘）")
        print("   - 某些参赛者粉丝支持度高但评委评分低（观众喜爱但技术一般）")
    
    # 建议
    print("\n4. 节目制作建议:")
    print("   - 平衡专业评审与观众投票的权重")
    print("   - 考虑不同行业背景参赛者的公平性")
    print("   - 优化投票机制以减少极端分歧案例")
    print("   - 加强评委与观众之间的沟通，解释评分标准")
    
    print("\n" + "="*60)
    print("分析完成")
    print("="*60)

def create_visualization_summary(judge_importance, fan_importance):
    """创建可视化总结"""
    # 合并两个模型的特征重要性
    judge_top10 = judge_importance.head(10)[['feature', 'importance_percent']]
    fan_top10 = fan_importance.head(10)[['feature', 'importance_percent']]
    
    judge_top10['model'] = '评委'
    fan_top10['model'] = '粉丝'
    
    combined = pd.concat([judge_top10, fan_top10])
    
    # 重塑数据用于绘图
    pivot_data = combined.pivot(index='feature', columns='model', values='importance_percent')
    pivot_data = pivot_data.fillna(0)
    
    # 保存为CSV供进一步分析
    pivot_data.to_csv('feature_importance_comparison_pivot.csv')
    
    print(f"\n已保存特征重要性对比数据到: feature_importance_comparison_pivot.csv")
    print("可用以下代码绘制对比图:")
    print("""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('feature_importance_comparison_pivot.csv', index_col=0)
fig, ax = plt.subplots(figsize=(12, 8))
data.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
plt.title('评委与粉丝模型特征重要性对比 (Top 10)')
plt.ylabel('重要性百分比 (%)')
plt.xlabel('特征')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance_comparison_summary.png', dpi=300)
plt.show()
    """)

def main():
    """主函数"""
    print("="*60)
    print("Dancing with the Stars 数据分析总结报告")
    print("="*60)
    
    # 加载数据
    judge_importance, fan_importance, controversial_cases, judge_interactions, fan_interactions = load_analysis_results()
    
    # 分析顶级特征
    judge_top5, fan_top5 = analyze_top_features(judge_importance, fan_importance)
    
    # 分析差异
    comparison_df = analyze_differences(judge_importance, fan_importance)
    
    # 分析争议案例
    controversial_cases_analyzed = analyze_controversial_cases(controversial_cases)
    
    # 生成关键见解
    generate_key_insights(judge_top5, fan_top5, comparison_df, controversial_cases_analyzed)
    
    # 创建可视化总结
    create_visualization_summary(judge_importance, fan_importance)
    
    print("\n分析报告生成完成！")

if __name__ == "__main__":
    main()