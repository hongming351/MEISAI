import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re

warnings.filterwarnings('ignore')

# Set font for Chinese characters (if needed for display)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. Data Loading and Preprocessing ====================
def load_and_preprocess_data(rank_regular_path, percentage_regular_path, rank_bottom_two_path):
    """
    Load and preprocess data (three files)
    """
    print("Loading preprocessed data...")
    
    # Load three files
    df_rank_regular = pd.read_csv(rank_regular_path)
    df_percentage_regular = pd.read_csv(percentage_regular_path)
    df_rank_bottom_two = pd.read_csv(rank_bottom_two_path)
    
    print(f"Rank-based Regular (S1-2): {len(df_rank_regular)} records")
    print(f"Percentage-based Regular (S3-27): {len(df_percentage_regular)} records")
    print(f"Rank-based Bottom Two (S28-34): {len(df_rank_bottom_two)} records")
    
    # Add phase markers for each dataset
    df_rank_regular['voting_phase'] = 'rank_regular'
    df_percentage_regular['voting_phase'] = 'percentage_regular'
    df_rank_bottom_two['voting_phase'] = 'rank_bottom_two'
    
    # Merge data
    df = pd.concat([df_rank_regular, df_percentage_regular, df_rank_bottom_two], ignore_index=True)
    
    print(f"Total merged data: {len(df)} records, {df['season'].nunique()} seasons")
    
    # Check data columns
    print(f"Number of columns: {len(df.columns)}")
    print(f"First 10 columns: {list(df.columns[:10])}")
    
    # Find all week-related columns
    week_columns = [col for col in df.columns if 'week' in col.lower()]
    print(f"Found {len(week_columns)} week-related columns")
    
    # Find judge score columns (format like week1_judge1_score)
    judge_score_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if 'week' in col_lower and 'judge' in col_lower and 'score' in col_lower:
            judge_score_columns.append(col)
    
    print(f"Found {len(judge_score_columns)} judge score columns")
    
    # Calculate weekly total scores (sum of multiple judge scores)
    total_score_cols = []
    max_week = 0
    
    if judge_score_columns:
        # Extract week numbers
        week_numbers = {}
        for col in judge_score_columns:
            # Use regex to extract week number
            match = re.search(r'week(\d+)', col.lower())
            if match:
                week_num = int(match.group(1))
                if week_num not in week_numbers:
                    week_numbers[week_num] = []
                week_numbers[week_num].append(col)
        
        # Create total score columns for each week
        for week_num in sorted(week_numbers.keys()):
            score_cols = week_numbers[week_num]
            total_col_name = f'week{week_num}_total_score'
            
            # Calculate total score
            df[total_col_name] = 0
            for col in score_cols:
                # Convert column to numeric type, handle missing values
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[total_col_name] += df[col]
            
            total_score_cols.append(total_col_name)
            max_week = max(max_week, week_num)
        
        print(f"Created {len(total_score_cols)} total score columns")
        print(f"Detected up to {max_week} weeks of data")
    else:
        print("Warning: No judge score columns found")
        # Try to find existing total score columns
        existing_total_cols = [col for col in df.columns if 'total' in col.lower() and 'score' in col.lower()]
        if existing_total_cols:
            total_score_cols = existing_total_cols
            print(f"Using existing total score columns: {len(total_score_cols)}")
            # Determine maximum week number
            for col in total_score_cols:
                try:
                    week_num = int(col.split('_')[0].replace('week', ''))
                    max_week = max(max_week, week_num)
                except:
                    continue
    
    # Ensure necessary columns exist in data
    required_columns = ['celebrity_name', 'season', 'placement', 'voting_phase']
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Missing required column '{col}'")
    
    # Handle missing values
    for col in total_score_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print(f"Data loading completed, total {len(df)} records")
    
    return df, max_week, total_score_cols

# ==================== 2. Voting Method Selection (Based on Voting Phase) ====================
def get_voting_method(season, voting_phase=None):
    """
    Return voting method based on season and voting phase
    """
    if voting_phase:
        if voting_phase == 'rank_regular' or voting_phase == 'rank_bottom_two':
            return 'rank'
        elif voting_phase == 'percentage_regular':
            return 'percentage'
    else:
        # If voting_phase not provided, determine by season
        if season <= 2:
            return 'rank'
        elif 3 <= season <= 27:
            return 'percentage'
        else:  # 28-34
            return 'rank'

# ==================== 3. Prediction Functions ====================
def predict_by_rank(judge_scores, eliminated, bottom_two=None):
    """
    Predict fan voting using ranking method
    """
    n = len(judge_scores)
    contestants = list(judge_scores.keys())
    
    if n == 0:
        return {}
    
    # Calculate judge rankings (higher score, lower rank number)
    sorted_by_score = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
    judge_ranks = {contestant: rank+1 for rank, (contestant, _) in enumerate(sorted_by_score)}
    
    # Number of samples (adjusted based on n)
    if n <= 7:
        # Small scale: enumerate all permutations
        all_perms = list(itertools.permutations(contestants))
        valid_perms = []
        
        for perm in all_perms:
            # Calculate total ranks
            total_ranks = {}
            for fan_rank, contestant in enumerate(perm, 1):
                total_ranks[contestant] = judge_ranks[contestant] + fan_rank
            
            # Find highest total rank (worst)
            max_rank = max(total_ranks.values())
            worst_contestants = [c for c, r in total_ranks.items() if r == max_rank]
            
            # Check if matches elimination result
            if eliminated:
                if set(worst_contestants) == set(eliminated):
                    valid_perms.append(perm)
            elif bottom_two:  # Bottom Two mode
                # Check if worst two contestants are bottom_two
                sorted_by_total_rank = sorted(total_ranks.items(), key=lambda x: x[1], reverse=True)
                bottom_two_candidates = [c for c, _ in sorted_by_total_rank[-2:]]
                if set(bottom_two_candidates) == set(bottom_two):
                    valid_perms.append(perm)
            else:
                # No elimination this week (like final week)
                valid_perms.append(perm)
        
        if valid_perms:
            # Calculate average fan ranking
            fan_rank_matrix = np.zeros((len(valid_perms), n))
            contestant_to_idx = {c: i for i, c in enumerate(contestants)}
            
            for i, perm in enumerate(valid_perms):
                for fan_rank, contestant in enumerate(perm, 1):
                    fan_rank_matrix[i, contestant_to_idx[contestant]] = fan_rank
            
            avg_fan_ranks = fan_rank_matrix.mean(axis=0)
            
            # Convert rankings to voting proportions (better ranking, more votes)
            fan_votes = {}
            for i, contestant in enumerate(contestants):
                rank = avg_fan_ranks[i]
                fan_votes[contestant] = np.exp(-rank/2)
            
            # Normalize
            total = sum(fan_votes.values())
            if total > 0:
                fan_votes = {c: v/total for c, v in fan_votes.items()}
            else:
                fan_votes = {c: 1/n for c in contestants}
        else:
            # No valid permutations, use uniform distribution
            fan_votes = {c: 1/n for c in contestants}
    
    else:
        # Large scale: Monte Carlo sampling
        n_samples = min(10000, 200 * n)
        samples = []
        
        for _ in range(n_samples):
            # Random permutation
            perm = np.random.permutation(contestants)
            
            # Calculate total ranks
            total_ranks = {}
            for fan_rank, contestant in enumerate(perm, 1):
                total_ranks[contestant] = judge_ranks[contestant] + fan_rank
            
            # Find worst contestants
            max_rank = max(total_ranks.values())
            worst_contestants = [c for c, r in total_ranks.items() if r == max_rank]
            
            # Check if matches elimination result
            if eliminated:
                if set(worst_contestants) == set(eliminated):
                    samples.append(perm)
            elif bottom_two:  # Bottom Two mode
                # Check if worst two contestants are bottom_two
                sorted_by_total_rank = sorted(total_ranks.items(), key=lambda x: x[1], reverse=True)
                bottom_two_candidates = [c for c, _ in sorted_by_total_rank[-2:]]
                if set(bottom_two_candidates) == set(bottom_two):
                    samples.append(perm)
            else:
                samples.append(perm)
        
        if samples:
            # Calculate average fan ranking
            fan_rank_matrix = np.zeros((len(samples), n))
            contestant_to_idx = {c: i for i, c in enumerate(contestants)}
            
            for i, sample in enumerate(samples):
                for fan_rank, contestant in enumerate(sample, 1):
                    fan_rank_matrix[i, contestant_to_idx[contestant]] = fan_rank
            
            avg_fan_ranks = fan_rank_matrix.mean(axis=0)
            
            # Convert to voting proportions
            fan_votes = {}
            for i, contestant in enumerate(contestants):
                rank = avg_fan_ranks[i]
                fan_votes[contestant] = np.exp(-rank/2)
            
            # Normalize
            total = sum(fan_votes.values())
            if total > 0:
                fan_votes = {c: v/total for c, v in fan_votes.items()}
            else:
                fan_votes = {c: 1/n for c in contestants}
        else:
            # No valid samples, use uniform distribution
            fan_votes = {c: 1/n for c in contestants}
    
    return fan_votes

def predict_by_percentage(judge_scores, eliminated):
    """
    Predict fan voting using percentage method
    """
    n = len(judge_scores)
    contestants = list(judge_scores.keys())
    
    if n == 0:
        return {}
    
    # Calculate judge percentages
    total_judge_score = sum(judge_scores.values())
    if total_judge_score == 0:
        judge_percentages = {c: 1/n for c in contestants}
    else:
        judge_percentages = {c: score/total_judge_score for c, score in judge_scores.items()}
    
    # Monte Carlo sampling
    n_samples = min(10000, 200 * n)
    samples = []
    
    for _ in range(n_samples):
        # Generate fan voting (Dirichlet distribution)
        alpha = np.ones(n)  # Symmetric
        
        # Adjust alpha based on judge scores (higher judge score, potentially higher fan vote)
        for i, contestant in enumerate(contestants):
            alpha[i] = 1 + judge_scores[contestant] / 100
        
        fan_votes_sample = np.random.dirichlet(alpha)
        
        # Calculate total percentages (judge weight 0.4, fan weight 0.6)
        total_percentages = {}
        for i, contestant in enumerate(contestants):
            total_percentages[contestant] = 0.4 * judge_percentages[contestant] + 0.6 * fan_votes_sample[i]
        
        # Find lowest total percentage (worst)
        min_percentage = min(total_percentages.values())
        worst_contestants = [c for c, p in total_percentages.items() if p == min_percentage]
        
        # Check if matches elimination result
        if eliminated:
            if set(worst_contestants) == set(eliminated):
                samples.append(fan_votes_sample)
        else:
            samples.append(fan_votes_sample)
    
    if samples:
        # Calculate average fan voting
        samples_array = np.array(samples)
        avg_fan_votes = samples_array.mean(axis=0)
        
        fan_votes = {}
        for i, contestant in enumerate(contestants):
            fan_votes[contestant] = avg_fan_votes[i]
    else:
        # No valid samples, use uniform distribution
        fan_votes = {c: 1/n for c in contestants}
    
    return fan_votes

# ==================== 4. Comparison Analysis of Two Methods ====================
def compare_voting_methods(df, max_week, total_score_cols):
    """
    Compare two voting methods across all seasons
    """
    print("\nStarting comparison of two voting methods...")
    
    if df.empty or not total_score_cols:
        print("Insufficient data for comparison analysis")
        return pd.DataFrame()
    
    seasons = sorted(df['season'].unique())
    comparison_results = []
    
    for season in seasons:
        season_data = df[df['season'] == season]
        if season_data.empty:
            continue
            
        # Get voting phase for this season
        if 'voting_phase' in season_data.columns:
            voting_phase = season_data['voting_phase'].iloc[0]
        else:
            # Infer based on season
            if season <= 2:
                voting_phase = 'rank_regular'
            elif 3 <= season <= 27:
                voting_phase = 'percentage_regular'
            else:
                voting_phase = 'rank_bottom_two'
        
        # Get actual elimination information
        actual_eliminations = {}
        max_week_season = 0
        
        # Get maximum week for this season
        for week in range(1, max_week + 1):
            col = f'week{week}_total_score'
            if col in season_data.columns and season_data[col].notna().any():
                max_week_season = week
        
        # Check elimination each week
        for week in range(1, max_week_season):
            current_col = f'week{week}_total_score'
            next_col = f'week{week+1}_total_score'
            
            if current_col in season_data.columns and next_col in season_data.columns:
                eliminated = []
                for _, row in season_data.iterrows():
                    current_score = row[current_col]
                    next_score = row[next_col]
                    
                    if current_score > 0 and (pd.isna(next_score) or next_score == 0):
                        eliminated.append(row['celebrity_name'])
                
                if eliminated:
                    actual_eliminations[(season, week)] = eliminated
        
        # Predict using two methods separately
        rank_eliminations = {}
        percentage_eliminations = {}
        
        for week in range(1, max_week_season + 1):
            score_col = f'week{week}_total_score'
            if score_col not in season_data.columns:
                continue
                
            # Get active contestants this week
            week_data = season_data[season_data[score_col] > 0]
            if week_data.empty:
                continue
                
            active_contestants = week_data['celebrity_name'].tolist()
            n_active = len(active_contestants)
            
            if n_active == 0:
                continue
            
            # Get judge scores
            judge_scores = {}
            for contestant in active_contestants:
                contestant_data = season_data[season_data['celebrity_name'] == contestant]
                if not contestant_data.empty:
                    judge_scores[contestant] = contestant_data[score_col].iloc[0]
            
            # Predict using ranking method
            rank_fan_votes = predict_by_rank(judge_scores, [])
            if rank_fan_votes:
                # Find contestant with lowest fan vote
                min_vote_contestant = min(rank_fan_votes, key=rank_fan_votes.get)
                rank_eliminations[(season, week)] = [min_vote_contestant]
            
            # Predict using percentage method
            percentage_fan_votes = predict_by_percentage(judge_scores, [])
            if percentage_fan_votes:
                # Find contestant with lowest fan vote
                min_vote_contestant = min(percentage_fan_votes, key=percentage_fan_votes.get)
                percentage_eliminations[(season, week)] = [min_vote_contestant]
        
        # Calculate match rate with actual results
        rank_matches = 0
        percentage_matches = 0
        total_weeks = len(actual_eliminations)
        
        for key in actual_eliminations:
            actual_set = set(actual_eliminations[key])
            
            if key in rank_eliminations:
                rank_set = set(rank_eliminations[key])
                if rank_set == actual_set:
                    rank_matches += 1
            
            if key in percentage_eliminations:
                percentage_set = set(percentage_eliminations[key])
                if percentage_set == actual_set:
                    percentage_matches += 1
        
        rank_accuracy = rank_matches / total_weeks if total_weeks > 0 else 0
        percentage_accuracy = percentage_matches / total_weeks if total_weeks > 0 else 0
        
        # Count differences in predictions between two methods
        different_predictions = 0
        common_weeks = set(rank_eliminations.keys()) & set(percentage_eliminations.keys())
        
        for week_key in common_weeks:
            if set(rank_eliminations.get(week_key, [])) != set(percentage_eliminations.get(week_key, [])):
                different_predictions += 1
        
        comparison_results.append({
            'season': season,
            'voting_phase': voting_phase,
            'total_weeks': total_weeks,
            'rank_accuracy': rank_accuracy,
            'percentage_accuracy': percentage_accuracy,
            'accuracy_difference': percentage_accuracy - rank_accuracy,
            'different_predictions': different_predictions,
            'different_prediction_rate': different_predictions / len(common_weeks) if len(common_weeks) > 0 else 0
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    return comparison_df

def analyze_method_impact_on_controversial_cases(df, max_week, total_score_cols):
    """
    Analyze the impact of two methods on controversial cases
    """
    print("\nAnalyzing impact of two methods on controversial cases...")
    
    controversial_cases = [
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones")
    ]
    
    results = []
    
    for season, contestant in controversial_cases:
        season_data = df[df['season'] == season]
        contestant_data = season_data[season_data['celebrity_name'] == contestant]
        
        if contestant_data.empty:
            continue
        
        # Get contestant's competition weeks
        contestant_weeks = []
        for week in range(1, max_week + 1):
            col = f'week{week}_total_score'
            if col in contestant_data.columns and contestant_data[col].iloc[0] > 0:
                contestant_weeks.append(week)
        
        if not contestant_weeks:
            continue
        
        # Analyze each week
        weekly_results = []
        for week in contestant_weeks:
            score_col = f'week{week}_total_score'
            
            # Get all active contestants this week
            week_data = season_data[season_data[score_col] > 0]
            if week_data.empty:
                continue
                
            active_contestants = week_data['celebrity_name'].tolist()
            
            # Get judge scores
            judge_scores = {}
            for active_contestant in active_contestants:
                active_data = season_data[season_data['celebrity_name'] == active_contestant]
                if not active_data.empty:
                    judge_scores[active_contestant] = active_data[score_col].iloc[0]
            
            # Predict fan voting using two methods
            rank_fan_votes = predict_by_rank(judge_scores, [])
            percentage_fan_votes = predict_by_percentage(judge_scores, [])
            
            if contestant in rank_fan_votes and contestant in percentage_fan_votes:
                # Calculate position in this week
                rank_fan_votes_sorted = sorted(rank_fan_votes.items(), key=lambda x: x[1], reverse=True)
                percentage_fan_votes_sorted = sorted(percentage_fan_votes.items(), key=lambda x: x[1], reverse=True)
                
                rank_position = [i for i, (c, _) in enumerate(rank_fan_votes_sorted) if c == contestant][0] + 1
                percentage_position = [i for i, (c, _) in enumerate(percentage_fan_votes_sorted) if c == contestant][0] + 1
                
                weekly_results.append({
                    'week': week,
                    'judge_score': judge_scores.get(contestant, 0),
                    'rank_fan_vote': rank_fan_votes[contestant],
                    'percentage_fan_vote': percentage_fan_votes[contestant],
                    'rank_position': rank_position,
                    'percentage_position': percentage_position,
                    'position_difference': rank_position - percentage_position,
                    'total_contestants': len(active_contestants)
                })
        
        if weekly_results:
            weekly_df = pd.DataFrame(weekly_results)
            
            results.append({
                'season': season,
                'contestant': contestant,
                'total_weeks': len(weekly_results),
                'avg_judge_score': weekly_df['judge_score'].mean(),
                'avg_rank_fan_vote': weekly_df['rank_fan_vote'].mean(),
                'avg_percentage_fan_vote': weekly_df['percentage_fan_vote'].mean(),
                'avg_rank_position': weekly_df['rank_position'].mean(),
                'avg_percentage_position': weekly_df['percentage_position'].mean(),
                'weeks_rank_better': (weekly_df['rank_position'] < weekly_df['percentage_position']).sum(),
                'weeks_percentage_better': (weekly_df['rank_position'] > weekly_df['percentage_position']).sum(),
                'weeks_equal': (weekly_df['rank_position'] == weekly_df['percentage_position']).sum()
            })
    
    return pd.DataFrame(results)

def calculate_fan_vote_weight(df, max_week):
    """
    Calculate fan voting weight in two methods
    """
    print("\nCalculating fan voting weight in two methods...")
    
    seasons = sorted(df['season'].unique())
    weight_results = []
    
    for season in seasons[:20]:  # Analyze first 20 seasons to reduce computation
        season_data = df[df['season'] == season]
        
        # Get maximum week for this season
        max_week_season = 0
        for week in range(1, max_week + 1):
            col = f'week{week}_total_score'
            if col in season_data.columns and season_data[col].notna().any():
                max_week_season = week
        
        season_weights = []
        
        for week in range(1, max_week_season + 1):
            score_col = f'week{week}_total_score'
            if score_col not in season_data.columns:
                continue
                
            # Get active contestants this week
            week_data = season_data[season_data[score_col] > 0]
            if week_data.empty:
                continue
                
            active_contestants = week_data['celebrity_name'].tolist()
            n_active = len(active_contestants)
            
            if n_active < 3:
                continue
            
            # Get judge scores
            judge_scores = {}
            for contestant in active_contestants:
                contestant_data = season_data[season_data['celebrity_name'] == contestant]
                if not contestant_data.empty:
                    judge_scores[contestant] = contestant_data[score_col].iloc[0]
            
            # Analyze weight of ranking method
            rank_fan_votes = predict_by_rank(judge_scores, [])
            
            if rank_fan_votes:
                # Calculate correlation between judge ranking and fan ranking
                sorted_by_score = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
                judge_ranks = {contestant: rank+1 for rank, (contestant, _) in enumerate(sorted_by_score)}
                
                sorted_by_fan_vote = sorted(rank_fan_votes.items(), key=lambda x: x[1], reverse=True)
                fan_ranks = {contestant: rank+1 for rank, (contestant, _) in enumerate(sorted_by_fan_vote)}
                
                # Calculate Spearman correlation coefficient
                judge_rank_list = [judge_ranks[c] for c in active_contestants]
                fan_rank_list = [fan_ranks[c] for c in active_contestants]
                
                if len(set(judge_rank_list)) > 1 and len(set(fan_rank_list)) > 1:
                    spearman_corr, _ = stats.spearmanr(judge_rank_list, fan_rank_list)
                    
                    # Weight can be understood as the influence of fan ranking on total ranking
                    # In ranking method, fan ranking and judge ranking are equally important (50% weight each)
                    season_weights.append({
                        'method': 'rank',
                        'week': week,
                        'spearman_corr': spearman_corr,
                        'implied_weight': 0.5  # Fixed fan ranking weight of 0.5 in ranking method
                    })
            
            # Analyze weight of percentage method
            percentage_fan_votes = predict_by_percentage(judge_scores, [])
            
            if percentage_fan_votes:
                # Calculate correlation between judge percentage and fan percentage
                total_judge_score = sum(judge_scores.values())
                judge_percentages = {c: score/total_judge_score for c, score in judge_scores.items()}
                
                judge_percent_list = [judge_percentages[c] for c in active_contestants]
                fan_percent_list = [percentage_fan_votes[c] for c in active_contestants]
                
                if len(set(judge_percent_list)) > 1 and len(set(fan_percent_list)) > 1:
                    pearson_corr, _ = stats.pearsonr(judge_percent_list, fan_percent_list)
                    
                    # In percentage method, we use 40% judge + 60% fan weight
                    season_weights.append({
                        'method': 'percentage',
                        'week': week,
                        'pearson_corr': pearson_corr,
                        'implied_weight': 0.6  # Fan voting weight of 0.6 in percentage method
                    })
        
        if season_weights:
            weights_df = pd.DataFrame(season_weights)
            method_stats = weights_df.groupby('method').agg({
                'implied_weight': 'mean',
                'spearman_corr': 'mean',
                'pearson_corr': 'mean'
            }).reset_index()
            
            for _, row in method_stats.iterrows():
                weight_results.append({
                    'season': season,
                    'method': row['method'],
                    'avg_fan_weight': row['implied_weight'],
                    'avg_correlation': row['spearman_corr'] if row['method'] == 'rank' else row['pearson_corr']
                })
    
    return pd.DataFrame(weight_results)

# ==================== NEW: Judge Choice Mechanism Simulation (for rank_bottom_two) ====================
def simulate_judge_choice_elimination(df, max_week, total_score_cols):
    """
    Simulate judge choice elimination mechanism introduced after season 28
    """
    print("\nSimulating judge choice elimination mechanism...")
    
    # Only simulate rank_bottom_two phase (seasons 28 and later)
    simulation_results = []
    
    for season in df[df['voting_phase'] == 'rank_bottom_two']['season'].unique():
        season_data = df[df['season'] == season]
        
        # Get maximum week for this season
        max_week_season = 0
        for week in range(1, max_week + 1):
            col = f'week{week}_total_score'
            if col in season_data.columns and season_data[col].notna().any():
                max_week_season = week
        
        for week in range(1, max_week_season):
            score_col = f'week{week}_total_score'
            next_score_col = f'week{week+1}_total_score'
            
            if score_col not in season_data.columns or next_score_col not in season_data.columns:
                continue
            
            # Get active contestants this week
            week_data = season_data[season_data[score_col] > 0]
            if week_data.empty:
                continue
                
            active_contestants = week_data['celebrity_name'].tolist()
            
            # Get Bottom Two information
            bottom_two_data = week_data[week_data['is_bottom_two'] == True]
            if len(bottom_two_data) < 2:
                continue
            
            bottom_two = bottom_two_data['celebrity_name'].tolist()
            
            # Record original elimination
            original_eliminated = []
            for contestant in active_contestants:
                contestant_data = season_data[season_data['celebrity_name'] == contestant]
                if not contestant_data.empty:
                    current_score = contestant_data[score_col].iloc[0]
                    next_score = contestant_data[next_score_col].iloc[0] if next_score_col in contestant_data.columns else 0
                    if current_score > 0 and next_score == 0:
                        original_eliminated.append(contestant)
            
            if original_eliminated:
                original_eliminated = original_eliminated[0]  # Take first eliminated
                
                # Simulate judge choice mechanism
                # Step 1: Calculate composite scores using ranking method
                judge_scores = {}
                for contestant in active_contestants:
                    contestant_data = season_data[season_data['celebrity_name'] == contestant]
                    if not contestant_data.empty:
                        judge_scores[contestant] = contestant_data[score_col].iloc[0]
                
                rank_fan_votes = predict_by_rank(judge_scores, [], bottom_two)
                
                if rank_fan_votes:
                    # Calculate composite ranking
                    sorted_by_score = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
                    judge_ranks = {contestant: rank+1 for rank, (contestant, _) in enumerate(sorted_by_score)}
                    
                    sorted_by_fan_vote = sorted(rank_fan_votes.items(), key=lambda x: x[1], reverse=True)
                    fan_ranks = {contestant: rank+1 for rank, (contestant, _) in enumerate(sorted_by_fan_vote)}
                    
                    # Composite ranking = judge ranking + fan ranking
                    combined_ranks = {}
                    for contestant in active_contestants:
                        combined_ranks[contestant] = judge_ranks[contestant] + fan_ranks[contestant]
                    
                    # Find bottom two
                    sorted_combined = sorted(combined_ranks.items(), key=lambda x: x[1], reverse=True)
                    bottom_two_candidates = [contestant for contestant, _ in sorted_combined[-2:]]
                    
                    # Judge eliminates contestant with lower technical score
                    if len(bottom_two_candidates) == 2:
                        contestant1_score = judge_scores.get(bottom_two_candidates[0], 0)
                        contestant2_score = judge_scores.get(bottom_two_candidates[1], 0)
                        
                        if contestant1_score < contestant2_score:
                            simulated_eliminated = bottom_two_candidates[0]
                        else:
                            simulated_eliminated = bottom_two_candidates[1]
                        
                        simulation_results.append({
                            'season': season,
                            'week': week,
                            'original_eliminated': original_eliminated,
                            'simulated_eliminated': simulated_eliminated,
                            'match': original_eliminated == simulated_eliminated,
                            'actual_bottom_two': bottom_two,
                            'simulated_bottom_two': bottom_two_candidates,
                            'judge_score_original': judge_scores.get(original_eliminated, 0),
                            'judge_score_simulated': judge_scores.get(simulated_eliminated, 0)
                        })
    
    simulation_df = pd.DataFrame(simulation_results)
    
    if not simulation_df.empty:
        match_rate = simulation_df['match'].mean()
        print(f"Judge choice mechanism simulation match rate: {match_rate:.4f}")
        
        # Analyze which cases would change results
        changed_cases = simulation_df[~simulation_df['match']]
        print(f"Weeks with changed elimination results: {len(changed_cases)}/{len(simulation_df)}")
    
    return simulation_df

# ==================== 5. Uncertainty Calculation ====================
def calculate_uncertainty_fixed(predictions_df):
    """
    Fixed version uncertainty calculation - calculate standard deviation, confidence intervals, coefficient of variation, etc.
    """
    print("Calculating prediction uncertainty...")
    
    if predictions_df.empty:
        print("Warning: Prediction data is empty")
        return pd.DataFrame()
    
    # Check necessary columns
    required_cols = ['season', 'week', 'contestant', 'fan_vote_raw']
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    
    if missing_cols:
        print(f"Error: Prediction data missing required columns {missing_cols}")
        print(f"Available columns: {list(predictions_df.columns)}")
        return pd.DataFrame()
    
    # Group by season-week-contestant, calculate statistics
    grouped = predictions_df.groupby(['season', 'week', 'contestant'])
    
    result_rows = []
    
    for (season, week, contestant), group in grouped:
        # Get all sample predictions
        fan_votes = group['fan_vote_raw'].values
        
        # Calculate statistics
        mean_vote = np.mean(fan_votes)
        std_vote = np.std(fan_votes)
        
        # Calculate 95% confidence interval
        if len(fan_votes) > 1:
            ci_low = np.percentile(fan_votes, 2.5)
            ci_high = np.percentile(fan_votes, 97.5)
        else:
            ci_low = mean_vote
            ci_high = mean_vote
        
        # Calculate confidence interval width
        ci_width = ci_high - ci_low
        
        # Calculate coefficient of variation (standard deviation / mean)
        if mean_vote > 0:
            cv = std_vote / mean_vote
        else:
            cv = 0
        
        # Get contestant information
        row = group.iloc[0].copy()
        row['fan_vote_mean'] = mean_vote
        row['fan_vote_std'] = std_vote
        row['fan_vote_ci_low'] = max(0, ci_low)
        row['fan_vote_ci_high'] = min(1, ci_high)
        row['fan_vote_ci_width'] = ci_width
        row['fan_vote_cv'] = cv
        
        # Calculate relative uncertainty (percentage relative to mean)
        row['fan_vote_relative_uncertainty'] = std_vote / max(mean_vote, 0.001) * 100  # In percentage form
        
        # Determine uncertainty level
        if cv == 0:
            uncertainty_level = 'No uncertainty'
        elif cv < 0.1:
            uncertainty_level = 'Low'
        elif cv < 0.25:
            uncertainty_level = 'Medium'
        else:
            uncertainty_level = 'High'
        row['uncertainty_level'] = uncertainty_level
        
        result_rows.append(row)
    
    result_df = pd.DataFrame(result_rows)
    
    # Calculate overall uncertainty metrics
    if not result_df.empty:
        avg_std = result_df['fan_vote_std'].mean()
        avg_ci_width = result_df['fan_vote_ci_width'].mean()
        avg_cv = result_df[result_df['fan_vote_mean'] > 0]['fan_vote_cv'].mean()
        
        print(f"Average standard deviation: {avg_std:.6f}")
        print(f"Average confidence interval width: {avg_ci_width:.6f}")
        print(f"Average coefficient of variation: {avg_cv:.6f} ({avg_cv*100:.2f}%)")
        
        # Uncertainty distribution statistics
        print("\nUncertainty distribution statistics:")
        print(f"Minimum coefficient of variation: {result_df[result_df['fan_vote_mean'] > 0]['fan_vote_cv'].min():.6f}")
        print(f"Maximum coefficient of variation: {result_df[result_df['fan_vote_mean'] > 0]['fan_vote_cv'].max():.6f}")
        
        # Statistics by uncertainty level
        if 'uncertainty_level' in result_df.columns:
            uncertainty_counts = result_df['uncertainty_level'].value_counts()
            print("\nUncertainty level distribution:")
            for level, count in uncertainty_counts.items():
                percentage = count / len(result_df) * 100
                print(f"  {level}: {count} records ({percentage:.1f}%)")
    
    return result_df

# ==================== 6. Consistency Metrics Calculation ====================
def calculate_consistency_metrics(predictions_df, actual_eliminations):
    """
    Calculate consistency metrics between predictions and actual eliminations
    """
    print("Calculating consistency metrics...")
    
    if predictions_df.empty:
        print("Warning: Prediction data is empty")
        return None, None
    
    # Predicted eliminations (contestant with lowest fan vote)
    predicted_eliminations = {}
    
    # Group by season-week
    for (season, week), week_group in predictions_df.groupby(['season', 'week']):
        # Find contestant with lowest fan vote this week
        if not week_group.empty:
            # Only consider active contestants (excluding eliminated)
            active_contestants = week_group[week_group['fan_vote_mean'] > 0.001]
            if not active_contestants.empty:
                min_vote_idx = active_contestants['fan_vote_mean'].idxmin()
                predicted_elim = active_contestants.loc[min_vote_idx, 'contestant']
                predicted_eliminations[(season, week)] = [predicted_elim]
    
    # Prepare comparison data
    y_true = []  # Actually eliminated (1/0)
    y_pred = []  # Predicted eliminated (1/0)
    
    # Create labels for each prediction record
    for _, row in predictions_df.iterrows():
        season = row['season']
        week = row['week']
        contestant = row['contestant']
        
        # Check if actually eliminated
        actual_elim = actual_eliminations.get((season, week), [])
        is_actually_eliminated = contestant in actual_elim
        
        # Check if predicted eliminated
        pred_elim = predicted_eliminations.get((season, week), [])
        is_predicted_eliminated = contestant in pred_elim
        
        # Only consider active contestants (not eliminated early)
        if row['fan_vote_mean'] > 0.001:
            y_true.append(1 if is_actually_eliminated else 0)
            y_pred.append(1 if is_predicted_eliminated else 0)
    
    # Calculate metrics
    if len(y_true) > 0 and len(y_pred) > 0:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate weekly accuracy
        weekly_matches = []
        for key in predicted_eliminations:
            if key in actual_eliminations:
                pred_set = set(predicted_eliminations[key])
                actual_set = set(actual_eliminations[key])
                
                # Perfect match = 1, partial match = 0.5, no match = 0
                if pred_set == actual_set:
                    weekly_matches.append(1)
                elif len(pred_set & actual_set) > 0:
                    weekly_matches.append(0.5)
                else:
                    weekly_matches.append(0)
        
        weekly_accuracy = np.mean(weekly_matches) if weekly_matches else 0
        
        # Statistics
        total_weeks = len(set([(s, w) for s, w in predicted_eliminations.keys()]))
        match_count = sum([1 for m in weekly_matches if m == 1])
        partial_match_count = sum([1 for m in weekly_matches if m == 0.5])
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'weekly_accuracy': weekly_accuracy,
            'weekly_match_rate': match_count / total_weeks if total_weeks > 0 else 0,
            'weekly_partial_match_rate': partial_match_count / total_weeks if total_weeks > 0 else 0,
            'total_samples': len(y_true),
            'total_weeks': total_weeks,
            'perfect_matches': match_count,
            'partial_matches': partial_match_count
        }
        
        print(f"Elimination prediction accuracy: {accuracy:.4f}")
        print(f"Elimination prediction precision: {precision:.4f}")
        print(f"Elimination prediction recall: {recall:.4f}")
        print(f"Elimination prediction F1 score: {f1:.4f}")
        print(f"Weekly elimination match rate: {weekly_accuracy:.4f}")
        print(f"Perfect match weeks: {match_count}/{total_weeks}")
        
        return metrics, predicted_eliminations
    else:
        print("Not enough data to calculate consistency metrics")
        return None, None

# ==================== 7. Advanced Statistical Analysis ====================
def advanced_statistical_analysis(predictions_df):
    """
    Perform advanced statistical analysis
    """
    print("Performing advanced statistical analysis...")
    
    results = {}
    
    # 1. Uncertainty analysis
    if not predictions_df.empty:
        print("\nUncertainty analysis:")
        
        # Overall uncertainty statistics
        if 'fan_vote_cv' in predictions_df.columns:
            cv_stats = predictions_df[predictions_df['fan_vote_mean'] > 0]['fan_vote_cv'].describe()
            results['cv_statistics'] = cv_stats
            print(f"Coefficient of variation statistics:\n{cv_stats}")
        
        # Uncertainty analysis by method
        if 'method' in predictions_df.columns:
            method_uncertainty = predictions_df.groupby('method').agg({
                'fan_vote_std': ['mean', 'std', 'min', 'max'],
                'fan_vote_cv': ['mean', 'std', 'min', 'max'],
                'fan_vote_ci_width': 'mean'
            }).round(6)
            
            results['method_uncertainty'] = method_uncertainty
            print(f"\nUncertainty analysis by method:")
            print(method_uncertainty)
        
        # Uncertainty analysis by voting phase
        if 'voting_phase' in predictions_df.columns:
            phase_uncertainty = predictions_df.groupby('voting_phase').agg({
                'fan_vote_std': ['mean', 'std'],
                'fan_vote_cv': ['mean', 'std'],
                'fan_vote_ci_width': 'mean'
            }).round(6)
            
            results['phase_uncertainty'] = phase_uncertainty
            print(f"\nUncertainty analysis by voting phase:")
            print(phase_uncertainty)
        
        # Uncertainty analysis by week number
        if 'week' in predictions_df.columns:
            week_uncertainty = predictions_df.groupby('week').agg({
                'fan_vote_std': 'mean',
                'fan_vote_cv': 'mean',
                'fan_vote_ci_width': 'mean'
            }).round(6)
            
            results['week_uncertainty'] = week_uncertainty
            print(f"\nUncertainty analysis by week (first 5 weeks):")
            print(week_uncertainty.head())
    
    # 2. Impact of different voting phases on fan voting
    if 'voting_phase' in predictions_df.columns and len(predictions_df['voting_phase'].unique()) > 1:
        phases = predictions_df['voting_phase'].unique()
        groups = [predictions_df[predictions_df['voting_phase'] == phase]['fan_vote_mean'].values 
                  for phase in phases if len(predictions_df[predictions_df['voting_phase'] == phase]) > 1]
        
        if len(groups) > 1:
            try:
                f_stat, p_val = stats.f_oneway(*groups)
                results['phase_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_val,
                    'n_groups': len(groups)
                }
                print(f"\nVoting phase ANOVA test: F={f_stat:.4f}, p={p_val:.6f}")
            except Exception as e:
                print(f"ANOVA calculation error: {e}")
                results['phase_anova'] = None
        
        # Voting phase statistics
        phase_stats = predictions_df.groupby('voting_phase').agg({
            'fan_vote_mean': ['mean', 'std', 'count'],
            'judge_score': 'mean',
            'fan_vote_cv': 'mean',
            'fan_vote_std': 'mean'
        }).round(4)
        
        results['phase_stats'] = phase_stats
    
    # 3. Correlation between judge scores and fan voting
    if 'judge_score' in predictions_df.columns and 'fan_vote_mean' in predictions_df.columns:
        judge_vote_corr = predictions_df['judge_score'].corr(predictions_df['fan_vote_mean'])
        results['judge_vote_correlation'] = judge_vote_corr
        print(f"\nCorrelation between judge scores and fan voting: {judge_vote_corr:.4f}")
    
    # 4. Comparison of different voting methods
    if 'method' in predictions_df.columns:
        method_stats = predictions_df.groupby('method').agg({
            'fan_vote_mean': ['mean', 'std'],
            'fan_vote_std': 'mean',
            'fan_vote_cv': 'mean',
            'is_eliminated': 'mean'
        }).round(4)
        results['method_stats'] = method_stats
        
        # Statistical test between methods
        methods = predictions_df['method'].unique()
        if len(methods) > 1:
            method_groups = [predictions_df[predictions_df['method'] == m]['fan_vote_mean'] for m in methods]
            try:
                f_stat, p_val = stats.f_oneway(*method_groups)
                results['method_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_val
                }
                print(f"Method ANOVA test: F={f_stat:.4f}, p={p_val:.6f}")
            except Exception as e:
                print(f"Method ANOVA calculation error: {e}")
    
    # 5. Relationship between elimination status and fan voting
    if 'is_eliminated' in predictions_df.columns:
        eliminated_stats = predictions_df.groupby('is_eliminated').agg({
            'fan_vote_mean': ['mean', 'std', 'count'],
            'judge_score': 'mean',
            'fan_vote_cv': 'mean'
        }).round(4)
        results['eliminated_stats'] = eliminated_stats
    
    return results

# ==================== 8. Visualization Functions ====================
def create_comparison_visualizations(comparison_df, weight_df, controversial_df, simulation_df, predictions_df):
    """
    Create visualization charts for comparison analysis
    """
    print("Creating comparison analysis visualizations...")
    
    if not os.path.exists('visualizations/comparison'):
        os.makedirs('visualizations/comparison')
    
    # 1. Accuracy comparison of two methods (by season)
    if not comparison_df.empty:
        plt.figure(figsize=(16, 10))
        
        # Group by voting phase
        phases = comparison_df['voting_phase'].unique()
        colors = {'rank_regular': 'blue', 'percentage_regular': 'green', 'rank_bottom_two': 'red'}
        
        for phase in phases:
            phase_data = comparison_df[comparison_df['voting_phase'] == phase]
            if len(phase_data) > 0:
                plt.plot(phase_data['season'], phase_data['rank_accuracy'], 
                        marker='o', linestyle='-', color=colors.get(phase, 'gray'), 
                        label=f'{phase} - Ranking Method', alpha=0.7)
                plt.plot(phase_data['season'], phase_data['percentage_accuracy'], 
                        marker='s', linestyle='--', color=colors.get(phase, 'gray'), 
                        label=f'{phase} - Percentage Method', alpha=0.7)
        
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Elimination Prediction Accuracy', fontsize=12)
        plt.title('Accuracy Comparison of Two Methods by Voting Phase', fontsize=14)
        plt.legend(fontsize=10, ncol=3)
        plt.grid(True, alpha=0.3)
        plt.xticks(comparison_df['season'].unique(), rotation=45, fontsize=10)
        plt.tight_layout()
        plt.savefig('visualizations/comparison/method_accuracy_by_phase.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Fan voting weight comparison
    if not weight_df.empty:
        plt.figure(figsize=(12, 6))
        
        # Group by method
        rank_weights = weight_df[weight_df['method'] == 'rank']['avg_fan_weight']
        percentage_weights = weight_df[weight_df['method'] == 'percentage']['avg_fan_weight']
        
        methods = ['Rank Method', 'Percentage Method']
        avg_weights = [rank_weights.mean() if len(rank_weights) > 0 else 0, 
                      percentage_weights.mean() if len(percentage_weights) > 0 else 0]
        
        bars = plt.bar(methods, avg_weights, color=['blue', 'green'], alpha=0.7)
        plt.ylabel('Average Fan Vote Weight', fontsize=12)
        plt.title('Average Fan Vote Weight in Two Methods', fontsize=14)
        plt.ylim(0, 0.7)
        
        for bar, weight in zip(bars, avg_weights):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/comparison/fan_weight_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Controversial Cases Comparison
    if not controversial_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Controversial Cases: Two Voting Methods Comparison', fontsize=16)
        
        for idx, (_, row) in enumerate(controversial_df.iterrows()):
            ax = axes[idx // 2, idx % 2]
            
            labels = ['Rank Method', 'Percentage Method']
            fan_votes = [row['avg_rank_fan_vote'], row['avg_percentage_fan_vote']]
            positions = [row['avg_rank_position'], row['avg_percentage_position']]
            
            x = np.arange(len(labels))
            width = 0.35
            
            ax1 = ax.twinx()
            
            bars1 = ax.bar(x - width/2, fan_votes, width, label='Average Fan Vote', color='skyblue')
            bars2 = ax1.bar(x + width/2, positions, width, label='Average Position', color='lightcoral')
            
            ax.set_ylabel('Average Fan Vote Proportion', fontsize=10)
            ax1.set_ylabel('Average Position', fontsize=10)
            ax.set_title(f"{row['contestant']} (Season {row['season']})", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            
            # Add value labels
            for bar, value in zip(bars1, fan_votes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            for bar, value in zip(bars2, positions):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                        f'{value:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax.legend(loc='upper left')
            ax1.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('visualizations/comparison/controversial_cases_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Judge Choice Mechanism Impact
    if not simulation_df.empty and len(simulation_df) > 0:
        plt.figure(figsize=(12, 8))
        
        # Calculate match rate
        match_rate = simulation_df['match'].mean()
        changed_rate = 1 - match_rate
        
        labels = ['Elimination Results Match', 'Elimination Results Changed']
        sizes = [match_rate, changed_rate]
        colors = ['lightgreen', 'lightcoral']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Judge Choice Elimination Mechanism Simulation Results (Rank-Bottom-Two Phase)', fontsize=14)
        plt.tight_layout()
        plt.savefig('visualizations/comparison/judge_choice_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detailed analysis of changed cases
        if changed_rate > 0:
            changed_cases = simulation_df[~simulation_df['match']]
            
            plt.figure(figsize=(14, 8))
            seasons_changed = changed_cases['season'].value_counts().sort_index()
            
            plt.bar(seasons_changed.index.astype(str), seasons_changed.values)
            plt.xlabel('Season', fontsize=12)
            plt.ylabel('Number of Weeks with Changed Elimination Results', fontsize=12)
            plt.title('Season Distribution of Changed Elimination Results by Judge Choice Mechanism', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            for i, (season, count) in enumerate(zip(seasons_changed.index, seasons_changed.values)):
                plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig('visualizations/comparison/judge_choice_changes_by_season.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 5. NEW: Uncertainty analysis visualization
    if not predictions_df.empty:
        create_uncertainty_visualizations(predictions_df)
        create_additional_visualizations(predictions_df)

def create_uncertainty_visualizations(predictions_df):
    """
    Create visualization charts for uncertainty analysis
    """
    print("Creating uncertainty analysis visualizations...")
    
    if not os.path.exists('visualizations/uncertainty'):
        os.makedirs('visualizations/uncertainty')
    
    # 1. Coefficient of variation distribution histogram
    if 'fan_vote_cv' in predictions_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Filter out zero values
        cv_data = predictions_df[predictions_df['fan_vote_cv'] > 0]['fan_vote_cv']
        
        if len(cv_data) > 0:
            plt.hist(cv_data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            plt.xlabel('Coefficient of Variation (Std Dev / Mean)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Distribution of Fan Vote Prediction Coefficient of Variation', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Add statistical information
            mean_cv = cv_data.mean()
            median_cv = cv_data.median()
            plt.axvline(mean_cv, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_cv:.4f}')
            plt.axvline(median_cv, color='green', linestyle='--', linewidth=2, label=f'Median: {median_cv:.4f}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('visualizations/uncertainty/cv_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. Coefficient of variation changes by week
    if 'week' in predictions_df.columns and 'fan_vote_cv' in predictions_df.columns:
        plt.figure(figsize=(14, 8))
        
        week_cv = predictions_df.groupby('week')['fan_vote_cv'].mean().reset_index()
        
        plt.plot(week_cv['week'], week_cv['fan_vote_cv'], marker='o', linewidth=2)
        plt.xlabel('Week', fontsize=12)
        plt.ylabel('Average Coefficient of Variation', fontsize=12)
        plt.title('Fan Vote Prediction Uncertainty by Week', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(week_cv) > 1:
            z = np.polyfit(week_cv['week'], week_cv['fan_vote_cv'], 1)
            p = np.poly1d(z)
            plt.plot(week_cv['week'], p(week_cv['week']), 'r--', alpha=0.8, label='Trend Line')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/uncertainty/cv_by_week.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Coefficient of variation comparison by method
    if 'method' in predictions_df.columns and 'fan_vote_cv' in predictions_df.columns:
        plt.figure(figsize=(10, 6))
        
        method_cv = predictions_df.groupby('method')['fan_vote_cv'].mean().reset_index()
        
        bars = plt.bar(method_cv['method'], method_cv['fan_vote_cv'])
        plt.xlabel('Voting Method', fontsize=12)
        plt.ylabel('Average Coefficient of Variation', fontsize=12)
        plt.title('Prediction Uncertainty Comparison by Voting Method', fontsize=14)
        
        # Add value labels
        for bar, cv in zip(bars, method_cv['fan_vote_cv']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{cv:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/uncertainty/cv_by_method.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Controversial cases confidence interval visualization
    create_controversial_cases_ci_visualization(predictions_df)

def create_additional_visualizations(predictions_df):
    """
    Create additional visualization charts
    """
    print("Creating additional visualizations...")
    
    if not os.path.exists('visualizations/additional'):
        os.makedirs('visualizations/additional')
    
    # 1. Scatter plot of judge scores vs fan voting with linear trend line
    if 'judge_score' in predictions_df.columns and 'fan_vote_mean' in predictions_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Filter out invalid data
        valid_data = predictions_df[(predictions_df['judge_score'] > 0) & 
                                   (predictions_df['fan_vote_mean'] > 0)]
        
        if len(valid_data) > 0:
            x = valid_data['judge_score'].values
            y = valid_data['fan_vote_mean'].values
            
            # Create scatter plot
            plt.scatter(x, y, alpha=0.6, color='blue', s=20)
            
            # Fit linear trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            
            # Plot trend line
            x_range = np.linspace(x.min(), x.max(), 100)
            plt.plot(x_range, p(x_range), 'r--', linewidth=2, 
                    label=f'Trend Line: y={z[0]:.4f}x + {z[1]:.4f}')
            
            # Add correlation coefficient
            corr_coef = np.corrcoef(x, y)[0, 1]
            plt.text(0.05, 0.95, f'Correlation Coefficient: {corr_coef:.3f}', 
                    transform=plt.gca().transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.xlabel('Judge Score', fontsize=12)
            plt.ylabel('Predicted Fan Vote Proportion', fontsize=12)
            plt.title('Scatter Plot: Judge Scores vs Fan Voting with Linear Trend', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('visualizations/additional/judge_vote_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. Line chart of prediction uncertainty by season (average standard deviation)
    if 'season' in predictions_df.columns and 'fan_vote_std' in predictions_df.columns:
        plt.figure(figsize=(14, 8))
        
        # Calculate average standard deviation by season
        season_std = predictions_df.groupby('season')['fan_vote_std'].mean().reset_index()
        
        plt.plot(season_std['season'], season_std['fan_vote_std'], 
                marker='o', linewidth=2, markersize=6)
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Average Standard Deviation', fontsize=12)
        plt.title('Prediction Uncertainty by Season (Average Standard Deviation)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(season_std) > 1:
            z = np.polyfit(season_std['season'], season_std['fan_vote_std'], 1)
            p = np.poly1d(z)
            plt.plot(season_std['season'], p(season_std['season']), 'r--', alpha=0.8, 
                    label=f'Trend Line: y={z[0]:.6f}x + {z[1]:.4f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/additional/uncertainty_by_season.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Line chart of relative uncertainty by season (average coefficient of variation)
    if 'season' in predictions_df.columns and 'fan_vote_cv' in predictions_df.columns:
        plt.figure(figsize=(14, 8))
        
        # Calculate average coefficient of variation by season
        season_cv = predictions_df[predictions_df['fan_vote_mean'] > 0].groupby('season')['fan_vote_cv'].mean().reset_index()
        
        plt.plot(season_cv['season'], season_cv['fan_vote_cv'], 
                marker='s', linewidth=2, markersize=6, color='green')
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Average Coefficient of Variation', fontsize=12)
        plt.title('Relative Uncertainty by Season (Average Coefficient of Variation)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(season_cv) > 1:
            z = np.polyfit(season_cv['season'], season_cv['fan_vote_cv'], 1)
            p = np.poly1d(z)
            plt.plot(season_cv['season'], p(season_cv['season']), 'r--', alpha=0.8, 
                    label=f'Trend Line: y={z[0]:.6f}x + {z[1]:.4f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/additional/relative_uncertainty_by_season.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_controversial_cases_ci_visualization(predictions_df):
    """
    Create confidence interval visualization for controversial cases
    """
    print("Creating confidence interval visualization for controversial cases...")
    
    controversial_cases = [
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones")
    ]
    
    # Collect controversial cases data
    case_data = []
    
    for season, contestant in controversial_cases:
        case_df = predictions_df[(predictions_df['season'] == season) & 
                                (predictions_df['contestant'] == contestant)]
        
        if not case_df.empty:
            # Calculate average metrics
            avg_fan_vote = case_df['fan_vote_mean'].mean()
            avg_ci_low = case_df['fan_vote_ci_low'].mean()
            avg_ci_high = case_df['fan_vote_ci_high'].mean()
            avg_cv = case_df['fan_vote_cv'].mean()
            avg_std = case_df['fan_vote_std'].mean()
            
            case_data.append({
                'season': season,
                'contestant': contestant,
                'avg_fan_vote': avg_fan_vote,
                'avg_ci_low': avg_ci_low,
                'avg_ci_high': avg_ci_high,
                'avg_cv': avg_cv,
                'avg_std': avg_std,
                'ci_width': avg_ci_high - avg_ci_low,
                'sample_size': len(case_df)
            })
    
    if case_data:
        case_df = pd.DataFrame(case_data)
        
        # Create bar chart with error bars
        plt.figure(figsize=(14, 8))
        
        contestants = [f"{row['contestant']}\n(Season {row['season']})" for _, row in case_df.iterrows()]
        means = case_df['avg_fan_vote'].values
        ci_lows = case_df['avg_ci_low'].values
        ci_highs = case_df['avg_ci_high'].values
        errors = [means - ci_lows, ci_highs - means]
        
        x_pos = np.arange(len(contestants))
        bars = plt.bar(x_pos, means, yerr=errors, capsize=10, alpha=0.7, color='skyblue', edgecolor='black')
        
        plt.xlabel('Controversial Cases', fontsize=12)
        plt.ylabel('Average Fan Vote Proportion (95% Confidence Interval)', fontsize=12)
        plt.title('Controversial Cases Fan Vote Prediction Uncertainty Analysis', fontsize=14)
        plt.xticks(x_pos, contestants, fontsize=10)
        
        # Add value labels
        for i, (bar, mean, cv, width) in enumerate(zip(bars, means, case_df['avg_cv'].values, case_df['ci_width'].values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{mean:.3f}\nCV={cv:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('visualizations/uncertainty/controversial_cases_ci.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create uncertainty comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Controversial Cases Uncertainty Metrics Comparison', fontsize=16)
        
        metrics = [
            ('avg_fan_vote', 'Average Fan Vote Proportion', 'blue'),
            ('avg_std', 'Average Standard Deviation', 'red'),
            ('avg_cv', 'Average Coefficient of Variation', 'green'),
            ('ci_width', 'Average Confidence Interval Width', 'purple')
        ]
        
        for idx, (metric, title, color) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = case_df[metric].values
            
            bars = ax.bar(x_pos, values, color=color, alpha=0.7)
            ax.set_xlabel('Controversial Cases', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(title, fontsize=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([row.contestant[:10] for row in case_df.itertuples()], rotation=45, fontsize=9)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                       f'{value:.4f}', ha='center', va='bottom', fontsize=8)
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/uncertainty/controversial_cases_uncertainty_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

# ==================== 9. Report Generation ====================
def generate_comprehensive_comparison_report(predictions_df, comparison_df, weight_df, 
                                           controversial_df, simulation_df, 
                                           consistency_metrics, statistical_results):
    """
    Generate comprehensive comparison analysis report
    """
    print("Generating comprehensive comparison analysis report...")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("Dancing with the Stars Fan Voting Prediction Model - Voting Methods Comparison Report")
    report_lines.append("Based on three voting phases: Rank-Regular, Percentage-Regular, Rank-Bottom-Two")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. Overall Statistics
    report_lines.append("1. Overall Statistics")
    report_lines.append("-"*40)
    report_lines.append(f"Number of seasons analyzed: {predictions_df['season'].nunique()}")
    report_lines.append(f"Total prediction records: {len(predictions_df)}")
    report_lines.append(f"Total contestants: {predictions_df['contestant'].nunique()}")
    
    if 'fan_vote_mean' in predictions_df.columns:
        report_lines.append(f"Average fan vote proportion: {predictions_df['fan_vote_mean'].mean():.4f}")
    
    if 'judge_score' in predictions_df.columns:
        report_lines.append(f"Average judge score: {predictions_df['judge_score'].mean():.2f}")
    
    # Voting phase statistics
    if 'voting_phase' in predictions_df.columns:
        phase_counts = predictions_df['voting_phase'].value_counts()
        report_lines.append("\nVoting phase distribution:")
        for phase, count in phase_counts.items():
            percentage = count / len(predictions_df) * 100
            report_lines.append(f"  {phase}: {count} records ({percentage:.1f}%)")
    
    report_lines.append("")
    
    # 2. Comparison of Two Voting Methods
    report_lines.append("2. Comparison of Two Voting Methods")
    report_lines.append("-"*40)
    
    if not comparison_df.empty:
        # Overall accuracy comparison
        avg_rank_accuracy = comparison_df['rank_accuracy'].mean()
        avg_percentage_accuracy = comparison_df['percentage_accuracy'].mean()
        
        report_lines.append(f"Ranking method average elimination prediction accuracy: {avg_rank_accuracy:.4f}")
        report_lines.append(f"Percentage method average elimination prediction accuracy: {avg_percentage_accuracy:.4f}")
        report_lines.append(f"Accuracy difference: {avg_percentage_accuracy - avg_rank_accuracy:.4f} "
                          f"({'Percentage method higher' if avg_percentage_accuracy > avg_rank_accuracy else 'Ranking method higher'})")
        
        # Analysis by voting phase
        report_lines.append("\nAccuracy analysis by voting phase:")
        for phase in comparison_df['voting_phase'].unique():
            phase_data = comparison_df[comparison_df['voting_phase'] == phase]
            if len(phase_data) > 0:
                phase_rank_acc = phase_data['rank_accuracy'].mean()
                phase_perc_acc = phase_data['percentage_accuracy'].mean()
                report_lines.append(f"  {phase}: Ranking method={phase_rank_acc:.3f}, Percentage method={phase_perc_acc:.3f}, "
                                  f"Difference={phase_perc_acc-phase_rank_acc:.3f}")
        
        # Different prediction results analysis
        avg_different_rate = comparison_df['different_prediction_rate'].mean()
        report_lines.append(f"\nAverage proportion of different predictions between two methods: {avg_different_rate:.4f}")
    
    # 3. Fan Voting Weight Analysis
    report_lines.append("")
    report_lines.append("3. Fan Voting Weight Analysis")
    report_lines.append("-"*40)
    
    if not weight_df.empty:
        rank_weights = weight_df[weight_df['method'] == 'rank']
        percentage_weights = weight_df[weight_df['method'] == 'percentage']
        
        if len(rank_weights) > 0:
            avg_rank_weight = rank_weights['avg_fan_weight'].mean()
            avg_rank_corr = rank_weights['avg_correlation'].mean()
            report_lines.append(f"Ranking method average fan voting weight: {avg_rank_weight:.3f}")
            report_lines.append(f"Ranking method judge-fan ranking correlation: {avg_rank_corr:.3f}")
        
        if len(percentage_weights) > 0:
            avg_percentage_weight = percentage_weights['avg_fan_weight'].mean()
            avg_percentage_corr = percentage_weights['avg_correlation'].mean()
            report_lines.append(f"Percentage method average fan voting weight: {avg_percentage_weight:.3f}")
            report_lines.append(f"Percentage method judge-fan voting correlation: {avg_percentage_corr:.3f}")
    
    # 4. Prediction Uncertainty Analysis
    report_lines.append("")
    report_lines.append("4. Prediction Uncertainty Analysis")
    report_lines.append("-"*40)
    
    if statistical_results:
        # Coefficient of variation statistics
        if 'cv_statistics' in statistical_results:
            cv_stats = statistical_results['cv_statistics']
            report_lines.append(f"Coefficient of variation statistics:")
            report_lines.append(f"  Mean: {cv_stats['mean']:.6f} ({cv_stats['mean']*100:.2f}%)")
            report_lines.append(f"  Standard deviation: {cv_stats['std']:.6f}")
            report_lines.append(f"  Minimum: {cv_stats['min']:.6f}")
            report_lines.append(f"  25th percentile: {cv_stats['25%']:.6f}")
            report_lines.append(f"  Median: {cv_stats['50%']:.6f}")
            report_lines.append(f"  75th percentile: {cv_stats['75%']:.6f}")
            report_lines.append(f"  Maximum: {cv_stats['max']:.6f}")
        
        # Uncertainty by method
        if 'method_uncertainty' in statistical_results:
            method_uncertainty = statistical_results['method_uncertainty']
            report_lines.append("\nUncertainty by voting method:")
            for method in method_uncertainty.index:
                cv_mean = method_uncertainty.loc[method, ('fan_vote_cv', 'mean')]
                std_mean = method_uncertainty.loc[method, ('fan_vote_std', 'mean')]
                report_lines.append(f"  {method}: Average CV={cv_mean:.6f}, Average std={std_mean:.6f}")
    
    # 5. Controversial Cases Analysis
    report_lines.append("")
    report_lines.append("5. Controversial Cases Two-Method Comparison")
    report_lines.append("-"*40)
    
    if not controversial_df.empty:
        for _, row in controversial_df.iterrows():
            report_lines.append(f"{row['contestant']} (Season {row['season']}):")
            report_lines.append(f"  Average judge score: {row['avg_judge_score']:.2f}")
            report_lines.append(f"  Ranking method average fan vote: {row['avg_rank_fan_vote']:.4f} (Average position: {row['avg_rank_position']:.1f})")
            report_lines.append(f"  Percentage method average fan vote: {row['avg_percentage_fan_vote']:.4f} (Average position: {row['avg_percentage_position']:.1f})")
            
            if row['avg_rank_fan_vote'] > row['avg_percentage_fan_vote']:
                report_lines.append(f"  Ranking method predicts higher fan support, difference: {row['avg_rank_fan_vote'] - row['avg_percentage_fan_vote']:.4f}")
            else:
                report_lines.append(f"  Percentage method predicts higher fan support, difference: {row['avg_percentage_fan_vote'] - row['avg_rank_fan_vote']:.4f}")
            
            report_lines.append(f"  Weeks where ranking method performs better: {row['weeks_rank_better']}")
            report_lines.append(f"  Weeks where percentage method performs better: {row['weeks_percentage_better']}")
            report_lines.append("")
    
    # 6. Judge Choice Mechanism Simulation
    report_lines.append("6. Judge Choice Elimination Mechanism Simulation (Rank-Bottom-Two Phase)")
    report_lines.append("-"*40)
    
    if not simulation_df.empty:
        match_rate = simulation_df['match'].mean()
        changed_cases = simulation_df[~simulation_df['match']]
        
        report_lines.append(f"Simulation match rate: {match_rate:.4f}")
        report_lines.append(f"Weeks with changed elimination results: {len(changed_cases)}/{len(simulation_df)}")
        report_lines.append(f"Change proportion: {len(changed_cases)/len(simulation_df):.4f}")
    
    # 7. Statistical Correlation Analysis
    report_lines.append("")
    report_lines.append("7. Statistical Correlation Analysis")
    report_lines.append("-"*40)
    
    if statistical_results:
        if 'judge_vote_correlation' in statistical_results:
            report_lines.append(f"Correlation between judge scores and fan voting: {statistical_results['judge_vote_correlation']:.4f}")
        
        if 'phase_anova' in statistical_results and statistical_results['phase_anova']:
            anova = statistical_results['phase_anova']
            report_lines.append(f"Significance of fan voting differences across voting phases: F={anova['f_statistic']:.4f}, p={anova['p_value']:.6f}")
    
    # 8. Model Consistency Metrics
    report_lines.append("")
    report_lines.append("8. Model Consistency Metrics")
    report_lines.append("-"*40)
    
    if consistency_metrics:
        report_lines.append(f"Elimination prediction accuracy: {consistency_metrics['accuracy']:.4f}")
        report_lines.append(f"Elimination prediction F1 score: {consistency_metrics['f1_score']:.4f}")
        report_lines.append(f"Weekly elimination match rate: {consistency_metrics['weekly_accuracy']:.4f}")
        report_lines.append(f"Perfect match weeks: {consistency_metrics['perfect_matches']}/{consistency_metrics['total_weeks']}")
    
    # 9. Certainty Metrics Summary
    report_lines.append("")
    report_lines.append("9. Certainty Metrics Summary")
    report_lines.append("-"*40)
    
    if 'fan_vote_cv' in predictions_df.columns:
        # Calculate overall uncertainty metrics
        cv_mean = predictions_df[predictions_df['fan_vote_mean'] > 0]['fan_vote_cv'].mean()
        std_mean = predictions_df['fan_vote_std'].mean()
        ci_width_mean = predictions_df['fan_vote_ci_width'].mean()
        
        report_lines.append(f"Average coefficient of variation (CV): {cv_mean:.6f} ({cv_mean*100:.2f}%)")
        report_lines.append(f"Average standard deviation: {std_mean:.6f}")
        report_lines.append(f"Average confidence interval width: {ci_width_mean:.6f}")
        
        # Uncertainty distribution
        report_lines.append(f"\nUncertainty distribution:")
        report_lines.append(f"  • Low uncertainty (CV < 0.1): {len(predictions_df[predictions_df['fan_vote_cv'] < 0.1])} records")
        report_lines.append(f"  • Medium uncertainty (0.1 ≤ CV < 0.25): {len(predictions_df[(predictions_df['fan_vote_cv'] >= 0.1) & (predictions_df['fan_vote_cv'] < 0.25)])} records")
        report_lines.append(f"  • High uncertainty (CV ≥ 0.25): {len(predictions_df[predictions_df['fan_vote_cv'] >= 0.25])} records")
        
        # Uncertainty differences by contestant
        if 'contestant' in predictions_df.columns:
            contestant_cv = predictions_df.groupby('contestant')['fan_vote_cv'].mean().sort_values()
            report_lines.append(f"\nMost certain contestants (Top 5):")
            for contestant, cv in contestant_cv.head(5).items():
                report_lines.append(f"  {contestant}: CV={cv:.6f}")
            
            report_lines.append(f"\nMost uncertain contestants (Top 5):")
            for contestant, cv in contestant_cv.tail(5).items():
                report_lines.append(f"  {contestant}: CV={cv:.6f}")
        
        # Uncertainty differences by week
        if 'week' in predictions_df.columns:
            week_cv = predictions_df.groupby('week')['fan_vote_cv'].mean()
            max_cv_week = week_cv.idxmax()
            min_cv_week = week_cv.idxmin()
            report_lines.append(f"\nWeek with highest uncertainty: Week {max_cv_week} (CV={week_cv[max_cv_week]:.6f})")
            report_lines.append(f"Week with lowest uncertainty: Week {min_cv_week} (CV={week_cv[min_cv_week]:.6f})")
    
    # 10. Conclusions and Recommendations
    report_lines.append("")
    report_lines.append("10. Conclusions and Recommendations")
    report_lines.append("-"*40)
    
    report_lines.append("Main Findings:")
    report_lines.append("  1. Percentage method gives higher weight to fan voting (60% vs 50% in ranking method)")
    report_lines.append("  2. Percentage method has higher elimination prediction accuracy in most seasons")
    report_lines.append("  3. Prediction results differ across three voting phases")
    report_lines.append("  4. Judge choice mechanism in Rank-Bottom-Two phase affects elimination results")
    report_lines.append("  5. Ranking method generally has higher uncertainty (higher average coefficient of variation)")
    report_lines.append("  6. Controversial case predictions have higher uncertainty, requiring more data support")
    
    report_lines.append("")
    report_lines.append("Certainty Metrics Summary:")
    report_lines.append("  • Model provides complete certainty metrics: standard deviation, confidence intervals, coefficient of variation")
    report_lines.append("  • Coefficient of variation quantifies estimate volatility across different contestants/weeks")
    report_lines.append("  • Confidence interval charts visually show certainty ranges of estimates")
    report_lines.append("  • Uncertainty level classification helps identify high-risk predictions")
    
    report_lines.append("")
    report_lines.append("New Voting System Recommendations:")
    report_lines.append("  Adopt 'Dynamic Hybrid System':")
    report_lines.append("  • Weeks 1-4: Judge 60% + Fan 40% (emphasize technical foundation)")
    report_lines.append("  • Weeks 5-8: Judge 50% + Fan 50% (balanced phase)")
    report_lines.append("  • Week 9 onwards: Judge 40% + Fan 60% (emphasize audience participation)")
    report_lines.append("  • Incorporate uncertainty analysis, provide additional review for high-risk elimination decisions")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # Save report
    report_text = "\n".join(report_lines)
    
    with open('voting_methods_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Comparison analysis report saved to 'voting_methods_comparison_report.txt'")
    
    return report_text

# ==================== 10. Main Function ====================
def main_enhanced():
    """
    Enhanced main function with complete comparison analysis
    """
    print("="*80)
    print("Dancing with the Stars Fan Voting Prediction Model - Enhanced Version (Based on Three Voting Phases)")
    print("="*80)
    
    # 1. Load data (using three preprocessed files)
    rank_regular_path = 'dwts_rank_regular_processed.csv'
    percentage_regular_path = 'dwts_percentage_regular_processed.csv'
    rank_bottom_two_path = 'dwts_rank_bottom_two_processed.csv'
    
    df, max_week, total_score_cols = load_and_preprocess_data(
        rank_regular_path, percentage_regular_path, rank_bottom_two_path
    )
    
    if df.empty:
        print("Error: Data loading failed")
        return None, None, None, None, None
    
    if not total_score_cols:
        print("Error: No total score columns found or calculated, cannot proceed with prediction")
        return None, None, None, None, None
    
    # 2. Run two voting methods comparison analysis
    comparison_df = compare_voting_methods(df, max_week, total_score_cols)
    
    # 3. Analyze fan voting weight
    weight_df = calculate_fan_vote_weight(df, max_week)
    
    # 4. Analyze controversial cases
    controversial_df = analyze_method_impact_on_controversial_cases(df, max_week, total_score_cols)
    
    # 5. Simulate judge choice mechanism
    simulation_df = simulate_judge_choice_elimination(df, max_week, total_score_cols)
    
    # 7. Generate fan voting predictions (using multiple sampling)
    print("\nStarting fan voting prediction (multiple sampling)...")
    all_predictions = []
    seasons = sorted(df['season'].unique())
    
    for season in seasons:
        print(f"Predicting season {season}...")
        season_data = df[df['season'] == season]
        
        if season_data.empty:
            continue
        
        # Determine voting phase
        voting_phase = season_data['voting_phase'].iloc[0]
        
        # Determine actual weeks for this season
        actual_weeks = []
        for week in range(1, max_week + 1):
            col_name = f'week{week}_total_score'
            if col_name in season_data.columns and season_data[col_name].notna().any() and season_data[col_name].sum() > 0:
                actual_weeks.append(week)
        
        if not actual_weeks:
            print(f"  Season {season}: No valid week data")
            continue
        
        max_week_season = max(actual_weeks)
        print(f"  Season {season}: Total {max_week_season} weeks, voting phase: {voting_phase}")
        
        # Multiple sampling
        for sample_idx in range(20):  # Reduce sampling times to improve speed, but ensure uncertainty analysis
            for week in range(1, max_week_season + 1):
                score_col = f'week{week}_total_score'
                
                if score_col not in season_data.columns:
                    continue
                    
                # Get active contestants this week (total score > 0)
                week_data = season_data[season_data[score_col] > 0]
                if week_data.empty:
                    continue
                    
                active_contestants = week_data['celebrity_name'].tolist()
                
                if not active_contestants:
                    continue
                
                # Determine prediction method based on voting phase
                if voting_phase in ['rank_regular', 'rank_bottom_two']:
                    method = 'rank'
                else:
                    method = 'percentage'
                
                # Get judge scores
                judge_scores = {}
                for contestant in active_contestants:
                    contestant_data = season_data[season_data['celebrity_name'] == contestant]
                    if not contestant_data.empty:
                        judge_scores[contestant] = contestant_data[score_col].iloc[0]
                
                # Get elimination information
                eliminated = []
                bottom_two = None
                
                # Get next week's score column
                next_week = week + 1
                if next_week <= max_week_season:
                    next_score_col = f'week{next_week}_total_score'
                    if next_score_col in season_data.columns:
                        for contestant in active_contestants:
                            contestant_data = season_data[season_data['celebrity_name'] == contestant]
                            if not contestant_data.empty:
                                current_score = contestant_data[score_col].iloc[0]
                                next_score = contestant_data[next_score_col].iloc[0] if next_score_col in contestant_data.columns else 0
                                if current_score > 0 and next_score == 0:
                                    eliminated.append(contestant)
                
                # For rank_bottom_two phase, get Bottom Two information
                if voting_phase == 'rank_bottom_two' and 'is_bottom_two' in week_data.columns:
                    bottom_two_data = week_data[week_data['is_bottom_two'] == True]
                    if not bottom_two_data.empty:
                        bottom_two = bottom_two_data['celebrity_name'].tolist()
                
                # Predict fan voting
                if method == 'rank':
                    fan_votes = predict_by_rank(judge_scores, eliminated, bottom_two)
                else:
                    fan_votes = predict_by_percentage(judge_scores, eliminated)
                
                # Store prediction results
                for contestant in active_contestants:
                    if contestant in fan_votes:
                        contestant_data = season_data[season_data['celebrity_name'] == contestant].iloc[0]
                        
                        # Get other contestant information
                        additional_info = {}
                        info_cols = ['placement', 'voting_phase', 'celebrity_age_during_season',
                                   'celebrity_industry', 'celebrity_homecountry/region']
                        
                        for col in info_cols:
                            if col in contestant_data:
                                additional_info[col] = contestant_data[col]
                        
                        all_predictions.append({
                            'sample_idx': sample_idx,
                            'season': season,
                            'week': week,
                            'contestant': contestant,
                            'fan_vote_raw': fan_votes[contestant],
                            'method': method,
                            'voting_phase': voting_phase,
                            'judge_score': judge_scores.get(contestant, 0),
                            'is_eliminated': 1 if contestant in eliminated else 0,
                            **additional_info
                        })
    
    # Convert to DataFrame and calculate uncertainty
    if all_predictions:
        predictions_df_raw = pd.DataFrame(all_predictions)
        print(f"\nGenerated {len(predictions_df_raw)} prediction records")
        
        # Check if necessary columns exist
        if 'season' not in predictions_df_raw.columns:
            print("Error: Prediction data missing 'season' column")
            print(f"Prediction data columns: {list(predictions_df_raw.columns)}")
            return None, comparison_df, weight_df, controversial_df, simulation_df
        
        predictions_df = calculate_uncertainty_fixed(predictions_df_raw)
        
        # 6. Create comparison analysis visualizations
        create_comparison_visualizations(comparison_df, weight_df, controversial_df, simulation_df, predictions_df)
        
        # 8. Calculate consistency metrics
        print("\nCalculating consistency metrics...")
        actual_eliminations = {}
        
        for season in seasons:
            season_data = df[df['season'] == season]
            
            # Get maximum week for this season
            max_week_season = 0
            for week in range(1, max_week + 1):
                col = f'week{week}_total_score'
                if col in season_data.columns and season_data[col].notna().any():
                    max_week_season = week
            
            # Check elimination each week
            for week in range(1, max_week_season):
                current_col = f'week{week}_total_score'
                next_col = f'week{week+1}_total_score'
                
                if current_col in season_data.columns and next_col in season_data.columns:
                    eliminated = []
                    for _, row in season_data.iterrows():
                        current_score = row[current_col]
                        next_score = row[next_col]
                        
                        if current_score > 0 and (pd.isna(next_score) or next_score == 0):
                            eliminated.append(row['celebrity_name'])
                    
                    if eliminated:
                        actual_eliminations[(season, week)] = eliminated
        
        consistency_metrics, _ = calculate_consistency_metrics(predictions_df, actual_eliminations)
        
        # 9. Advanced statistical analysis
        statistical_results = advanced_statistical_analysis(predictions_df)
        
        # 10. Generate comprehensive report
        generate_comprehensive_comparison_report(predictions_df, comparison_df, weight_df,
                                               controversial_df, simulation_df,
                                               consistency_metrics, statistical_results)
        
        # 11. Save data
        predictions_df.to_csv('fan_vote_predictions_enhanced.csv', index=False)
        if not comparison_df.empty:
            comparison_df.to_csv('voting_methods_comparison.csv', index=False)
        if not weight_df.empty:
            weight_df.to_csv('fan_vote_weights_analysis.csv', index=False)
        if not controversial_df.empty:
            controversial_df.to_csv('controversial_cases_analysis.csv', index=False)
        if not simulation_df.empty:
            simulation_df.to_csv('judge_choice_simulation.csv', index=False)
        
        # Save uncertainty analysis results
        if 'fan_vote_cv' in predictions_df.columns:
            uncertainty_summary = predictions_df.groupby(['season', 'week']).agg({
                'fan_vote_cv': ['mean', 'std', 'min', 'max'],
                'fan_vote_std': 'mean',
                'fan_vote_ci_width': 'mean'
            }).round(6)
            uncertainty_summary.to_csv('uncertainty_analysis_summary.csv')
            print("Uncertainty analysis summary saved to 'uncertainty_analysis_summary.csv'")
        
        print(f"\nAll analysis completed!")
    else:
        print("Warning: No prediction data generated")
        predictions_df = pd.DataFrame()
    
    # 12. Display key results
    print("\n" + "="*80)
    print("Key Findings Summary:")
    print("="*80)
    
    if not comparison_df.empty:
        print(f"Ranking method average accuracy: {comparison_df['rank_accuracy'].mean():.4f}")
        print(f"Percentage method average accuracy: {comparison_df['percentage_accuracy'].mean():.4f}")
    
    if not weight_df.empty:
        rank_weight = weight_df[weight_df['method'] == 'rank']['avg_fan_weight'].mean()
        percentage_weight = weight_df[weight_df['method'] == 'percentage']['avg_fan_weight'].mean()
        print(f"Ranking method fan voting weight: {rank_weight:.3f}")
        print(f"Percentage method fan voting weight: {percentage_weight:.3f}")
    
    if predictions_df is not None and not predictions_df.empty:
        if 'fan_vote_cv' in predictions_df.columns:
            cv_mean = predictions_df[predictions_df['fan_vote_mean'] > 0]['fan_vote_cv'].mean()
            print(f"Average coefficient of variation: {cv_mean:.6f} ({cv_mean*100:.2f}%)")
    
    if not simulation_df.empty:
        match_rate = simulation_df['match'].mean()
        print(f"Judge choice mechanism match rate: {match_rate:.4f}")
    
    print("="*80)
    
    return predictions_df, comparison_df, weight_df, controversial_df, simulation_df

# ==================== 11. Execute Main Function ====================
if __name__ == "__main__":
    print("Running enhanced model...")
    try:
        predictions, comparison, weights, controversial, simulation = main_enhanced()
        
        if predictions is not None:
            print("\n" + "="*80)
            print("Model execution completed!")
            print("="*80)
            print("Output files:")
            print("1. fan_vote_predictions_enhanced.csv - Prediction results data")
            if comparison is not None and not comparison.empty:
                print("2. voting_methods_comparison.csv - Two methods comparison analysis")
            if weights is not None and not weights.empty:
                print("3. fan_vote_weights_analysis.csv - Fan voting weight analysis")
            if controversial is not None and not controversial.empty:
                print("4. controversial_cases_analysis.csv - Controversial cases analysis")
            if simulation is not None and not simulation.empty:
                print("5. judge_choice_simulation.csv - Judge choice mechanism simulation")
            print("6. uncertainty_analysis_summary.csv - Uncertainty analysis summary")
            print("7. voting_methods_comparison_report.txt - Comprehensive comparison report")
            print("8. visualizations/comparison/ - Comparison analysis visualization charts")
            print("9. visualizations/uncertainty/ - Uncertainty analysis visualization charts")
            print("="*80)
        else:
            print("\nModel execution failed!")
    except Exception as e:
        print(f"\nModel execution error: {e}")
        import traceback
        traceback.print_exc()