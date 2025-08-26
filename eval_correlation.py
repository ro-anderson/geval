"""
Evaluation correlation script for G-EVAL results.

Configuration:
- Input file: Set EVAL_INPUT_FP environment variable or modify config.eval_input_fp in settings.py
- Dimension: Set EVALUATION_DIMENSION environment variable or modify config.evaluation_dimension in settings.py
- Available dimensions: consistency, coherence, fluency, relevance

Usage:
    python eval_correlation.py

The script uses the configuration from settings.py (no command line arguments).
"""

from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from settings import config


def calculate_correlation(pred_score, human_score, result):
    """Calculate correlation coefficients between predicted and human scores."""
    assert len(pred_score) == len(human_score)

    if len(result) == 0:
        result = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    result['pearson'] += pearsonr(pred_score, human_score)[0]
    result['spearman'] += spearmanr(pred_score, human_score)[0]
    result['kendalltau'] += kendalltau(pred_score, human_score)[0]

    return result


def print_correlations(result, n):
    """Print correlation results in a formatted table."""
    table = PrettyTable(['Pearson', 'Spearman', 'Kendall'])
    if n == 0:
        n = 1
    table.add_row([
        round(result['pearson'] / n, 4), 
        round(result['spearman'] / n, 4), 
        round(result['kendalltau'] / n, 4)
    ])
    print(table)


def parse_output(output):
    """
    Parse numeric score from text output.
    This function is kept for backward compatibility with old data formats.
    """
    if isinstance(output, (int, float)):
        # If it's already a number, return it directly
        return float(output)
    
    # Try to parse string output
    matched = re.search(r"^ ?([\d\.]+)", str(output))
    if matched:
        try:
            score = float(matched.group(1))
        except:
            score = 0
    else:
        score = 0
    return score


def get_dimension_from_filepath(filepath):
    """Extract evaluation dimension from file path."""
    if 'coh' in filepath:
        return 'coherence'
    elif 'con' in filepath:
        return 'consistency'  
    elif 'flu' in filepath:
        return 'fluency'
    elif 'rel' in filepath:
        return 'relevance'
    else:
        return 'consistency'  # default


def create_correlation_visualizations(df, dimension, n_samples):
    """Create and save correlation visualizations using seaborn."""
    
    # Set style for better-looking plots
    sns.set_style("whitegrid")
    plt.style.use('default')
    
    # Calculate correlations for titles
    pearson_corr = df['predicted_scores'].corr(df['human_scores'], method='pearson')
    spearman_corr = df['predicted_scores'].corr(df['human_scores'], method='spearman')
    kendall_corr = df['predicted_scores'].corr(df['human_scores'], method='kendall')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'G-EVAL Correlation Analysis - {dimension.title()} Dimension\n'
                 f'Dataset: {n_samples} document-summary pairs', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot with regression line
    sns.scatterplot(data=df, x='human_scores', y='predicted_scores', 
                   alpha=0.7, s=60, color='steelblue', ax=axes[0,0])
    sns.regplot(data=df, x='human_scores', y='predicted_scores', 
               scatter=False, color='red', ax=axes[0,0])
    axes[0,0].set_title(f'Predicted vs Human Scores\nPearson r = {pearson_corr:.3f}', 
                       fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Human Scores', fontsize=11)
    axes[0,0].set_ylabel('G-EVAL Predicted Scores', fontsize=11)
    axes[0,0].grid(True, alpha=0.3)
    
    # Add diagonal reference line
    min_val = min(df['human_scores'].min(), df['predicted_scores'].min())
    max_val = max(df['human_scores'].max(), df['predicted_scores'].max())
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect correlation')
    axes[0,0].legend()
    
    # 2. Distribution comparison
    axes[0,1].hist(df['human_scores'], alpha=0.7, label='Human Scores', 
                  bins=20, color='lightcoral', density=True)
    axes[0,1].hist(df['predicted_scores'], alpha=0.7, label='G-EVAL Scores', 
                  bins=20, color='steelblue', density=True)
    axes[0,1].set_title('Score Distributions Comparison', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Score Value', fontsize=11)
    axes[0,1].set_ylabel('Density', fontsize=11)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Residuals plot
    residuals = df['predicted_scores'] - df['human_scores']
    sns.scatterplot(x=df['human_scores'], y=residuals, alpha=0.7, 
                   color='green', s=60, ax=axes[1,0])
    axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1,0].set_title('Residuals Plot\n(Predicted - Human)', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Human Scores', fontsize=11)
    axes[1,0].set_ylabel('Residuals', fontsize=11)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Correlation summary table as plot
    corr_data = {
        'Metric': ['Pearson', 'Spearman', 'Kendall'],
        'Correlation': [pearson_corr, spearman_corr, kendall_corr],
        'Interpretation': [
            'Linear relationship',
            'Monotonic relationship', 
            'Rank concordance'
        ]
    }
    corr_df = pd.DataFrame(corr_data)
    
    axes[1,1].axis('tight')
    axes[1,1].axis('off')
    table = axes[1,1].table(cellText=[[f'{row.Metric}', f'{row.Correlation:.4f}', f'{row.Interpretation}'] 
                                     for _, row in corr_df.iterrows()],
                           colLabels=['Metric', 'Value', 'Measures'],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.25, 0.25, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1,1].set_title('Correlation Summary', fontsize=12, fontweight='bold')
    
    # Style the table
    for i in range(len(corr_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f'correlation_analysis_{dimension}_{n_samples}samples.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Correlation visualization saved as: {output_filename}")
    
    # Show the plot
    plt.show()
    
    return pearson_corr, spearman_corr, kendall_corr


if __name__ == '__main__':
    # Use config values for evaluation
    input_fp = config.eval_input_fp
    dimension = config.evaluation_dimension
    
    # Auto-detect dimension from filepath if config dimension doesn't seem to match
    auto_detected_dimension = get_dimension_from_filepath(input_fp)
    if auto_detected_dimension != dimension:
        print(f"Note: Config dimension '{dimension}' differs from auto-detected '{auto_detected_dimension}' from filepath")
        print(f"Using config dimension: {dimension}")
    
    print(f"Input file: {input_fp}")
    print(f"Evaluation dimension: {dimension}")

    try:
        jobj = json.load(open(input_fp))
    except FileNotFoundError:
        print(f"Error: File {input_fp} not found.")
        print(f"Make sure you've run gpt4_eval.py first to generate the results.")
        print(f"To use a different file, set EVAL_INPUT_FP in your environment or modify settings.py")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {input_fp}")
        exit(1)

    # Prepare data for pandas DataFrame
    data_rows = []

    print("Calculating correlation for G-Eval")
    print("Note: Each item represents a unique document-summary pair")
    
    for item in jobj:
        doc_id = item["doc_id"]

        # Handle both old and new data formats
        all_responses = item["all_responses"]
        
        if all_responses:
            # New format: all_responses contains pre-calculated scores (floats)
            if isinstance(all_responses[0], (int, float)):
                all_scores = [float(x) for x in all_responses]
            else:
                # Old format: all_responses contains text that needs parsing
                all_scores = [parse_output(x) for x in all_responses]
            
            if all_scores:  # Only proceed if we have valid scores
                score = sum(all_scores) / len(all_scores)
                
                # Check if dimension exists in scores
                if 'scores' in item and dimension in item['scores']:
                    data_rows.append({
                        'doc_id': doc_id,
                        'predicted_scores': score,
                        'human_scores': item['scores'][dimension],
                        'system_id': item.get('system_id', 'unknown'),
                        'score_variance': np.var(all_scores) if len(all_scores) > 1 else 0.0
                    })
                else:
                    print(f"Warning: Dimension '{dimension}' not found in item {doc_id}")
                    print(f"Available dimensions: {list(item.get('scores', {}).keys())}")

    # Create pandas DataFrame
    df = pd.DataFrame(data_rows)
    
    print(f'Total items processed: {len(jobj)}')
    print(f'Valid score pairs: {len(df)}')

    if len(df) == 0:
        print("Error: No valid score pairs found for correlation calculation.")
        print("Check that:")
        print("1. The input file contains the correct data format")
        print("2. The dimension name matches what's in the data")
        print("3. Both predicted and human scores are available")
        exit(1)

    if len(df) < 2:
        print("Error: Need at least 2 score pairs for correlation calculation.")
        exit(1)

    # Check for variance in scores
    if df['predicted_scores'].nunique() <= 1:
        print("Error: No variance in predicted scores - all scores are the same.")
        exit(1)
        
    if df['human_scores'].nunique() <= 1:
        print("Error: No variance in human scores - all scores are the same.")
        exit(1)

    print(f"Calculating correlation across {len(df)} document-summary pairs:")
    
    # Calculate correlations using pandas
    pearson_corr = df['predicted_scores'].corr(df['human_scores'], method='pearson')
    spearman_corr = df['predicted_scores'].corr(df['human_scores'], method='spearman')
    kendall_corr = df['predicted_scores'].corr(df['human_scores'], method='kendall')
    
    # Display results in table format
    table = PrettyTable(['Pearson', 'Spearman', 'Kendall'])
    table.add_row([
        round(pearson_corr, 4), 
        round(spearman_corr, 4), 
        round(kendall_corr, 4)
    ])
    print(table)
    
    # Print detailed statistics using pandas
    print(f"\n📊 Detailed Score Statistics:")
    print(f"Predicted scores - Mean: {df['predicted_scores'].mean():.3f}, "
          f"Std: {df['predicted_scores'].std():.3f}, "
          f"Range: [{df['predicted_scores'].min():.3f}, {df['predicted_scores'].max():.3f}]")
    print(f"Human scores - Mean: {df['human_scores'].mean():.3f}, "
          f"Std: {df['human_scores'].std():.3f}, "
          f"Range: [{df['human_scores'].min():.3f}, {df['human_scores'].max():.3f}]")
    
    # Show score distribution
    print(f"\n📈 Score Distributions:")
    print("Human scores distribution:")
    print(df['human_scores'].value_counts().sort_index().head(10))
    
    # Create and display visualizations
    print(f"\n🎨 Creating correlation visualizations...")
    create_correlation_visualizations(df, dimension, len(df))
    
    # Save DataFrame for further analysis
    csv_filename = f'correlation_data_{dimension}_{len(df)}samples.csv'
    df.to_csv(csv_filename, index=False)
    print(f"📄 Correlation data saved as: {csv_filename}")
