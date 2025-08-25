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

    pred_scores = []
    human_scores = []

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
                    pred_scores.append(score)
                    human_scores.append(item['scores'][dimension])
                else:
                    print(f"Warning: Dimension '{dimension}' not found in item {doc_id}")
                    print(f"Available dimensions: {list(item.get('scores', {}).keys())}")

    print(f'Total items processed: {len(jobj)}')
    print(f'Valid score pairs: {len(pred_scores)}')

    if len(pred_scores) == 0:
        print("Error: No valid score pairs found for correlation calculation.")
        print("Check that:")
        print("1. The input file contains the correct data format")
        print("2. The dimension name matches what's in the data")
        print("3. Both predicted and human scores are available")
        exit(1)

    if len(pred_scores) < 2:
        print("Error: Need at least 2 score pairs for correlation calculation.")
        exit(1)

    # Check for variance in scores
    if len(set(pred_scores)) <= 1:
        print("Error: No variance in predicted scores - all scores are the same.")
        exit(1)
        
    if len(set(human_scores)) <= 1:
        print("Error: No variance in human scores - all scores are the same.")
        exit(1)

    print(f"Calculating correlation across {len(pred_scores)} document-summary pairs:")
    
    # Calculate correlation directly between the two lists
    pearson_corr, _ = pearsonr(pred_scores, human_scores)
    spearman_corr, _ = spearmanr(pred_scores, human_scores)
    kendall_corr, _ = kendalltau(pred_scores, human_scores)
    
    # Display results in table format
    table = PrettyTable(['Pearson', 'Spearman', 'Kendall'])
    table.add_row([
        round(pearson_corr, 4), 
        round(spearman_corr, 4), 
        round(kendall_corr, 4)
    ])
    print(table)
    
    # Print additional statistics
    if pred_scores and human_scores:
        print(f"\nScore statistics:")
        print(f"Predicted scores - Mean: {sum(pred_scores)/len(pred_scores):.3f}, "
              f"Range: [{min(pred_scores):.3f}, {max(pred_scores):.3f}]")
        print(f"Human scores - Mean: {sum(human_scores)/len(human_scores):.3f}, "
              f"Range: [{min(human_scores):.3f}, {max(human_scores):.3f}]")
