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

    pred_scores, human_scores = {}, {}

    print("Calculating correlation for G-Eval")
    
    for item in jobj:
        doc_id = item["doc_id"]
        if doc_id not in pred_scores:
            pred_scores[doc_id] = []
            human_scores[doc_id] = []

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
                pred_scores[doc_id].append(score)
                
                # Check if dimension exists in scores
                if 'scores' in item and dimension in item['scores']:
                    human_scores[doc_id].append(item['scores'][dimension])
                else:
                    print(f"Warning: Dimension '{dimension}' not found in item {doc_id}")
                    print(f"Available dimensions: {list(item.get('scores', {}).keys())}")

    print(f'Documents with predicted scores: {len(pred_scores)}')
    print(f'Documents with human scores: {len(human_scores)}')

    # Check if we have matching data
    valid_docs = []
    for doc_id in pred_scores:
        if doc_id in human_scores and len(pred_scores[doc_id]) > 0 and len(human_scores[doc_id]) > 0:
            valid_docs.append(doc_id)

    print(f'Valid documents for correlation: {len(valid_docs)}')

    if len(valid_docs) == 0:
        print("Error: No valid document pairs found for correlation calculation.")
        print("Check that:")
        print("1. The input file contains the correct data format")
        print("2. The dimension name matches what's in the data")
        print("3. Both predicted and human scores are available")
        exit(1)

    results = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    d_ctr = 0
    
    for doc_id in valid_docs:
        pred_scores_doc = pred_scores[doc_id]
        human_scores_doc = human_scores[doc_id]
        
        # Skip documents with no variance in scores
        if (len(set(human_scores_doc)) <= 1) or (len(set(pred_scores_doc)) <= 1):
            continue

        results = calculate_correlation(pred_scores_doc, human_scores_doc, results)
        d_ctr += 1

    if d_ctr == 0:
        print("Error: No documents with sufficient score variance for correlation calculation.")
        exit(1)

    print(f"Correlation calculated over {d_ctr} documents:")
    print_correlations(results, n=d_ctr)
    
    # Print additional statistics
    all_pred_scores = [score for doc_scores in pred_scores.values() for score in doc_scores]
    all_human_scores = [score for doc_scores in human_scores.values() for score in doc_scores]
    
    if all_pred_scores and all_human_scores:
        print(f"\nScore statistics:")
        print(f"Predicted scores - Mean: {sum(all_pred_scores)/len(all_pred_scores):.3f}, "
              f"Range: [{min(all_pred_scores):.3f}, {max(all_pred_scores):.3f}]")
        print(f"Human scores - Mean: {sum(all_human_scores)/len(all_human_scores):.3f}, "
              f"Range: [{min(all_human_scores):.3f}, {max(all_human_scores):.3f}]")
