"""
Generate shuffled sample of summeval dataset.

This script creates a randomly shuffled sample of the summeval.json dataset.
The sample size is controlled by the -n parameter, and reproducibility is 
ensured by the --seed parameter.

Usage:
    python generate_shuffle_n_summeval.py -n 100 --seed 42
    python generate_shuffle_n_summeval.py -n 50  # Random seed
    python generate_shuffle_n_summeval.py -n 1000 --seed 123

Output: 
    ./data/summeval_shuffle_n.json (where n is the actual number specified)
"""

import argparse
import json
import random
import os
from pathlib import Path


def load_summeval_data(file_path):
    """Load the original summeval dataset."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        print("Make sure the summeval.json file exists in the data/ directory.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
        exit(1)


def generate_shuffled_sample(data, n, seed=None):
    """
    Generate a shuffled sample of the data.
    
    Args:
        data: List of data items
        n: Number of items to sample
        seed: Random seed for reproducibility (optional)
    
    Returns:
        List of shuffled sample items
    """
    if n > len(data):
        print(f"Warning: Requested sample size ({n}) is larger than dataset size ({len(data)})")
        print(f"Using full dataset size: {len(data)}")
        n = len(data)
    
    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    else:
        print("Using random seed (not specified)")
    
    # Create a copy and shuffle
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    # Take first n items
    sample = data_copy[:n]
    print(f"Generated sample of {len(sample)} items")
    
    return sample


def save_sample(sample, output_path):
    """Save the sample to a JSON file."""
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        print(f"Sample saved to: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate shuffled sample of summeval dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_shuffle_n_summeval.py -n 100 --seed 42
  python generate_shuffle_n_summeval.py -n 50
  python generate_shuffle_n_summeval.py -n 1000 --seed 123

The output file will be saved as ./data/summeval_shuffle_n.json where n is your specified sample size.
        """
    )
    
    parser.add_argument(
        '-n', 
        type=int, 
        required=True,
        help='Number of samples to generate from the dataset'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible shuffling (optional)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/summeval.json',
        help='Path to input summeval.json file (default: data/summeval.json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for generated file (default: data)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.n <= 0:
        print("Error: Sample size (-n) must be a positive integer")
        exit(1)
    
    # Generate output filename
    output_filename = f"summeval_shuffle_{args.n}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"Generating shuffled sample...")
    print(f"Input file: {args.input}")
    print(f"Sample size: {args.n}")
    print(f"Output file: {output_path}")
    print("-" * 50)
    
    # Load data
    data = load_summeval_data(args.input)
    
    # Generate sample
    sample = generate_shuffled_sample(data, args.n, args.seed)
    
    # Save sample
    save_sample(sample, output_path)
    
    print("-" * 50)
    print("✅ Successfully generated shuffled sample!")
    
    # Show some statistics
    if len(sample) > 0:
        # Count unique doc_ids to verify diversity
        doc_ids = set(item.get('doc_id', '') for item in sample)
        print(f"📊 Sample statistics:")
        print(f"   • Total items: {len(sample)}")
        print(f"   • Unique document IDs: {len(doc_ids)}")
        
        if args.seed is not None:
            print(f"   • Reproducible with seed: {args.seed}")
        
        # Show first item's doc_id as verification
        if 'doc_id' in sample[0]:
            print(f"   • First item doc_id: {sample[0]['doc_id']}")


if __name__ == '__main__':
    main()
