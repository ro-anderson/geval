# utils.py - Token usage tracking utilities
from typing import Dict, Any

def extract_token_usage(response: Any) -> Dict[str, int]:
    """
    Extract token usage information from an OpenAI API response.
    Returns a dictionary with token counts or zeros if not available.
    """
    if hasattr(response, 'usage') and response.usage:
        return {
            'total_tokens': response.usage.total_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens
        }
    else:
        return {
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0
        }


def accumulate_token_usage(
    current_totals: Dict[str, int], 
    new_usage: Dict[str, int]
) -> Dict[str, int]:
    """
    Accumulate token usage from a new response into running totals.
    """
    return {
        'total_tokens': current_totals.get('total_tokens', 0) + new_usage.get('total_tokens', 0),
        'prompt_tokens': current_totals.get('prompt_tokens', 0) + new_usage.get('prompt_tokens', 0),
        'completion_tokens': current_totals.get('completion_tokens', 0) + new_usage.get('completion_tokens', 0)
    }


def print_token_usage_summary(
    token_totals: Dict[str, int], 
    instances_processed: int, 
    instances_ignored: int = 0
) -> None:
    """
    Print a comprehensive summary of token usage and costs.
    """
    print(f'Ignored total: {instances_ignored}')
    print(f'Total instances processed: {instances_processed}')
    print(f'Total tokens used: {token_totals["total_tokens"]:,}')
    print(f'Total prompt tokens: {token_totals["prompt_tokens"]:,}')
    print(f'Total completion tokens: {token_totals["completion_tokens"]:,}')
    
    if instances_processed > 0:
        avg_tokens = token_totals["total_tokens"] / instances_processed
        print(f'Average tokens per instance: {avg_tokens:.1f}')
    else:
        print('Average tokens per instance: 0.0')
