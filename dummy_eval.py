"""
Dummy evaluation script for baseline comparison with G-EVAL.

This script provides the simplest possible evaluation approach:
- Single API call per document-summary pair
- Direct score extraction from structured output
- No G-EVAL algorithmic processing (no logprobs, no probability weighting)
- Used to demonstrate the effectiveness of G-EVAL's sophisticated approach

The output format is compatible with eval_correlation.py for direct comparison.
"""

import json
import tqdm
import time
from openai import OpenAI
from pydantic import BaseModel
from settings import config
from utils import extract_token_usage, accumulate_token_usage, print_token_usage_summary


class EvaluationScore(BaseModel):
    """Structured output for evaluation scores."""
    score: float

if __name__ == '__main__':
    # Initialize OpenAI client with API key from config
    client = OpenAI(api_key=config.openai_api_key)

    summeval = json.load(open(config.summeval_fp))
    prompt = open(config.prompt_fp).read()

    ct, ignore = 0, 0
    token_totals = {
        'total_tokens': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0
    }

    new_json = []
    for instance in tqdm.tqdm(summeval):
        source = instance['source']
        system_output = instance['system_output']
        cur_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)
        instance['prompt'] = cur_prompt
        while True:
            try:
                # Using Chat Completions API with Structured Outputs - single call, single response
                _response = client.chat.completions.parse(
                    model=config.model,
                    messages=[{"role": "system", "content": cur_prompt}],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                    response_format=EvaluationScore,
                    logprobs=False  # Don't need logprobs for dummy evaluation
                    # Note: No n parameter - single response only for simplest baseline
                )
                time.sleep(config.sleep_time)
                
                # Track token usage for cost analysis
                usage = extract_token_usage(_response)
                instance['total_usage_tokens'] = usage['total_tokens']
                instance['prompt_tokens'] = usage['prompt_tokens']
                instance['completion_tokens'] = usage['completion_tokens']
                
                # Accumulate totals for overall cost tracking
                token_totals = accumulate_token_usage(token_totals, usage)
                
                # Extract single direct score from structured output (simplest baseline approach)
                choice = _response.choices[0]  # Only one choice since n=1 by default
                
                if choice.message.parsed:
                    # Get the direct score from structured output
                    direct_score = choice.message.parsed.score
                    
                    # Validate score is in expected range
                    min_score = 1
                    max_score = 5
                    if min_score <= direct_score <= max_score:
                        final_score = direct_score
                    else:
                        # If score is out of range, clamp it to valid range
                        final_score = max(min_score, min(max_score, direct_score))
                        print(f"Warning: Score {direct_score} out of range [{min_score}, {max_score}], clamped to {final_score}")
                else:
                    # If parsing failed, raise an error
                    raise ValueError(f"Failed to parse structured output")

                # Store as single-item list for compatibility with eval_correlation.py
                instance['all_responses'] = [final_score]
                new_json.append(instance)
                ct += 1
                break
            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(config.rate_limit_sleep)
                else:
                    ignore += 1
                    print('ignored', ignore)
                    break

    # Print comprehensive token usage summary
    print_token_usage_summary(token_totals, ct, ignore)
    
    # Save to a different file to distinguish from G-EVAL results
    dummy_save_fp = config.save_fp.replace('.json', '_dummy.json')
    with open(dummy_save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
    
    print(f"\n🎯 Dummy (baseline) evaluation completed!")
    print(f"📁 Results saved to: {dummy_save_fp}")
    print(f"💡 This baseline uses single API calls with direct score extraction")
    print(f"🔬 Compare with G-EVAL results using eval_correlation.py")
