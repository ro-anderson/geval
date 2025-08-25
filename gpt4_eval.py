import json
import tqdm
import time
from openai import OpenAI
from pydantic import BaseModel
from settings import config
import math
from utils import extract_token_usage, accumulate_token_usage, print_token_usage_summary, count_positive_integers_in_range


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
                # Using Chat Completions API with Structured Outputs - single call with multiple responses
                _response = client.chat.completions.parse(
                    model=config.model,
                    messages=[{"role": "system", "content": cur_prompt}],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    top_p=config.top_p,
                    top_logprobs=5,
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                    response_format=EvaluationScore,
                    logprobs=True,
                    n=config.n_responses  # Generate multiple responses in single call (original behavior)
                )
                time.sleep(config.sleep_time)
                
                # Track token usage for cost analysis
                usage = extract_token_usage(_response)
                instance['total_usage_tokens'] = usage['total_tokens']
                instance['prompt_tokens'] = usage['prompt_tokens']
                instance['completion_tokens'] = usage['completion_tokens']
                
                # Accumulate totals for overall cost tracking
                token_totals = accumulate_token_usage(token_totals, usage)
                
                # Accessing log probabilities from the response
                choices_final_scores = []  # Initialize outside the choice loop
                
                for choice in tqdm.tqdm(_response.choices, desc="Processing choices"):
                    linear_probs_sum = 0.0
                    weighted_score_sum = 0.0
                    
                    if choice.logprobs and choice.logprobs.content:

                        # lets take the token at position 3 as the score token - we assume that the score is the 4th token as we are using structured outputs
                        score_token_position=3
                        min_score = 1
                        max_score = 5
                        score_token_info = choice.logprobs.content[score_token_position]

                        for top_logprob_tokens in tqdm.tqdm(score_token_info.top_logprobs, desc="Processing top logprobs"):
                            # the score in an string
                            score_token_score = top_logprob_tokens.token

                            # if not a number
                            if not score_token_score.isdecimal():
                                continue
                            score = int(score_token_score)
                            
                            # if score value not in scale - TODO: the min and max score should be defined in the prompt / metric
                            if not min_score <= score <= max_score:
                                continue
                            prob = math.exp(top_logprob_tokens.logprob)
                            linear_probs_sum += prob
                            weighted_score_sum += score * prob
                        
                        # Calculate one final score per choice (after processing all top_logprobs for this choice)
                        if linear_probs_sum > 0:
                            # use without normalization when we inteend to run the eval_correlation.py
                            choice_final_score = weighted_score_sum / linear_probs_sum # / (count_positive_integers_in_range(min_score, max_score))

                            # use with normalization when we inteend to generate the score in (0,1) 
                            # choice_final_score = weighted_score_sum / linear_probs_sum / (count_positive_integers_in_range(min_score, max_score))
                            choices_final_scores.append(choice_final_score)
                        else:
                            # If no valid tokens were found, we must stop - G-EVAL requires valid score tokens
                            raise ValueError(f"No valid score tokens found for choice {len(choices_final_scores)}. "
                                           f"G-EVAL requires valid numerical score tokens in range [{min_score}, {max_score}] "
                                           f"at token position {score_token_position}.")

                    else:
                        # If no logprobs available, we must stop - G-EVAL requires logprobs
                        raise ValueError(f"No logprobs available for choice {len(choices_final_scores)}. "
                                       f"G-EVAL requires logprobs=True and top_logprobs > 0 in the API call. "
                                       f"Current config: n_responses={config.n_responses}, model={config.model}")

                instance['all_responses'] = choices_final_scores
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
    
    with open(config.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
