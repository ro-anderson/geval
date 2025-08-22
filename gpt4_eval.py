import json
import tqdm
import time
from openai import OpenAI
from pydantic import BaseModel
from settings import config


class EvaluationScore(BaseModel):
    """Structured output for evaluation scores."""
    score: float

if __name__ == '__main__':
    # Initialize OpenAI client with API key from config
    client = OpenAI(api_key=config.openai_api_key)

    summeval = json.load(open(config.summeval_fp))
    prompt = open(config.prompt_fp).read()

    ct, ignore = 0, 0

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
                    frequency_penalty=config.frequency_penalty,
                    presence_penalty=config.presence_penalty,
                    response_format=EvaluationScore,
                    n=config.n_responses  # Generate multiple responses in single call (original behavior)
                )
                time.sleep(config.sleep_time)

                # Extract all structured responses (same as original code logic)
                all_responses = [_response.choices[i].message.parsed.score for i in
                                 range(len(_response.choices))]

                instance['all_responses'] = all_responses
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

    print('ignored total', ignore)
    with open(config.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
