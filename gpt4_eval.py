import json
import tqdm
from openai import OpenAI
from settings import config
from geval import GEval
from utils import print_token_usage_summary


if __name__ == '__main__':
    # Initialize OpenAI client with API key from config
    client = OpenAI(api_key=config.openai_api_key)

    # Load dataset
    summeval = json.load(open(config.summeval_fp))

    # Initialize G-Eval with prompt configuration and model settings
    geval = GEval.from_json_config(
        config_path=config.prompt_fp,
        client=client,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        n_responses=config.n_responses,
        sleep_time=config.sleep_time,
        rate_limit_sleep=config.rate_limit_sleep
    )

    ct, ignore = 0, 0
    new_json = []
    
    for instance in tqdm.tqdm(summeval, desc=f"Evaluating instances for criteria {geval.name}"):
        try:
            actual_output = instance.get('system_output', '')  # The summary being evaluated
            expected_output = instance.get('source', '')       # The source document
            
            # Use GEval to evaluate the instance
            # For metrics that need comparison (consistency, coherence, relevance): pass both
            # For metrics that don't (fluency): pass only actual_output by setting expected_output=None
            if geval.name.lower() in ['fluency']:
                result = geval.evaluate(actual_output=actual_output)
            else:
                result = geval.evaluate(actual_output=actual_output, expected_output=expected_output)
            
            # Merge original instance with results
            evaluated_instance = {**instance, **result}
            new_json.append(evaluated_instance)
            ct += 1
            
        except Exception as e:
            print(f"Error evaluating instance: {e}")
            ignore += 1
            continue

    # Print comprehensive token usage summary
    token_totals = geval.get_token_usage_summary()
    print_token_usage_summary(token_totals, ct, ignore)
    
    # Save results
    with open(config.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
