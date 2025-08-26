import json
import math
import time
import tqdm
from functools import cached_property
from typing import Dict, List, Optional, Union
from openai import OpenAI
from pydantic import BaseModel
from utils import extract_token_usage, accumulate_token_usage
from metrics.llm_judges.template import G_EVAL_COT_TEMPLATE, G_EVAL_QUERY_TEMPLATE


class EvaluationScore(BaseModel):
    """Structured output for evaluation scores."""
    score: float


class GEval:
    """
    G-Eval metric implementation for evaluating text quality using logprobs.
    
    This class encapsulates the G-Eval methodology which uses weighted scoring
    based on log probabilities to produce more robust evaluation scores.
    """
    
    def __init__(
        self,
        name: str,
        task_introduction: str,
        evaluation_criteria: str,
        min_score: int = 1,
        max_score: int = 5,
        client: Optional[OpenAI] = None,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 2.0,
        max_tokens: int = 2500,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        n_responses: int = 10,
        sleep_time: float = 0.0,
        rate_limit_sleep: float = 0.0,
        score_token_position: int = 3,
        custom_chain_of_thought: Optional[str] = None
    ):
        """
        Initialize the G-Eval metric.
        
        Args:
            name: Name of the metric (e.g., "Consistency", "Fluency")
            task_introduction: Description of the evaluation task
            evaluation_criteria: Specific criteria for evaluation
            min_score: Minimum score in the evaluation scale
            max_score: Maximum score in the evaluation scale
            client: OpenAI client instance (if None, will be created)
            model: Model name to use for evaluation
            temperature: Temperature for model generation
            max_tokens: Maximum tokens for response
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            n_responses: Number of responses to generate
            sleep_time: Sleep time between API calls
            rate_limit_sleep: Sleep time when rate limited
            score_token_position: Position of score token in structured output (0-indexed)
            custom_chain_of_thought: Optional custom CoT to use instead of generating one
        """
        self.name = name
        self.task_introduction = task_introduction
        self.evaluation_criteria = evaluation_criteria
        self.min_score = min_score
        self.max_score = max_score
        self.score_token_position = score_token_position
        
        # Model configuration
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.n_responses = n_responses
        self.sleep_time = sleep_time
        self.rate_limit_sleep = rate_limit_sleep
        
        # Token usage tracking
        self.token_totals = {
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0
        }
        
        # Set custom CoT if provided
        if custom_chain_of_thought is not None:
            self.set_custom_chain_of_thought(custom_chain_of_thought)
    
    @cached_property
    def llm_chain_of_thought(self) -> str:
        """
        Generate Chain of Thought (CoT) once and cache it for all evaluations.
        This ensures consistency across all n_responses and improves efficiency.
        
        Returns:
            Generated chain of thought reasoning
        """
        if not self.client:
            raise ValueError("OpenAI client is required for CoT generation. Please provide a client instance.")
        
        cot_prompt = G_EVAL_COT_TEMPLATE.format(
            task_introduction=self.task_introduction,
            evaluation_criteria=self.evaluation_criteria,
            min_score=self.min_score,
            max_score=self.max_score
        )
        
        # Generate CoT using a separate API call with lower temperature for consistency
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": cot_prompt}],
            temperature=0.7,  # Lower temperature for more consistent CoT generation
            max_tokens=1000,  # Reasonable limit for CoT
            top_p=1.0
        )
        
        # Track token usage for CoT generation
        cot_usage = extract_token_usage(response)
        self.token_totals = accumulate_token_usage(self.token_totals, cot_usage)
        
        return response.choices[0].message.content.strip()
    
    def set_custom_chain_of_thought(self, custom_cot: str) -> None:
        """
        Set a custom Chain of Thought, bypassing automatic generation.
        This will clear any cached CoT and use the provided one instead.
        
        Args:
            custom_cot: Custom chain of thought reasoning text
        """
        # Clear the cached property if it exists
        if 'llm_chain_of_thought' in self.__dict__:
            del self.__dict__['llm_chain_of_thought']
        
        # Set the custom CoT directly
        self.__dict__['llm_chain_of_thought'] = custom_cot
    
    def clear_chain_of_thought_cache(self) -> None:
        """
        Clear the cached Chain of Thought to force regeneration on next access.
        Useful if you want to regenerate the CoT with different parameters.
        """
        if 'llm_chain_of_thought' in self.__dict__:
            del self.__dict__['llm_chain_of_thought']
    
    def _create_prompt(self, actual_output: str, expected_output: Optional[str] = None) -> str:
        """
        Create evaluation prompt using the cached Chain of Thought.
        
        Args:
            actual_output: The output being evaluated (e.g., summary, response)
            expected_output: Optional reference output (e.g., source document)
            
        Returns:
            Formatted prompt string with CoT reasoning
        """
        # Construct input text based on whether expected_output is provided
        if expected_output:
            input_text = f"Expected Output:\n\n{expected_output}\n\nActual Output:\n\n{actual_output}"
        else:
            input_text = f"Actual Output:\n\n{actual_output}"
        
        # Use the G_EVAL_QUERY_TEMPLATE with cached CoT
        prompt = G_EVAL_QUERY_TEMPLATE.format(
            task_introduction=self.task_introduction,
            evaluation_criteria=self.evaluation_criteria,
            chain_of_thought=self.llm_chain_of_thought,  # This triggers CoT generation once and caches it
            input=input_text
        )
        
        return prompt
    
    def _calculate_weighted_score(self, choice) -> float:
        """
        Calculate weighted score based on log probabilities.
        
        Args:
            choice: OpenAI API response choice object
            
        Returns:
            Weighted score based on log probabilities
        """
        linear_probs_sum = 0.0
        weighted_score_sum = 0.0
        
        if not (choice.logprobs and choice.logprobs.content):
            raise ValueError(f"No logprobs available for choice. "
                           f"G-EVAL requires logprobs=True and top_logprobs > 0 in the API call.")
        
        score_token_info = choice.logprobs.content[self.score_token_position]
        
        for top_logprob_token in tqdm.tqdm(score_token_info.top_logprobs, desc="Processing top logprobs"):
            # the score in an string
            score_token_score = top_logprob_token.token
            
            # if not a number
            if not score_token_score.isdecimal():
                continue
            score = int(score_token_score)
            
            # if score value not in scale
            if not self.min_score <= score <= self.max_score:
                continue
            prob = math.exp(top_logprob_token.logprob)
            linear_probs_sum += prob
            weighted_score_sum += score * prob
        
        if linear_probs_sum == 0:
            raise ValueError(f"No valid score tokens found. "
                           f"G-EVAL requires valid numerical score tokens in range [{self.min_score}, {self.max_score}] "
                           f"at token position {self.score_token_position}.")
        
        # Calculate weighted average (without normalization for correlation analysis)
        return weighted_score_sum / linear_probs_sum
    
    def evaluate(self, actual_output: str, expected_output: Optional[str] = None) -> Dict:
        """
        Evaluate actual output against optional expected output.
        
        Args:
            actual_output: The output being evaluated (required)
            expected_output: Optional reference output for comparison
            
        Returns:
            Dictionary containing evaluation results and metadata
            
        Raises:
            ValueError: If actual_output is empty or only expected_output is provided
        """
        # Validation
        if not actual_output:
            raise ValueError(f"actual_output is required for {self.name} evaluation")
        
        if expected_output is not None and not actual_output:
            raise ValueError(f"Cannot evaluate {self.name} with only expected_output. actual_output is required.")
        
        prompt = self._create_prompt(actual_output, expected_output)
        
        while True:
            try:
                # Make API call with structured outputs
                response = self.client.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    top_logprobs=5,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    response_format=EvaluationScore,
                    logprobs=True,
                    n=self.n_responses
                )
                
                if self.sleep_time > 0:
                    time.sleep(self.sleep_time)
                
                # Track token usage
                usage = extract_token_usage(response)
                self.token_totals = accumulate_token_usage(self.token_totals, usage)
                
                # Calculate scores for all choices
                scores = []
                for choice in tqdm.tqdm(response.choices, desc="Processing choices"):
                    score = self._calculate_weighted_score(choice)
                    scores.append(score)
                
                # Return comprehensive result
                return {
                    'prompt': prompt,
                    'all_responses': scores,
                    'total_usage_tokens': usage['total_tokens'],
                    'prompt_tokens': usage['prompt_tokens'],
                    'completion_tokens': usage['completion_tokens']
                }
                
            except Exception as e:
                if "limit" in str(e):
                    time.sleep(self.rate_limit_sleep)
                else:
                    raise e
    
    def evaluate_dataset(self, dataset: List[Dict], actual_output_key: str = 'system_output', expected_output_key: Optional[str] = 'source') -> List[Dict]:
        """
        Evaluate an entire dataset.
        
        Args:
            dataset: List of dictionaries containing the outputs to evaluate
            actual_output_key: Key for the actual output in each instance (default: 'system_output')
            expected_output_key: Key for the expected output in each instance (default: 'source', None to disable)
            
        Returns:
            Tuple of (results, ignored_count)
        """
        results = []
        ignored_count = 0
        
        for instance in dataset:
            try:
                actual_output = instance.get(actual_output_key, '')
                expected_output = instance.get(expected_output_key, '') if expected_output_key else None
                
                # Skip if expected_output is empty string when expected_output_key is provided
                if expected_output == '':
                    expected_output = None
                
                result = self.evaluate(actual_output=actual_output, expected_output=expected_output)
                
                # Merge original instance with results
                evaluated_instance = {**instance, **result}
                results.append(evaluated_instance)
                
            except Exception as e:
                print(f"Error evaluating instance: {e}")
                ignored_count += 1
                continue
        
        return results, ignored_count
    
    def get_token_usage_summary(self) -> Dict:
        """Get summary of token usage."""
        return self.token_totals.copy()
    
    @classmethod
    def from_json_config(cls, config_path: str, name: Optional[str] = None, **kwargs) -> 'GEval':
        """
        Create GEval instance from JSON configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            name: Optional name for the metric. If not provided, will be derived from filename
            **kwargs: Additional arguments to override config
            
        Returns:
            GEval instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Set name - prioritize provided name, then config name, then derive from filename
        if name is None:
            if 'name' in config:
                name = config['name']
            else:
                import os
                filename = os.path.basename(config_path)
                name = filename.replace('.json', '').replace('_detailed', '').title()
        
        # Merge config with kwargs, ensuring name parameter takes precedence
        merged_config = {**config, **kwargs, 'name': name}
        
        # Remove JSON-specific fields that aren't part of GEval constructor
        merged_config.pop('requires_document', None)
        
        return cls(**merged_config)
