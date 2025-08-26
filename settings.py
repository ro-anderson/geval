import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for GPT-4 evaluation settings."""
    
    def __init__(self):
        # File paths with default values
        self.prompt_fp = os.getenv('PROMPT_FP', 'prompts/summeval/con_detailed.txt')
        self.save_fp = os.getenv('SAVE_FP', 'results/gpt4_con_detailed_openai_500.json')
        self.summeval_fp = os.getenv('SUMMEVAL_FP', 'data/summeval_shuffle_500.json')
        
        # OpenAI API configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Model configuration - using gpt-4o for structured outputs support
        self.model = os.getenv('MODEL', 'gpt-4o-2024-08-06')
        
        # API call parameters
        self.temperature = float(os.getenv('TEMPERATURE', '2'))
        self.max_tokens = int(os.getenv('MAX_TOKENS', '2500'))  # Increased for structured JSON output
        self.top_p = float(os.getenv('TOP_P', '1'))
        self.frequency_penalty = float(os.getenv('FREQUENCY_PENALTY', '0'))
        self.presence_penalty = float(os.getenv('PRESENCE_PENALTY', '0'))
        self.n_responses = int(os.getenv('N_RESPONSES', '10'))
        
        # Sleep time between API calls
        self.sleep_time = float(os.getenv('SLEEP_TIME', '0'))
        self.rate_limit_sleep = float(os.getenv('RATE_LIMIT_SLEEP', '0'))
        
        # Evaluation correlation settings
        self.eval_input_fp = os.getenv('EVAL_INPUT_FP', self.save_fp)  # Default to the save_fp from gpt4_eval
        self.evaluation_dimension = os.getenv('EVALUATION_DIMENSION', 'consistency')  # Default dimension


# Create a global config instance
config = Config()
