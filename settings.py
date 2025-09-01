import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for GPT-4 evaluation settings."""
    
    def __init__(self):
        # File paths with default values
        self.prompt_fp = os.getenv('PROMPT_FP', 'prompts/summeval/con_detailed.json')  # Changed to JSON
        self.save_fp = os.getenv('SAVE_FP', 'results/gpt4_con_detailed_openai_4.json')
        self.summeval_fp = os.getenv('SUMMEVAL_FP', 'data/summeval_shuffle_4.json')
        
        # Database configuration
        self.database_type = os.getenv('DATABASE_TYPE', 'sqlite')  # 'sqlite' or 'dynamodb'
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.dynamodb_endpoint = os.getenv('DYNAMODB_ENDPOINT')  # For local DynamoDB
        
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
        #self.eval_input_fp = os.getenv('EVAL_INPUT_FP', self.save_fp)  # Default to the save_fp from gpt4_eval
        self.eval_input_fp = os.getenv('EVAL_INPUT_FP', 'results/gpt4_con_detailed_openai_4.json')  # Default to the save_fp from gpt4_eval
        self.evaluation_dimension = os.getenv('EVALUATION_DIMENSION', 'consistency')  # Default dimension

    def load_prompt_config(self, prompt_path: str = None) -> dict:
        """
        Load prompt configuration from JSON file.
        
        Args:
            prompt_path: Path to prompt JSON file. If None, uses self.prompt_fp
            
        Returns:
            Dictionary containing prompt configuration
        """
        if prompt_path is None:
            prompt_path = self.prompt_fp
            
        try:
            with open(prompt_path, 'r') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = ['task_introduction', 'evaluation_criteria']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' in prompt config: {prompt_path}")
            
            # Set default values for optional fields
            config.setdefault('min_score', 1)
            config.setdefault('max_score', 5)
            config.setdefault('requires_document', True)
            
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt configuration file not found: {prompt_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompt configuration file {prompt_path}: {e}")


# Create a global config instance
config = Config()
