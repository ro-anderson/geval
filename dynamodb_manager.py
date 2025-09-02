"""
DynamoDB adapter for G-Eval FastAPI application.

This module provides a DynamoDB-based implementation that maintains backward compatibility
with the existing SQLite DatabaseManager interface while leveraging DynamoDB's strengths.
"""

import json
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError


class DynamoDBManager:
    """
    DynamoDB manager class for G-Eval operations.
    Maintains the same interface as DatabaseManager for backward compatibility.
    """
    
    def __init__(self, region: str = "us-east-1", endpoint_url: Optional[str] = None):
        """
        Initialize DynamoDB manager.
        
        Args:
            region: AWS region for DynamoDB
            endpoint_url: Optional endpoint URL (for local DynamoDB)
        """
        self.dynamodb = boto3.resource(
            'dynamodb',
            region_name=region,
            endpoint_url=endpoint_url
        )
        
        # Table references with rdidier- prefix (3-table architecture)
        alias_ = 'rdidier-'
        
        self.cases_table = self.dynamodb.Table(f'{alias_}CasesConfiguration')
        self.judges_table = self.dynamodb.Table(f'{alias_}JudgesConfiguration')
        self.runs_table = self.dynamodb.Table(f'{alias_}EvaluationRuns')
        
        # In-memory storage for models and metrics (embedded approach)
        self._metrics = {}
        self._models = {}
        self._temp_documents = {}  # Temporary storage for documents until run creation
        
        # Initialize default metrics and models
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize common metrics and models in memory."""
        # Default metrics
        default_metrics = [
            {"id": str(uuid.uuid4()), "name": "geval", "created_at": datetime.utcnow().isoformat()},
            {"id": str(uuid.uuid4()), "name": "claude_eval", "created_at": datetime.utcnow().isoformat()},
        ]
        for metric in default_metrics:
            self._metrics[metric["id"]] = metric
        
        # Default models
        default_models = [
            {"id": str(uuid.uuid4()), "name": "gpt-4o-2024-08-06", "provider": "openai", "created_at": datetime.utcnow().isoformat()},
            {"id": str(uuid.uuid4()), "name": "gpt-4o-mini", "provider": "openai", "created_at": datetime.utcnow().isoformat()},
            {"id": str(uuid.uuid4()), "name": "claude-3-5-sonnet-20241022", "provider": "anthropic", "created_at": datetime.utcnow().isoformat()},
        ]
        for model in default_models:
            self._models[model["id"]] = model
    
    def _convert_decimals(self, obj):
        """Convert DynamoDB Decimal types to float/int for JSON serialization."""
        if isinstance(obj, list):
            return [self._convert_decimals(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_decimals(value) for key, value in obj.items()}
        elif isinstance(obj, Decimal):
            return float(obj) if obj % 1 else int(obj)
        return obj
    
    def _convert_floats_to_decimals(self, obj):
        """Convert float objects to Decimal for DynamoDB storage."""
        if isinstance(obj, list):
            return [self._convert_floats_to_decimals(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_floats_to_decimals(value) for key, value in obj.items()}
        elif isinstance(obj, float):
            return Decimal(str(obj))
        return obj
    
    # METRICS Operations (in-memory for embedded approach)
    def create_metric(self, name: str) -> str:
        """Create a new evaluation methodology."""
        # Check if metric already exists
        for metric_id, metric in self._metrics.items():
            if metric["name"] == name.lower():
                raise ValueError(f"Metric with name '{name}' already exists")
        
        metric_id = str(uuid.uuid4())
        metric = {
            "id": metric_id,
            "name": name.lower(),
            "created_at": datetime.utcnow().isoformat()
        }
        self._metrics[metric_id] = metric
        return metric_id
    
    def get_metric(self, metric_id: str) -> Optional[Dict[str, Any]]:
        """Get metric by ID."""
        return self._metrics.get(metric_id)
    
    def list_metrics(self) -> List[Dict[str, Any]]:
        """List all metrics."""
        return list(self._metrics.values())
    
    # MODELS Operations (in-memory for embedded approach)
    def create_model(self, name: str, provider: str) -> str:
        """Create a new LLM model configuration."""
        # Check if model already exists
        for model_id, model in self._models.items():
            if model["name"] == name and model["provider"] == provider.lower():
                raise ValueError(f"Model with name '{name}' and provider '{provider}' already exists")
        
        model_id = str(uuid.uuid4())
        model = {
            "id": model_id,
            "name": name,
            "provider": provider.lower(),
            "created_at": datetime.utcnow().isoformat()
        }
        self._models[model_id] = model
        return model_id
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID."""
        return self._models.get(model_id)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models."""
        return list(self._models.values())
    
    # THRESHOLDS Operations (now embedded in CasesConfiguration)
    # Threshold history is stored within each case record for better access patterns
    
    def get_threshold(self, threshold_id: str) -> Optional[Dict[str, Any]]:
        """Get threshold by ID - now searches within case threshold history."""
        # Since thresholds are embedded in cases, we'd need to scan cases to find by threshold_id
        # This is typically not needed since thresholds are accessed via case context
        return None
    
    def list_thresholds(self, case_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List thresholds - now returns embedded threshold data from cases."""
        if case_id:
            case = self.get_case(case_id)
            if case and 'threshold_history' in case:
                return case['threshold_history']
        # For global threshold listing, we'd need to scan all cases
        return []
    
    # CASES Operations
    def create_case(
        self,
        name: str,
        task_introduction: str,
        evaluation_criteria: str,
        min_score: int = 1,
        max_score: int = 5,
        requires_reference: bool = False,
        score_threshold: float = 0.5
    ) -> str:
        """Create a new evaluation case with embedded threshold history."""
        case_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        version = "1.0"
        threshold_id = str(uuid.uuid4())
        
        # Create initial threshold history entry
        initial_threshold = {
            'thresholdId': threshold_id,
            'score': Decimal(str(score_threshold)),
            'createdBy': 'system',
            'reason': 'Initial case creation',
            'createdAt': timestamp
        }
        
        try:
            self.cases_table.put_item(
                Item={
                    'partitionKey': case_id,
                    'sortKey': version,
                    'active': 'true',
                    'caseId': case_id,
                    'name': name.lower(),
                    'taskIntroduction': task_introduction,
                    'evaluationCriteria': evaluation_criteria,
                    'scoreRange': {
                        'min': min_score,
                        'max': max_score
                    },
                    'requiresReference': requires_reference,
                    'currentThreshold': Decimal(str(score_threshold)),
                    'thresholdHistory': [initial_threshold],
                    'created_at': timestamp,
                    'createdOn': timestamp,
                    'lastUpdatedOn': timestamp,
                    'versionTag': version
                }
            )
        except ClientError as e:
            if 'ConditionalCheckFailedException' in str(e):
                raise ValueError(f"Case with name '{name}' already exists")
            raise RuntimeError(f"Failed to create case: {e}")
        
        return case_id
    
    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get case by ID with current threshold."""
        try:
            response = self.cases_table.query(
                IndexName='ActiveCasesIndex',
                KeyConditionExpression='#active = :active AND caseId = :case_id',
                ExpressionAttributeNames={'#active': 'active'},
                ExpressionAttributeValues={
                    ':active': 'true',
                    ':case_id': case_id
                }
            )
            
            items = response.get('Items', [])
            if items:
                item = self._convert_decimals(items[0])
                # Convert to expected format with embedded threshold
                latest_threshold = item.get('thresholdHistory', [{}])[-1] if item.get('thresholdHistory') else {}
                return {
                    'id': item['caseId'],
                    'name': item['name'],
                    'task_introduction': item['taskIntroduction'],
                    'evaluation_criteria': item['evaluationCriteria'],
                    'min_score': item['scoreRange']['min'],
                    'max_score': item['scoreRange']['max'],
                    'requires_reference': item['requiresReference'],
                    'score_threshold': item.get('currentThreshold', 0.5),
                    'threshold_id': latest_threshold.get('thresholdId', 'unknown'),
                    'created_at': item['created_at']
                }
            return None
        except ClientError:
            return None
    
    def list_cases(self) -> List[Dict[str, Any]]:
        """List all cases with current thresholds."""
        try:
            response = self.cases_table.query(
                IndexName='ActiveCasesIndex',
                KeyConditionExpression='#active = :active',
                ExpressionAttributeNames={'#active': 'active'},
                ExpressionAttributeValues={':active': 'true'}
            )
            
            items = response.get('Items', [])
            cases = []
            
            for item in items:
                converted = self._convert_decimals(item)
                latest_threshold = converted.get('thresholdHistory', [{}])[-1] if converted.get('thresholdHistory') else {}
                cases.append({
                    'id': converted['caseId'],
                    'name': converted['name'],
                    'task_introduction': converted['taskIntroduction'],
                    'evaluation_criteria': converted['evaluationCriteria'],
                    'min_score': converted['scoreRange']['min'],
                    'max_score': converted['scoreRange']['max'],
                    'requires_reference': converted['requiresReference'],
                    'score_threshold': converted.get('currentThreshold', 0.5),
                    'threshold_id': latest_threshold.get('thresholdId', 'unknown'),
                    'created_at': converted['created_at']
                })
            
            return sorted(cases, key=lambda x: x['created_at'], reverse=True)
        except ClientError:
            return []
    
    def update_case(
        self,
        case_id: str,
        name: Optional[str] = None,
        task_introduction: Optional[str] = None,
        evaluation_criteria: Optional[str] = None,
        min_score: Optional[int] = None,
        max_score: Optional[int] = None,
        requires_reference: Optional[bool] = None,
        score_threshold: Optional[float] = None
    ) -> bool:
        """Update case information with embedded threshold history."""
        
        # Build update expression
        update_expressions = []
        expression_values = {}
        expression_attribute_names = {}
        
        if name is not None:
            update_expressions.append('#name = :name')
            expression_values[':name'] = name.lower()
            expression_attribute_names['#name'] = 'name'
        
        if task_introduction is not None:
            update_expressions.append('taskIntroduction = :task_intro')
            expression_values[':task_intro'] = task_introduction
        
        if evaluation_criteria is not None:
            update_expressions.append('evaluationCriteria = :eval_criteria')
            expression_values[':eval_criteria'] = evaluation_criteria
        
        if min_score is not None:
            update_expressions.append('scoreRange.#min = :min_score')
            expression_values[':min_score'] = min_score
            expression_attribute_names['#min'] = 'min'
        
        if max_score is not None:
            update_expressions.append('scoreRange.#max = :max_score')
            expression_values[':max_score'] = max_score
            expression_attribute_names['#max'] = 'max'
        
        if requires_reference is not None:
            update_expressions.append('requiresReference = :requires_ref')
            expression_values[':requires_ref'] = requires_reference
        
        # Handle threshold update with embedded history
        if score_threshold is not None:
            timestamp = datetime.utcnow().isoformat()
            threshold_id = str(uuid.uuid4())
            
            # Create new threshold history entry
            new_threshold_entry = {
                'thresholdId': threshold_id,
                'score': Decimal(str(score_threshold)),
                'createdBy': 'user',  # Could be enhanced to pass actual user
                'reason': 'Manual threshold update',
                'createdAt': timestamp
            }
            
            # Update current threshold and append to history
            update_expressions.append('currentThreshold = :current_threshold')
            update_expressions.append('thresholdHistory = list_append(if_not_exists(thresholdHistory, :empty_list), :new_threshold)')
            
            expression_values[':current_threshold'] = Decimal(str(score_threshold))
            expression_values[':new_threshold'] = [new_threshold_entry]
            expression_values[':empty_list'] = []
        
        if not update_expressions:
            return True
        
        # Add lastUpdatedOn
        update_expressions.append('lastUpdatedOn = :updated_on')
        expression_values[':updated_on'] = datetime.utcnow().isoformat()
        
        try:
            update_params = {
                'Key': {
                    'partitionKey': case_id,
                    'sortKey': '1.0'  # Assuming version 1.0 for active cases
                },
                'UpdateExpression': f"SET {', '.join(update_expressions)}",
                'ExpressionAttributeValues': expression_values
            }
            
            # Only include ExpressionAttributeNames if we have any
            if expression_attribute_names:
                update_params['ExpressionAttributeNames'] = expression_attribute_names
            
            self.cases_table.update_item(**update_params)
            return True
        except ClientError as e:
            print(f"Update case error: {e}")
            return False
    
    # DOCUMENTS Operations (temporary storage for backward compatibility)
    def create_document(
        self,
        actual_output: str,
        expected_output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new document for evaluation (temporarily stored)."""
        doc_id = str(uuid.uuid4())
        document = {
            'id': doc_id,
            'actual_output': actual_output,
            'expected_output': expected_output,
            'metadata': metadata or {},
            'created_at': datetime.utcnow().isoformat()
        }
        self._temp_documents[doc_id] = document
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        return self._temp_documents.get(doc_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents."""
        return list(self._temp_documents.values())
    
    # JUDGES Operations
    def create_judge(
        self,
        name: Optional[str],
        model_id: str,
        case_id: str,
        metric_id: str,
        parameters: Dict[str, Any],
        description: Optional[str] = None
    ) -> str:
        """Create a new specialized judge."""
        judge_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        version = "1.0"
        
        # Get model and metric info
        model = self.get_model(model_id)
        metric = self.get_metric(metric_id)
        case = self.get_case(case_id)
        
        if not model or not metric or not case:
            raise ValueError("Model, metric, or case not found")
        
        try:
            self.judges_table.put_item(
                Item={
                    'partitionKey': judge_id,
                    'sortKey': version,
                    'active': 'true',
                    'judgeId': judge_id,
                    'name': name or f"{case['name'].title()} Judge",
                    'description': description,
                    'caseId': case_id,
                    'caseName': case['name'],
                    
                    # Embedded metric configuration
                    'metric': {
                        'id': metric_id,
                        'type': metric['name'],
                        'name': f"{metric['name'].title()} Metric"
                    },
                    
                    # Embedded LLM configuration
                    'llm': {
                        'id': model_id,
                        'type': model['provider'],
                        'model': model['name'],
                        'temperature': Decimal(str(parameters.get('temperature', 2.0))),
                        'maxTokens': parameters.get('max_tokens', 2500),
                        'topP': Decimal(str(parameters.get('top_p', 1.0))),
                        'frequencyPenalty': Decimal(str(parameters.get('frequency_penalty', 0.0))),
                        'presencePenalty': Decimal(str(parameters.get('presence_penalty', 0.0)))
                    },
                    
                    # Evaluation parameters
                    'evaluationParameters': {
                        'nResponses': parameters.get('n_responses', 10),
                        'sleepTime': Decimal(str(parameters.get('sleep_time', 0.0))),
                        'rateLimitSleep': Decimal(str(parameters.get('rate_limit_sleep', 0.0))),
                        'retryAttempts': 3
                    },
                    
                    'createdOn': timestamp,
                    'lastUpdatedOn': timestamp,
                    'versionTag': version
                }
            )
        except ClientError as e:
            raise RuntimeError(f"Failed to create judge: {e}")
        
        return judge_id
    
    def get_judge(self, judge_id: str) -> Optional[Dict[str, Any]]:
        """Get judge by ID with associated case, metric, and model data."""
        try:
            response = self.judges_table.query(
                IndexName='ActiveJudgesIndex',
                KeyConditionExpression='#active = :active AND judgeId = :judge_id',
                ExpressionAttributeNames={'#active': 'active'},
                ExpressionAttributeValues={
                    ':active': 'true',
                    ':judge_id': judge_id
                }
            )
            
            items = response.get('Items', [])
            if items:
                item = self._convert_decimals(items[0])
                
                # Get case details
                case = self.get_case(item['caseId'])
                if not case:
                    return None
                
                # Convert to expected format
                return {
                    'id': item['judgeId'],
                    'name': item['name'],
                    'model_id': item['llm']['id'],
                    'case_id': item['caseId'],
                    'metric_id': item['metric']['id'],
                    'case_name': item['caseName'],
                    'metric_name': item['metric']['type'],
                    'model_name': item['llm']['model'],
                    'model_provider': item['llm']['type'],
                    'task_introduction': case['task_introduction'],
                    'evaluation_criteria': case['evaluation_criteria'],
                    'min_score': case['min_score'],
                    'max_score': case['max_score'],
                    'requires_reference': case['requires_reference'],
                    'parameters': {
                        'temperature': item['llm']['temperature'],
                        'max_tokens': item['llm']['maxTokens'],
                        'top_p': item['llm']['topP'],
                        'frequency_penalty': item['llm']['frequencyPenalty'],
                        'presence_penalty': item['llm']['presencePenalty'],
                        'n_responses': item['evaluationParameters']['nResponses'],
                        'sleep_time': item['evaluationParameters']['sleepTime'],
                        'rate_limit_sleep': item['evaluationParameters']['rateLimitSleep']
                    },
                    'description': item.get('description'),
                    'created_at': item['createdOn']
                }
            return None
        except ClientError:
            return None
    
    def list_judges(self) -> List[Dict[str, Any]]:
        """List all judges with their case, metric, and model info."""
        try:
            response = self.judges_table.query(
                IndexName='ActiveJudgesIndex',
                KeyConditionExpression='#active = :active',
                ExpressionAttributeNames={'#active': 'active'},
                ExpressionAttributeValues={':active': 'true'}
            )
            
            items = response.get('Items', [])
            judges = []
            
            for item in items:
                converted = self._convert_decimals(item)
                
                # Get case details for each judge
                case = self.get_case(converted['caseId'])
                if case:
                    judges.append({
                        'id': converted['judgeId'],
                        'name': converted['name'],
                        'model_id': converted['llm']['id'],
                        'case_id': converted['caseId'],
                        'metric_id': converted['metric']['id'],
                        'case_name': converted['caseName'],
                        'metric_name': converted['metric']['type'],
                        'model_name': converted['llm']['model'],
                        'model_provider': converted['llm']['type'],
                        'parameters': {
                            'temperature': converted['llm']['temperature'],
                            'max_tokens': converted['llm']['maxTokens'],
                            'top_p': converted['llm']['topP'],
                            'frequency_penalty': converted['llm']['frequencyPenalty'],
                            'presence_penalty': converted['llm']['presencePenalty'],
                            'n_responses': converted['evaluationParameters']['nResponses'],
                            'sleep_time': converted['evaluationParameters']['sleepTime'],
                            'rate_limit_sleep': converted['evaluationParameters']['rateLimitSleep']
                        },
                        'description': converted.get('description'),
                        'created_at': converted['createdOn']
                    })
            
            return sorted(judges, key=lambda x: x['created_at'], reverse=True)
        except ClientError:
            return []
    
    def update_judge(
        self,
        judge_id: str,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> bool:
        """Update judge information."""
        # Build update expression
        update_expressions = []
        expression_values = {}
        
        if name is not None:
            update_expressions.append('#name = :name')
            expression_values[':name'] = name
        
        if description is not None:
            update_expressions.append('description = :description')
            expression_values[':description'] = description
        
        if model_id is not None:
            model = self.get_model(model_id)
            if not model:
                return False
            
            update_expressions.append('llm.id = :model_id')
            update_expressions.append('llm.#type = :model_provider')
            update_expressions.append('llm.model = :model_name')
            expression_values[':model_id'] = model_id
            expression_values[':model_provider'] = model['provider']
            expression_values[':model_name'] = model['name']
        
        if parameters is not None:
            for param, value in parameters.items():
                if param == 'temperature':
                    update_expressions.append('llm.temperature = :temperature')
                    expression_values[':temperature'] = Decimal(str(value))
                elif param == 'max_tokens':
                    update_expressions.append('llm.maxTokens = :max_tokens')
                    expression_values[':max_tokens'] = value
                elif param == 'top_p':
                    update_expressions.append('llm.topP = :top_p')
                    expression_values[':top_p'] = Decimal(str(value))
                elif param == 'frequency_penalty':
                    update_expressions.append('llm.frequencyPenalty = :frequency_penalty')
                    expression_values[':frequency_penalty'] = Decimal(str(value))
                elif param == 'presence_penalty':
                    update_expressions.append('llm.presencePenalty = :presence_penalty')
                    expression_values[':presence_penalty'] = Decimal(str(value))
                elif param == 'n_responses':
                    update_expressions.append('evaluationParameters.nResponses = :n_responses')
                    expression_values[':n_responses'] = value
                elif param == 'sleep_time':
                    update_expressions.append('evaluationParameters.sleepTime = :sleep_time')
                    expression_values[':sleep_time'] = Decimal(str(value))
                elif param == 'rate_limit_sleep':
                    update_expressions.append('evaluationParameters.rateLimitSleep = :rate_limit_sleep')
                    expression_values[':rate_limit_sleep'] = Decimal(str(value))
        
        if not update_expressions:
            return True
        
        # Add lastUpdatedOn
        update_expressions.append('lastUpdatedOn = :updated_on')
        expression_values[':updated_on'] = datetime.utcnow().isoformat()
        
        try:
            # First check if the judge exists
            response = self.judges_table.get_item(
                Key={
                    'partitionKey': judge_id,
                    'sortKey': '1.0'
                }
            )
            
            if 'Item' not in response:
                print(f"Judge {judge_id} not found in DynamoDB")
                return False
            
            # Build expression attribute names dynamically based on what we're updating
            expression_attribute_names = {}
            
            # Check if we need #name (when updating name)
            if name is not None:
                expression_attribute_names['#name'] = 'name'
            
            # Check if we need #type (when updating model_id) 
            if model_id is not None:
                expression_attribute_names['#type'] = 'type'
            
            # Update the judge
            update_params = {
                'Key': {
                    'partitionKey': judge_id,
                    'sortKey': '1.0'  # Assuming version 1.0 for active judges
                },
                'UpdateExpression': f"SET {', '.join(update_expressions)}",
                'ExpressionAttributeValues': expression_values
            }
            
            # Only include ExpressionAttributeNames if we have any
            if expression_attribute_names:
                update_params['ExpressionAttributeNames'] = expression_attribute_names
            
            self.judges_table.update_item(**update_params)
            return True
        except ClientError as e:
            print(f"Update judge error: {e}")
            return False
    
    # RUNS Operations
    def create_run(
        self, 
        judge_id: str, 
        document_id: Optional[str] = None,
        actual_output: Optional[str] = None,
        expected_output: Optional[str] = None
    ) -> int:
        """Create a new run record."""
        # Generate UUID for run (DynamoDB uses string keys, not auto-increment)
        run_uuid = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        run_date = timestamp.split('T')[0]  # YYYY-MM-DD
        
        # Get judge info for snapshot
        judge = self.get_judge(judge_id)
        if not judge:
            raise ValueError("Judge not found")
        
        # Handle document content - support both modes
        if document_id:
            # Legacy mode: get document from storage
            document = self.get_document(document_id)
            if not document:
                raise ValueError("Document not found")
            evaluation_target = document['actual_output']
            expected_output_content = document.get('expected_output')
        else:
            # Direct content mode: use provided content
            if not actual_output:
                raise ValueError("Either document_id or actual_output must be provided")
            evaluation_target = actual_output
            expected_output_content = expected_output
            document_id = None  # No document reference
        
        # Create run with snapshot
        try:
            self.runs_table.put_item(
                Item={
                    'partitionKey': f"RUN#{run_uuid}",
                    'sortKey': timestamp,
                    'runId': run_uuid,
                    'runDate': run_date,
                    'startedAt': timestamp,
                    'status': 'pending',
                    'evaluationStatus': 'pending',
                    
                    # Judge snapshot
                    'judgeSnapshot': {
                        'judgeId': judge['id'],
                        'judgeName': judge['name'],
                        'version': '1.0',
                        'caseId': judge['case_id'],
                        'caseName': judge['case_name'],
                        'metric': {
                            'type': judge['metric_name'],
                            'name': f"{judge['metric_name'].title()} Metric"
                        },
                        'llm': {
                            'type': judge['model_provider'],
                            'model': judge['model_name'],
                            'temperature': Decimal(str(judge['parameters']['temperature']))
                        },
                        'evaluationParameters': {
                            'nResponses': int(judge['parameters']['n_responses'])  # Ensure integer
                        },
                        'scoreRange': {
                            'min': judge['min_score'],
                            'max': judge['max_score']
                        }
                    },
                    
                    # Evaluated artifact (renamed fields)
                    'evaluatedArtifact': {
                        'evaluationTarget': evaluation_target,
                        'expectedOutput': expected_output_content,
                        'metadata': {}  # No metadata for direct content mode
                    },
                    
                    # For GSI queries (backward compatibility)
                    'judgeId': judge_id,
                    'caseId': judge['case_id'],
                    'documentId': document_id  # May be None for direct content mode
                }
            )
            
            # Return a hash of the UUID as integer for backward compatibility
            return hash(run_uuid) % (10**10)  # Ensure it fits in typical int range
            
        except ClientError as e:
            raise RuntimeError(f"Failed to create run: {e}")
    
    def update_run_status(self, run_id: int, status: str, error_message: Optional[str] = None):
        """Update run status."""
        # Find run by run_id hash
        run_uuid = self._find_run_uuid_by_id(run_id)
        if not run_uuid:
            return
        
        update_expressions = ['#status = :status']
        expression_values = {':status': status}
        
        if error_message is not None:
            update_expressions.append('errorMessage = :error_msg')
            expression_values[':error_msg'] = error_message
        
        try:
            # We need to get the sort key first
            response = self.runs_table.query(
                KeyConditionExpression='partitionKey = :pk',
                ExpressionAttributeValues={':pk': f"RUN#{run_uuid}"},
                Limit=1
            )
            
            if response['Items']:
                sort_key = response['Items'][0]['sortKey']
                
                self.runs_table.update_item(
                    Key={
                        'partitionKey': f"RUN#{run_uuid}",
                        'sortKey': sort_key
                    },
                    UpdateExpression=f"SET {', '.join(update_expressions)}",
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues=expression_values
                )
        except ClientError:
            pass  # Silently fail for backward compatibility
    
    def _find_run_uuid_by_id(self, run_id: int) -> Optional[str]:
        """Find run UUID by integer run_id (reverse of hash)."""
        # This is inefficient but needed for backward compatibility
        # In production, you might want to maintain a mapping table
        try:
            response = self.runs_table.scan()
            for item in response.get('Items', []):
                run_uuid = item['runId']
                if hash(run_uuid) % (10**10) == run_id:
                    return run_uuid
            return None
        except ClientError:
            return None
    
    def complete_run(
        self,
        run_id: int,
        final_score: float,
        final_score_normalized: float,
        all_responses: List[float],
        total_usage_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        execution_time_seconds: float,
        prompt_used: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Complete a run with results and calculate evaluation status."""
        run_uuid = self._find_run_uuid_by_id(run_id)
        if not run_uuid:
            return
        
        try:
            # Get run to access judge snapshot
            response = self.runs_table.query(
                KeyConditionExpression='partitionKey = :pk',
                ExpressionAttributeValues={':pk': f"RUN#{run_uuid}"},
                Limit=1
            )
            
            if not response['Items']:
                return
            
            run_item = response['Items'][0]
            sort_key = run_item['sortKey']
            
            # Get threshold from judge snapshot (or fetch from case)
            case_id = run_item['judgeSnapshot']['caseId']
            case = self.get_case(case_id)
            score_threshold = case['score_threshold'] if case else 0.5
            
            # Determine evaluation status
            evaluation_status = 'pass' if final_score_normalized >= score_threshold else 'fail'
            
            self.runs_table.update_item(
                Key={
                    'partitionKey': f"RUN#{run_uuid}",
                    'sortKey': sort_key
                },
                UpdateExpression="""
                    SET #status = :status,
                        evaluationStatus = :eval_status,
                        completedAt = :completed_at,
                        results = :results
                """,
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':status': 'completed',
                    ':eval_status': evaluation_status,
                    ':completed_at': datetime.utcnow().isoformat(),
                    ':results': {
                        'finalScore': Decimal(str(final_score)),
                        'finalScoreNormalized': Decimal(str(final_score_normalized)),
                        'allResponses': [Decimal(str(r)) for r in all_responses],
                        'promptUsed': prompt_used,
                        'tokenUsage': {
                            'prompt': prompt_tokens,
                            'completion': completion_tokens,
                            'total': total_usage_tokens
                        },
                        'executionTimeSeconds': Decimal(str(execution_time_seconds))
                    }
                }
            )
        except ClientError:
            pass  # Silently fail for backward compatibility
    
    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Get run by ID with full details."""
        run_uuid = self._find_run_uuid_by_id(run_id)
        if not run_uuid:
            return None
        
        try:
            response = self.runs_table.query(
                KeyConditionExpression='partitionKey = :pk',
                ExpressionAttributeValues={':pk': f"RUN#{run_uuid}"},
                Limit=1
            )
            
            if not response['Items']:
                return None
            
            item = self._convert_decimals(response['Items'][0])
            
            # Convert to expected format
            judge_snapshot = item['judgeSnapshot']
            evaluated_artifact = item['evaluatedArtifact']
            results = item.get('results', {})
            
            return {
                'id': run_id,  # Use the integer ID for backward compatibility
                'judge_id': item['judgeId'],
                'document_id': item.get('documentId', ''),
                'judge_name': judge_snapshot['judgeName'],
                'case_name': judge_snapshot['caseName'],
                'metric_name': judge_snapshot['metric']['type'],
                'model_name': judge_snapshot['llm']['model'],
                'model_provider': judge_snapshot['llm']['type'],
                'status': item['status'],
                'evaluation_status': item.get('evaluationStatus'),
                'final_score': results.get('finalScore'),
                'final_score_normalized': results.get('finalScoreNormalized'),
                'all_responses': results.get('allResponses', []),
                'total_usage_tokens': results.get('tokenUsage', {}).get('total'),
                'prompt_tokens': results.get('tokenUsage', {}).get('prompt'),
                'completion_tokens': results.get('tokenUsage', {}).get('completion'),
                'execution_time_seconds': results.get('executionTimeSeconds'),
                'prompt_used': results.get('promptUsed'),
                'error_message': item.get('errorMessage'),
                'metadata': results.get('metadata'),
                'actual_output': evaluated_artifact['evaluationTarget'],  # Map back to original name
                'expected_output': evaluated_artifact.get('expectedOutput'),
                'started_at': item['startedAt'],
                'completed_at': item.get('completedAt'),
                'created_at': item['startedAt']  # Use startedAt as created_at
            }
        except ClientError:
            return None
    
    def list_runs(self, judge_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List runs, optionally filtered by judge."""
        try:
            if judge_id:
                response = self.runs_table.query(
                    IndexName='RunsByJudgeIndex',
                    KeyConditionExpression='judgeId = :judge_id',
                    ExpressionAttributeValues={':judge_id': judge_id},
                    ScanIndexForward=False,
                    Limit=limit
                )
            else:
                response = self.runs_table.scan(Limit=limit)
            
            items = response.get('Items', [])
            runs = []
            
            for item in items:
                converted = self._convert_decimals(item)
                
                judge_snapshot = converted['judgeSnapshot']
                evaluated_artifact = converted['evaluatedArtifact']
                results = converted.get('results', {})
                
                # Generate integer run_id for backward compatibility
                run_uuid = converted['runId']
                run_id = hash(run_uuid) % (10**10)
                
                runs.append({
                    'id': run_id,
                    'judge_id': converted['judgeId'],
                    'document_id': converted.get('documentId', ''),
                    'judge_name': judge_snapshot['judgeName'],
                    'case_name': judge_snapshot['caseName'],
                    'metric_name': judge_snapshot['metric']['type'],
                    'model_name': judge_snapshot['llm']['model'],
                    'model_provider': judge_snapshot['llm']['type'],
                    'status': converted['status'],
                    'evaluation_status': converted.get('evaluationStatus'),
                    'final_score': results.get('finalScore'),
                    'final_score_normalized': results.get('finalScoreNormalized'),
                    'all_responses': results.get('allResponses', []),
                    'total_usage_tokens': results.get('tokenUsage', {}).get('total'),
                    'prompt_tokens': results.get('tokenUsage', {}).get('prompt'),
                    'completion_tokens': results.get('tokenUsage', {}).get('completion'),
                    'execution_time_seconds': results.get('executionTimeSeconds'),
                    'prompt_used': results.get('promptUsed'),
                    'error_message': converted.get('errorMessage'),
                    'metadata': results.get('metadata'),
                    'actual_output': evaluated_artifact['evaluationTarget'],  # Map back to original name
                    'expected_output': evaluated_artifact.get('expectedOutput'),
                    'started_at': converted['startedAt'],
                    'completed_at': converted.get('completedAt'),
                    'created_at': converted['startedAt']
                })
            
            return sorted(runs, key=lambda x: x['created_at'], reverse=True)
        except ClientError:
            return []
