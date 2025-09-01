#!/usr/bin/env python3
"""
DynamoDB Schema Design for G-Eval System
Infrastructure as Code using AWS CDK (Python)
"""

from aws_cdk import (
    Stack,
    aws_dynamodb as dynamodb,
    RemovalPolicy,
    CfnOutput
)
from constructs import Construct

class GEvalDynamoDBStack(Stack):
    """
    DynamoDB tables for G-Eval evaluation system
    Following naming convention: PascalCase for table names with didier- prefix
    """
    
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Resource prefix for easy identification and cleanup
        self.prefix = "didier-"
        
        # ====================
        # 1. CasesConfiguration Table
        # ====================
        # Stores evaluation cases (what to evaluate)
        # Pattern: caseId#version for versioning
        
        self.cases_table = dynamodb.Table(
            self, "CasesConfiguration",
            table_name=f"{self.prefix}CasesConfiguration",
            partition_key=dynamodb.Attribute(
                name="partitionKey",  # caseId (e.g., "consistency")
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="sortKey",  # version (e.g., "1.0", "2.0")
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,  # For dev - use RETAIN in prod
            point_in_time_recovery=True,
            
            # Global Secondary Indexes
            # GSI for active cases lookup
        )
        
        # GSI for finding active cases
        self.cases_table.add_global_secondary_index(
            index_name="ActiveCasesIndex",
            partition_key=dynamodb.Attribute(
                name="active",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="caseId",
                type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL
        )
        
        # ====================
        # 2. JudgesConfiguration Table
        # ====================
        # Stores judge configurations (specialized evaluators)
        # Includes embedded model and metric information
        
        self.judges_table = dynamodb.Table(
            self, "JudgesConfiguration",
            table_name=f"{self.prefix}JudgesConfiguration",
            partition_key=dynamodb.Attribute(
                name="partitionKey",  # judgeId (e.g., "consistency-expert")
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="sortKey",  # version (e.g., "1.0", "1.1")
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            point_in_time_recovery=True
        )
        
        # GSI for finding judges by case
        self.judges_table.add_global_secondary_index(
            index_name="JudgesByCaseIndex",
            partition_key=dynamodb.Attribute(
                name="caseId",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="active",
                type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL
        )
        
        # GSI for active judges
        self.judges_table.add_global_secondary_index(
            index_name="ActiveJudgesIndex",
            partition_key=dynamodb.Attribute(
                name="active",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="judgeId",
                type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL
        )
        
        # ====================
        # 3. EvaluationRuns Table
        # ====================
        # Stores evaluation execution history with complete snapshots
        # No need for separate documents table - embedded in runs
        
        self.runs_table = dynamodb.Table(
            self, "EvaluationRuns",
            table_name=f"{self.prefix}EvaluationRuns",
            partition_key=dynamodb.Attribute(
                name="partitionKey",  # Format: "RUN#{uuid}"
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="sortKey",  # Timestamp for ordering
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            point_in_time_recovery=True,
            stream=dynamodb.StreamViewType.NEW_AND_OLD_IMAGES  # For event processing
        )
        
        # GSI for runs by judge
        self.runs_table.add_global_secondary_index(
            index_name="RunsByJudgeIndex",
            partition_key=dynamodb.Attribute(
                name="judgeId",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="startedAt",
                type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL
        )
        
        # GSI for runs by case
        self.runs_table.add_global_secondary_index(
            index_name="RunsByCaseIndex",
            partition_key=dynamodb.Attribute(
                name="caseId",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="startedAt",
                type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL
        )
        
        # GSI for runs by date (for dashboard queries)
        self.runs_table.add_global_secondary_index(
            index_name="RunsByDateIndex",
            partition_key=dynamodb.Attribute(
                name="runDate",  # YYYY-MM-DD format
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="startedAt",
                type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL
        )
        
        # GSI for runs by status
        self.runs_table.add_global_secondary_index(
            index_name="RunsByStatusIndex",
            partition_key=dynamodb.Attribute(
                name="status",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="startedAt",
                type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL
        )
        
        # ====================
        # 4. Thresholds Table (Optional - could be embedded)
        # ====================
        # Historical tracking of threshold changes
        # Could also be embedded in CasesConfiguration with versioning
        
        self.thresholds_table = dynamodb.Table(
            self, "ThresholdsHistory",
            table_name=f"{self.prefix}ThresholdsHistory",
            partition_key=dynamodb.Attribute(
                name="partitionKey",  # caseId
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="sortKey",  # timestamp
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            point_in_time_recovery=True
        )
        
        # Outputs
        CfnOutput(self, "CasesTableName", value=self.cases_table.table_name)
        CfnOutput(self, "JudgesTableName", value=self.judges_table.table_name)
        CfnOutput(self, "RunsTableName", value=self.runs_table.table_name)
        CfnOutput(self, "ThresholdsTableName", value=self.thresholds_table.table_name)


# ====================
# Data Structure Examples
# ====================

"""
CasesConfiguration Item Example:
{
    "partitionKey": "consistency",           # caseId
    "sortKey": "1.0",                        # version
    "active": "true",                         # only one version active at a time
    "caseId": "consistency",                  
    "name": "Consistency Evaluation",
    "taskIntroduction": "Evaluate text consistency...",
    "evaluationCriteria": "The text should...",
    "scoreRange": {
        "min": 1,
        "max": 5
    },
    "requiresReference": false,
    "threshold": {
        "score": 0.7,                        # Normalized 0-1
        "description": "70% minimum for pass"
    },
    "createdBy": "arn:aws:sts::...",
    "createdOn": "2025-01-01T10:00:00Z",
    "lastUpdatedBy": "arn:aws:sts::...",
    "lastUpdatedOn": "2025-01-01T10:00:00Z",
    "description": "Evaluates logical consistency",
    "versionTag": "1.0",
    "metadata": {
        "category": "quality",
        "difficulty": "medium"
    }
}

JudgesConfiguration Item Example:
{
    "partitionKey": "consistency-expert",     # judgeId
    "sortKey": "1.0",                        # version
    "active": "true",
    "judgeId": "consistency-expert",
    "name": "Consistency Expert Judge",
    "description": "Specialized in evaluating text consistency",
    "caseId": "consistency",                 # Reference to case
    "caseName": "Consistency Evaluation",    # Denormalized for quick access
    
    # Embedded metric configuration
    "metric": {
        "type": "geval",                     # or "claude_eval", "custom"
        "name": "G-Eval Metric",
        "configuration": {
            "algorithm": "geval-v1"
        }
    },
    
    # Embedded LLM configuration (similar to AgentsConfiguration)
    "llm": {
        "type": "openai",
        "model": "gpt-4o-2024-08-06",
        "temperature": 0.7,
        "maxTokens": 2500,
        "topP": 1.0,
        "frequencyPenalty": 0.0,
        "presencePenalty": 0.0
    },
    
    # Evaluation parameters
    "evaluationParameters": {
        "nResponses": 10,
        "sleepTime": 0.0,
        "rateLimitSleep": 0.0,
        "retryAttempts": 3
    },
    
    "createdBy": "arn:aws:sts::...",
    "createdOn": "2025-01-01T10:00:00Z",
    "lastUpdatedBy": "arn:aws:sts::...",
    "lastUpdatedOn": "2025-01-01T10:00:00Z",
    "versionTag": "1.0"
}

EvaluationRuns Item Example:
{
    "partitionKey": "RUN#550e8400-e29b-41d4-a716-446655440000",
    "sortKey": "2025-01-01T10:30:00.000Z",
    "runId": "550e8400-e29b-41d4-a716-446655440000",
    "runDate": "2025-01-01",                 # For date-based queries
    "startedAt": "2025-01-01T10:30:00.000Z",
    "completedAt": "2025-01-01T10:30:15.000Z",
    "status": "completed",                   # pending|running|completed|failed
    "evaluationStatus": "pass",              # pass|fail|pending
    
    # Judge snapshot at time of execution
    "judgeSnapshot": {
        "judgeId": "consistency-expert",
        "judgeName": "Consistency Expert Judge",
        "version": "1.0",
        "caseId": "consistency",
        "caseName": "Consistency Evaluation",
        "metric": {
            "type": "geval",
            "name": "G-Eval Metric"
        },
        "llm": {
            "type": "openai",
            "model": "gpt-4o-2024-08-06",
            "temperature": 0.7
        },
        "evaluationParameters": {
            "nResponses": 10
        },
        "scoreRange": {
            "min": 1,
            "max": 5
        },
        "threshold": 0.7
    },
    
    # Evaluated artifact (replaces eval_documents)
    "evaluatedArtifact": {
        "evaluationTarget": "The text being evaluated...",  # was actual_output
        "expectedOutput": "The reference text...",          # was expected_output (optional)
        "metadata": {
            "source": "api",
            "requestId": "req-123",
            "userId": "user-456"
        }
    },
    
    # Evaluation results
    "results": {
        "finalScore": 4.2,
        "finalScoreNormalized": 0.8,         # (4.2-1)/(5-1) = 0.8
        "allResponses": [4.0, 4.5, 4.1, 4.3, 4.2],
        "promptUsed": "Evaluate the following...",
        "tokenUsage": {
            "prompt": 1500,
            "completion": 500,
            "total": 2000
        },
        "executionTimeSeconds": 15.3
    },
    
    "judgeId": "consistency-expert",         # For GSI queries
    "caseId": "consistency",                 # For GSI queries
    "createdBy": "arn:aws:sts::...",
    "errorMessage": null,
    "metadata": {
        "environment": "production",
        "apiVersion": "1.0"
    }
}

ThresholdsHistory Item Example (Optional):
{
    "partitionKey": "consistency",           # caseId
    "sortKey": "2025-01-01T10:00:00Z",      # timestamp
    "thresholdId": "thr-123",
    "score": 0.7,
    "previousScore": 0.6,
    "reason": "Adjusted based on performance analysis",
    "changedBy": "arn:aws:sts::...",
    "metadata": {
        "analysisRunId": "analysis-456"
    }
}
"""

# ====================
# Alternative: Single Table Design
# ====================

"""
Alternative approach using single table design with composite keys:

Table: GEvalConfiguration
- PK: entityType#entityId (e.g., "CASE#consistency", "JUDGE#expert-1", "RUN#uuid")
- SK: version or timestamp (e.g., "v1.0", "2025-01-01T10:00:00Z")

Benefits:
- Single table to manage
- Atomic transactions across entities
- Simplified backup/restore

Drawbacks:
- More complex access patterns
- Potential hot partitions
- Harder to understand for team members new to DynamoDB
"""
