#!/usr/bin/env python3

import aws_cdk as cdk
from dynamodb_stack import GEvalDynamoDBStack

app = cdk.App()

# Create the DynamoDB stack with didier- prefix
GEvalDynamoDBStack(app, "didier-GEvalDynamoDBStack",
    # Optional: Add tags for organization
    tags={
        "Project": "G-Eval",
        "Environment": "Development",
        "Owner": "didier"
    }
)

app.synth()
