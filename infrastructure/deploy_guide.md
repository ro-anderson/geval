# DynamoDB Infrastructure Deployment Guide

## Overview

This guide will deploy DynamoDB tables for the G-Eval system with "didier-" prefix for easy identification and cleanup.

## Prerequisites

### 1. Install AWS CDK CLI

```bash
npm install -g aws-cdk
```

### 2. Verify AWS Credentials

```bash
aws sts get-caller-identity
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Deployment Steps

### 1. Bootstrap CDK (One-time setup per region)

```bash
cdk bootstrap
```

### 2. Synthesize CloudFormation Template (Optional - to preview)

```bash
cdk synth
```

### 3. Deploy the Stack

```bash
cdk deploy didier-GEvalDynamoDBStack
```

### 4. Verify Deployment

```bash
aws dynamodb list-tables --query 'TableNames[?starts_with(@, `didier-`)]'
```

## Expected Resources

The deployment will create these DynamoDB tables:

- `didier-CasesConfiguration`
- `didier-JudgesConfiguration`
- `didier-EvaluationRuns`
- `didier-ThresholdsHistory`

Each table includes:

- ✅ Pay-per-request billing
- ✅ Point-in-time recovery
- ✅ Global Secondary Indexes for efficient queries
- ✅ Proper tags for organization

## Post-Deployment

### Switch Application to DynamoDB

```bash
export DATABASE_TYPE=dynamodb
export AWS_REGION=us-east-1
```

### Restart Application

```bash
uvicorn app:app --reload
```

## Cleanup (If Needed)

### Delete the Stack

```bash
cdk destroy didier-GEvalDynamoDBStack
```

### Verify Cleanup

```bash
aws dynamodb list-tables --query 'TableNames[?starts_with(@, `didier-`)]'
```

## Troubleshooting

### If CDK Bootstrap Fails

```bash
# Use explicit region and account
cdk bootstrap aws://ACCOUNT-NUMBER/REGION
```

### If Deployment Fails

```bash
# Check CloudFormation events
aws cloudformation describe-stack-events --stack-name didier-GEvalDynamoDBStack
```

### Table Access Issues

Ensure your AWS credentials have DynamoDB permissions:

- `dynamodb:CreateTable`
- `dynamodb:DescribeTable`
- `dynamodb:PutItem`
- `dynamodb:GetItem`
- `dynamodb:Query`
- `dynamodb:Scan`
- `dynamodb:UpdateItem`
- `dynamodb:DeleteItem`

## Cost Estimation

With pay-per-request billing, you only pay for what you use:

- **Writes**: $1.25 per million write request units
- **Reads**: $0.25 per million read request units
- **Storage**: $0.25 per GB-month

For development usage, expect costs under $5/month.
