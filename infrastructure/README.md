# G-Eval DynamoDB Infrastructure

## Why Node.js was mentioned?

AWS CDK CLI is distributed via npm (Node.js package manager), but **you don't need Node.js!**

We have better Python-only alternatives:

## 🐍 **Option 1: Pure Python (Recommended)**

Deploy using our custom Python script:

```bash
# Deploy tables
python deploy_dynamodb.py

# Deploy with custom prefix/region
python deploy_dynamodb.py myprefix- us-west-2

# Delete tables (cleanup)
python deploy_dynamodb.py delete didier-
```

### Advantages:

- ✅ **Pure Python** - no Node.js needed
- ✅ **Simple** - single script
- ✅ **Fast** - direct boto3 calls
- ✅ **Easy cleanup** - built-in delete function

## 🏗️ **Option 2: AWS CDK (if you prefer)**

If you want the full CDK experience:

```bash
# Install CDK (requires Node.js)
npm install -g aws-cdk

# Deploy
cdk deploy didier-GEvalDynamoDBStack
```

## 📊 **What gets created:**

All tables with `didier-` prefix:

- `didier-CasesConfiguration`
- `didier-JudgesConfiguration`
- `didier-EvaluationRuns`
- `didier-ThresholdsHistory`

Each with:

- ✅ Pay-per-request billing
- ✅ Point-in-time recovery
- ✅ Global Secondary Indexes
- ✅ Proper tags for organization

## 🚀 **Quick Start:**

```bash
# 1. Deploy DynamoDB tables
cd infrastructure
python deploy_dynamodb.py

# 2. Switch your app to DynamoDB
cd ..
export DATABASE_TYPE=dynamodb
export AWS_REGION=us-east-1

# 3. Restart your app
uvicorn app:app --reload
```

## 🧹 **Cleanup:**

```bash
# Delete all didier- tables
python deploy_dynamodb.py delete didier-
```

## 💰 **Cost:**

Pay-per-request pricing means you only pay for what you use - typically under $5/month for development.
