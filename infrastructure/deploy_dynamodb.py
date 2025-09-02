#!/usr/bin/env python3
"""
Deploy DynamoDB tables for G-Eval system using boto3 (no CDK/Node.js needed)
"""

import boto3
import json
from botocore.exceptions import ClientError


def create_dynamodb_tables(region='us-east-1', prefix='didier-'):
    """
    Create DynamoDB tables for G-Eval system with specified prefix
    """
    dynamodb = boto3.client('dynamodb', region_name=region)
    
    tables_created = []
    
    # 1. CasesConfiguration Table
    try:
        print(f"Creating {prefix}CasesConfiguration table...")
        dynamodb.create_table(
            TableName=f'{prefix}CasesConfiguration',
            KeySchema=[
                {'AttributeName': 'partitionKey', 'KeyType': 'HASH'},
                {'AttributeName': 'sortKey', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'partitionKey', 'AttributeType': 'S'},
                {'AttributeName': 'sortKey', 'AttributeType': 'S'},
                {'AttributeName': 'active', 'AttributeType': 'S'},
                {'AttributeName': 'caseId', 'AttributeType': 'S'}
            ],
            BillingMode='PAY_PER_REQUEST',
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'ActiveCasesIndex',
                    'KeySchema': [
                        {'AttributeName': 'active', 'KeyType': 'HASH'},
                        {'AttributeName': 'caseId', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                }
            ],

            Tags=[
                {'Key': 'Project', 'Value': 'G-Eval'},
                {'Key': 'Environment', 'Value': 'Development'},
                {'Key': 'Owner', 'Value': 'didier'}
            ]
        )
        tables_created.append(f'{prefix}CasesConfiguration')
        
        # Enable point-in-time recovery
        try:
            dynamodb.update_continuous_backups(
                TableName=f'{prefix}CasesConfiguration',
                PointInTimeRecoverySpecification={'PointInTimeRecoveryEnabled': True}
            )
        except ClientError:
            pass  # May not be available in all regions
            
        print("✅ CasesConfiguration created")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUseException':
            print(f"⚠️  {prefix}CasesConfiguration already exists")
        else:
            print(f"❌ Error creating CasesConfiguration: {e}")
    
    # 2. JudgesConfiguration Table
    try:
        print(f"Creating {prefix}JudgesConfiguration table...")
        dynamodb.create_table(
            TableName=f'{prefix}JudgesConfiguration',
            KeySchema=[
                {'AttributeName': 'partitionKey', 'KeyType': 'HASH'},
                {'AttributeName': 'sortKey', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'partitionKey', 'AttributeType': 'S'},
                {'AttributeName': 'sortKey', 'AttributeType': 'S'},
                {'AttributeName': 'caseId', 'AttributeType': 'S'},
                {'AttributeName': 'active', 'AttributeType': 'S'},
                {'AttributeName': 'judgeId', 'AttributeType': 'S'}
            ],
            BillingMode='PAY_PER_REQUEST',
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'JudgesByCaseIndex',
                    'KeySchema': [
                        {'AttributeName': 'caseId', 'KeyType': 'HASH'},
                        {'AttributeName': 'active', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'ActiveJudgesIndex',
                    'KeySchema': [
                        {'AttributeName': 'active', 'KeyType': 'HASH'},
                        {'AttributeName': 'judgeId', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                }
            ],

            Tags=[
                {'Key': 'Project', 'Value': 'G-Eval'},
                {'Key': 'Environment', 'Value': 'Development'},
                {'Key': 'Owner', 'Value': 'didier'}
            ]
        )
        tables_created.append(f'{prefix}JudgesConfiguration')
        
        # Enable point-in-time recovery
        try:
            dynamodb.update_continuous_backups(
                TableName=f'{prefix}JudgesConfiguration',
                PointInTimeRecoverySpecification={'PointInTimeRecoveryEnabled': True}
            )
        except ClientError:
            pass  # May not be available in all regions
            
        print("✅ JudgesConfiguration created")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUseException':
            print(f"⚠️  {prefix}JudgesConfiguration already exists")
        else:
            print(f"❌ Error creating JudgesConfiguration: {e}")
    
    # 3. EvaluationRuns Table
    try:
        print(f"Creating {prefix}EvaluationRuns table...")
        dynamodb.create_table(
            TableName=f'{prefix}EvaluationRuns',
            KeySchema=[
                {'AttributeName': 'partitionKey', 'KeyType': 'HASH'},
                {'AttributeName': 'sortKey', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'partitionKey', 'AttributeType': 'S'},
                {'AttributeName': 'sortKey', 'AttributeType': 'S'},
                {'AttributeName': 'judgeId', 'AttributeType': 'S'},
                {'AttributeName': 'startedAt', 'AttributeType': 'S'},
                {'AttributeName': 'caseId', 'AttributeType': 'S'},
                {'AttributeName': 'runDate', 'AttributeType': 'S'},
                {'AttributeName': 'status', 'AttributeType': 'S'}
            ],
            BillingMode='PAY_PER_REQUEST',
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'RunsByJudgeIndex',
                    'KeySchema': [
                        {'AttributeName': 'judgeId', 'KeyType': 'HASH'},
                        {'AttributeName': 'startedAt', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'RunsByCaseIndex',
                    'KeySchema': [
                        {'AttributeName': 'caseId', 'KeyType': 'HASH'},
                        {'AttributeName': 'startedAt', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'RunsByDateIndex',
                    'KeySchema': [
                        {'AttributeName': 'runDate', 'KeyType': 'HASH'},
                        {'AttributeName': 'startedAt', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                },
                {
                    'IndexName': 'RunsByStatusIndex',
                    'KeySchema': [
                        {'AttributeName': 'status', 'KeyType': 'HASH'},
                        {'AttributeName': 'startedAt', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                }
            ],
            StreamSpecification={
                'StreamEnabled': True,
                'StreamViewType': 'NEW_AND_OLD_IMAGES'
            },

            Tags=[
                {'Key': 'Project', 'Value': 'G-Eval'},
                {'Key': 'Environment', 'Value': 'Development'},
                {'Key': 'Owner', 'Value': 'didier'}
            ]
        )
        tables_created.append(f'{prefix}EvaluationRuns')
        
        # Enable point-in-time recovery
        try:
            dynamodb.update_continuous_backups(
                TableName=f'{prefix}EvaluationRuns',
                PointInTimeRecoverySpecification={'PointInTimeRecoveryEnabled': True}
            )
        except ClientError:
            pass  # May not be available in all regions
            
        print("✅ EvaluationRuns created")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUseException':
            print(f"⚠️  {prefix}EvaluationRuns already exists")
        else:
            print(f"❌ Error creating EvaluationRuns: {e}")
    return tables_created


def wait_for_tables(table_names, region='us-east-1'):
    """Wait for tables to become active"""
    dynamodb = boto3.client('dynamodb', region_name=region)
    
    for table_name in table_names:
        print(f"Waiting for {table_name} to become active...")
        waiter = dynamodb.get_waiter('table_exists')
        try:
            waiter.wait(TableName=table_name, WaiterConfig={'Delay': 2, 'MaxAttempts': 30})
            print(f"✅ {table_name} is active")
        except Exception as e:
            print(f"❌ Error waiting for {table_name}: {e}")


def list_tables_with_prefix(prefix='didier-', region='us-east-1'):
    """List all tables with the specified prefix"""
    dynamodb = boto3.client('dynamodb', region_name=region)
    
    try:
        response = dynamodb.list_tables()
        tables = [name for name in response['TableNames'] if name.startswith(prefix)]
        return tables
    except ClientError as e:
        print(f"❌ Error listing tables: {e}")
        return []


def delete_tables_with_prefix(prefix='didier-', region='us-east-1'):
    """Delete all tables with the specified prefix"""
    dynamodb = boto3.client('dynamodb', region_name=region)
    
    tables = list_tables_with_prefix(prefix, region)
    
    if not tables:
        print(f"No tables found with prefix '{prefix}'")
        return
    
    print(f"Found {len(tables)} tables to delete:")
    for table in tables:
        print(f"  - {table}")
    
    confirm = input("\nAre you sure you want to delete these tables? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Deletion cancelled")
        return
    
    for table_name in tables:
        try:
            print(f"Deleting {table_name}...")
            dynamodb.delete_table(TableName=table_name)
            print(f"✅ {table_name} deletion initiated")
        except ClientError as e:
            print(f"❌ Error deleting {table_name}: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'delete':
        # Delete tables
        prefix = sys.argv[2] if len(sys.argv) > 2 else 'didier-'
        region = sys.argv[3] if len(sys.argv) > 3 else 'us-east-1'
        delete_tables_with_prefix(prefix, region)
    else:
        # Create tables
        prefix = sys.argv[1] if len(sys.argv) > 1 else 'didier-'
        region = sys.argv[2] if len(sys.argv) > 2 else 'us-east-1'
        
        print(f"🚀 Creating DynamoDB tables with prefix '{prefix}' in region '{region}'")
        print(f"🔑 Using AWS credentials for account: {boto3.Session().get_credentials().access_key[:8]}...")
        
        tables_created = create_dynamodb_tables(region, prefix)
        
        if tables_created:
            print(f"\n⏳ Waiting for {len(tables_created)} tables to become active...")
            wait_for_tables(tables_created, region)
            
            print(f"\n🎉 Deployment complete! Created tables:")
            for table in tables_created:
                print(f"  ✅ {table}")
            
            print(f"\n📋 To use DynamoDB, set these environment variables:")
            print(f"export DATABASE_TYPE=dynamodb")
            print(f"export AWS_REGION={region}")
        else:
            print("\n⚠️  No new tables were created (they might already exist)")
        
        # List all tables with prefix
        all_tables = list_tables_with_prefix(prefix, region)
        if all_tables:
            print(f"\n📊 All tables with prefix '{prefix}':")
            for table in all_tables:
                print(f"  - {table}")
