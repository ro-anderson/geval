# G-Eval System DynamoDB Migration Implementation Guide

## Executive Summary

This guide outlines the migration of the G-Eval evaluation system from SQLite to AWS DynamoDB. The system must maintain full backward compatibility with the existing FastAPI (`app.py`) and Streamlit dashboard (`eval_dashboard.py`) while leveraging DynamoDB's strengths for scalability and performance.

## Project Context

### Current State
- **Database**: SQLite with normalized relational schema
- **API**: FastAPI application serving REST endpoints
- **Dashboard**: Streamlit application for visualization and management
- **Functionality**: Complete LLM evaluation system with judges, cases, metrics, and execution tracking

### Target State
- **Database**: AWS DynamoDB with denormalized schema
- **API**: Same FastAPI endpoints, updated database layer
- **Dashboard**: Same Streamlit interface, no user-facing changes
- **Functionality**: Identical features with improved scalability

## Core Architecture Decisions

### 1. Data Model Transformation

**Decision**: Move from normalized relational model to denormalized DynamoDB model

**Rationale**:
- DynamoDB performs best with denormalized data
- Reduces read operations and costs
- Eliminates complex JOIN operations
- Improves query performance for dashboard

**Implementation Requirements**:
- Embed related data instead of using foreign keys
- Store complete snapshots in evaluation runs
- Use composite keys for versioning

### 2. Snapshot Pattern for Runs

**Decision**: Each evaluation run stores complete snapshots of judge configuration and document content

**Rationale**:
- Full reproducibility of historical evaluations
- Single read operation for run details
- No dependency on referenced entities that might change
- Simplified dashboard queries

**Implementation Requirements**:
- `judgeSnapshot`: Complete judge configuration at execution time
- `evaluatedArtifact`: Full document content (not just IDs)
- All configuration parameters frozen in time

### 3. Document Management

**Decision**: Eliminate separate documents table; embed documents directly in runs

**Rationale**:
- Documents are evaluation inputs, not standalone entities
- Reduces table management overhead
- Simplifies API interface
- Aligns with actual usage pattern

**Implementation Requirements**:
- Store document content in `evaluatedArtifact` field within runs
- Rename fields: `actual_output` → `evaluationTarget`, `expected_output` → `expectedOutput`
- Include document metadata within the artifact

### 4. Model and Metric Embedding

**Decision**: Embed model and metric configurations within JudgesConfiguration

**Rationale**:
- Follows existing AgentsConfiguration pattern
- Reduces table count
- Simplifies configuration management
- Natural grouping of related settings

**Implementation Requirements**:
- `llm` field structure matching AgentsConfiguration pattern
- `metric` field for evaluation methodology
- No separate Models or Metrics tables

## DynamoDB Schema

The complete schema definition is provided in `dynamodb-schema-design-and-iac.md`. This includes:
- Table structures
- Global Secondary Indexes (GSIs)
- Infrastructure as Code (CDK)
- Example data items

### Tables Overview

1. **CasesConfiguration**: Evaluation scenarios with versioning
2. **JudgesConfiguration**: Specialized evaluators with embedded LLM/metric config
3. **EvaluationRuns**: Complete execution history with snapshots
4. **ThresholdsHistory**: Optional audit trail for threshold changes

## Implementation Requirements

### Phase 1: Database Layer (`database.py`)

Create a new `DynamoDBManager` class that provides the same interface as the current `DatabaseManager`:

**Required Methods** (must maintain same signatures):
```python
# Cases
- create_case(name, task_introduction, evaluation_criteria, ...)
- get_case(case_id)
- list_cases()
- update_case(case_id, ...)

# Judges  
- create_judge(name, model_id, case_id, metric_id, parameters, ...)
- get_judge(judge_id)
- list_judges()
- update_judge(judge_id, ...)

# Runs
- create_run(judge_id, document_id)
- update_run_status(run_id, status, error_message=None)
- complete_run(run_id, ...)
- get_run(run_id)
- list_runs(judge_id=None, limit=100)

# Metrics (now embedded in judges)
- create_metric(name)  # Store in memory or config
- get_metric(metric_id)
- list_metrics()

# Models (now embedded in judges)
- create_model(name, provider)  # Store in memory or config
- get_model(model_id)
- list_models()

# Documents (now embedded in runs)
- create_document(actual_output, expected_output, metadata)  # Return temporary ID
- get_document(document_id)  # Retrieve from temporary storage
- list_documents()  # Return empty or from temporary storage
```

**Key Implementation Details**:
- Use boto3 DynamoDB client
- Implement version management (active flag pattern)
- Handle GSI queries for filtering
- Maintain backward compatibility with existing return formats

### Phase 2: API Compatibility (`app.py`)

**No changes to endpoint signatures** - the API must continue working exactly as before

**Implementation Requirements**:
1. Replace `DatabaseManager` with `DynamoDBManager`
2. Handle the document flow:
   - When creating documents, store temporarily (in memory or cache)
   - When running evaluation, embed document in run
3. Adapt model/metric references:
   - Create virtual IDs for models/metrics
   - Map these to embedded configurations in judges

### Phase 3: Dashboard Compatibility (`eval_dashboard.py`)

**No changes to user interface** - dashboard functionality must remain identical

**Implementation Requirements**:
1. Queries should work with the same API endpoints
2. Performance should improve due to denormalized data
3. All charts and metrics continue functioning

## Migration Strategy

### Step 1: Deploy DynamoDB Tables
```bash
# Use the CDK code from dynamodb-schema-design-and-iac.md
cdk deploy GEvalDynamoDBStack
```

### Step 2: Implement DynamoDBManager
- Create new database adapter
- Implement all required methods
- Add comprehensive error handling
- Include retry logic for DynamoDB throttling

### Step 3: Data Migration (if needed)
```python
# Pseudo-code for migration
old_data = SQLiteManager.export_all()
transformed_data = transform_to_dynamodb_format(old_data)
DynamoDBManager.import_all(transformed_data)
```

### Step 4: Update Configuration
```python
# settings.py
DATABASE_TYPE = "dynamodb"  # Switch from "sqlite"
AWS_REGION = "us-east-1"
DYNAMODB_ENDPOINT = None  # Or "http://localhost:8000" for local
```

### Step 5: Testing & Validation
1. Run all existing API tests
2. Verify dashboard functionality
3. Performance benchmarking
4. Data integrity checks

## Success Criteria

The migration is successful when:

1. **Functional Parity**: All existing features work identically
2. **API Compatibility**: No changes required in API consumers
3. **Performance**: Improved query performance, especially for dashboard
4. **Scalability**: Can handle 1000x current load without degradation
5. **Cost Efficiency**: Reduced operational costs with pay-per-request model
6. **Data Integrity**: All historical data preserved and accessible

## Important Considerations

### Data Consistency
- Use DynamoDB transactions for multi-item updates
- Implement optimistic locking where needed
- Handle eventual consistency in GSI queries

### Error Handling
- Implement exponential backoff for throttled requests
- Graceful degradation for service issues
- Comprehensive logging for debugging

### Monitoring
- Set up CloudWatch alarms for table metrics
- Track consumed capacity and throttles
- Monitor API latency changes

### Security
- Use IAM roles for table access
- Implement least-privilege principles
- Encrypt data at rest and in transit

## Testing Checklist

- [ ] All API endpoints return same response format
- [ ] Dashboard displays all data correctly
- [ ] Evaluation runs complete successfully
- [ ] Historical data is accessible
- [ ] Filters and searches work
- [ ] Performance metrics improved
- [ ] No data loss during migration
- [ ] Rollback procedure tested

## Rollback Plan

If issues arise:
1. Switch `DATABASE_TYPE` back to "sqlite"
2. Restore SQLite database from backup
3. Restart services
4. Investigate issues before retry

## Summary

This migration preserves all existing functionality while leveraging DynamoDB's strengths. The key principle is **complete backward compatibility** - the API and dashboard must work exactly as before, with only the underlying database changing. The denormalized, snapshot-based approach will improve performance and scalability while maintaining data integrity and reproducibility.