# G-Eval DynamoDB Migration Guide

## Overview

Successfully migrated G-Eval system from SQLite to DynamoDB while maintaining **complete backward compatibility**. The system now supports both databases via configuration switching.

## Migration Achievements ✅

1. **Database Factory Pattern**: Created `database_factory.py` to switch between SQLite and DynamoDB
2. **DynamoDB Implementation**: Complete `DynamoDBManager` class with same interface as `DatabaseManager`
3. **Field Name Updates**: Renamed `actual_output` → `evaluationTarget`, `expected_output` → `expectedOutput` (with backward compatibility)
4. **Infrastructure as Code**: CDK templates for DynamoDB deployment
5. **Configuration Switching**: Environment variable-based database selection
6. **Full API Compatibility**: All FastAPI endpoints work identically with both databases

## Configuration

### SQLite Mode (Default)

```bash
export DATABASE_TYPE=sqlite
# or omit - defaults to sqlite
```

### DynamoDB Mode

```bash
export DATABASE_TYPE=dynamodb
export AWS_REGION=us-east-1
# Optional: for local DynamoDB
export DYNAMODB_ENDPOINT=http://localhost:8000
```

## Deployment

### DynamoDB Infrastructure

```bash
cd infrastructure/
pip install -r requirements.txt
cdk deploy GEvalDynamoDBStack
```

### Application Deployment

```bash
# Install dependencies
uv add boto3

# Set environment variables
export DATABASE_TYPE=dynamodb
export AWS_REGION=us-east-1

# Start application
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Key Features

### 1. **Snapshot Pattern**

- Complete judge configuration stored in each run
- Full reproducibility of historical evaluations
- No dependency on changing referenced entities

### 2. **Embedded Data Model**

- Models and metrics embedded within judges
- Documents embedded within runs (eliminating separate table)
- Reduced read operations and improved performance

### 3. **Historical Tracking**

- Threshold changes tracked with versioning
- Case configuration versioning support
- Complete audit trail

### 4. **Field Name Migration**

- `actual_output` → `evaluationTarget`
- `expected_output` → `expectedOutput`
- Backward compatibility maintained via Pydantic aliases

## Database Schema Comparison

### SQLite (Original)

```
metrics → models → judges ← cases ← thresholds
                     ↓
eval_documents ← runs
```

### DynamoDB (New)

```
CasesConfiguration (versioned)
JudgesConfiguration (embedded models/metrics)
EvaluationRuns (embedded documents/snapshots)
ThresholdsHistory (optional)
```

## Benefits

### Performance

- **Single-table access patterns** for most operations
- **Denormalized data** reduces JOIN complexity
- **GSI optimization** for dashboard queries

### Scalability

- **Pay-per-request** billing model
- **Auto-scaling** with demand
- **Global availability** via AWS regions

### Reliability

- **Point-in-time recovery** enabled
- **DynamoDB Streams** for event processing
- **Multi-AZ redundancy** built-in

## Testing

### API Compatibility Test

```bash
# Health check
curl -X GET "http://localhost:8000/"

# Test endpoints
curl -X GET "http://localhost:8000/cases"
curl -X GET "http://localhost:8000/judges"
curl -X GET "http://localhost:8000/runs"
```

### Streamlit Dashboard

```bash
streamlit run evals_playground.py --server.port 8501
```

## Migration Verification

All tests pass ✅:

- [x] API endpoints return same response format
- [x] Streamlit dashboard displays correctly
- [x] Field name aliases work (both old and new names accepted)
- [x] Database factory switches correctly based on config
- [x] No breaking changes to existing code

## Next Steps

1. **Production Deployment**: Deploy DynamoDB tables to production
2. **Monitoring**: Set up CloudWatch alarms and dashboards
3. **Data Migration**: If needed, migrate existing SQLite data to DynamoDB
4. **Performance Testing**: Load test with production-scale data

## Rollback Plan

If issues arise:

```bash
export DATABASE_TYPE=sqlite
# Restart services - no code changes needed
```

The system automatically falls back to SQLite with zero downtime.

---

**Status**: ✅ **Migration Complete - Fully Backward Compatible**
