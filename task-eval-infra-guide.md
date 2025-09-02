# Tako AI Eval System Infrastructure Guide - DynamoDB Resources

## Overview

This document provides specifications for creating the Tako AI Eval system DynamoDB infrastructure using CDK/TypeScript. The system is designed as a scalable evaluation platform for LLM outputs with full traceability and historical data preservation.

## Table of Contents

1. [DynamoDB Tables Design](#dynamodb-tables-design) - Complete schemas and structures
2. [API Context & Usage](#api-context--usage) - Endpoint context for query patterns
3. [Data Relationships](#data-relationships) - Entity relationships and access patterns
4. [CDK Implementation Requirements](#cdk-implementation-requirements) - Technical specifications for CDK
5. [Future Steps](#future-steps) - Implementation phases and roadmap

---

## System Architecture Principles

The Tako AI Eval DynamoDB design follows these principles:

1. **Denormalized Data Model**: Optimized for DynamoDB single-table patterns with embedded configurations
2. **Snapshot Pattern**: Complete configuration snapshots stored in evaluation runs for reproducibility
3. **Historical Tracking**: Immutable records for threshold changes and audit trails
4. **Efficient Querying**: GSIs designed for common access patterns

---

## DynamoDB Tables Design

### Table Naming Convention

Tables should follow the project's naming patterns and conventions. The system uses CamelCase naming:

- `CasesConfiguration` _(includes embedded threshold history)_
- `JudgesConfiguration`
- `EvaluationRuns`

**Important**: The current implementation uses a "didier-" prefix for all table names (e.g., `didier-CasesConfiguration`) for testing and identification purposes. For production deployment, remove this prefix and use the base table names. Apply appropriate prefixes or naming conventions based on your project requirements.

### 1. CasesConfiguration

**Purpose**: Stores evaluation case definitions with embedded threshold history for complete audit trail.

```json
{
  "partitionKey": "case-uuid",
  "sortKey": "1.0",
  "active": "true",
  "caseId": "case-uuid",
  "name": "consistency over news article summary",
  "taskIntroduction": "You will be given...",
  "evaluationCriteria": "Evaluate based on...",
  "scoreRange": {
    "min": 1,
    "max": 5
  },
  "requiresReference": false,
  "currentThreshold": 0.5,
  "thresholdHistory": [
    {
      "thresholdId": "threshold-uuid-1",
      "score": 0.5,
      "createdBy": "system",
      "reason": "Initial case creation",
      "createdAt": "2024-01-01T00:00:00Z"
    },
    {
      "thresholdId": "threshold-uuid-2",
      "score": 0.32,
      "createdBy": "user-123",
      "reason": "Performance optimization",
      "createdAt": "2024-01-15T10:30:00Z"
    }
  ],
  "createdOn": "2024-01-01T00:00:00Z",
  "lastUpdatedOn": "2024-01-15T10:30:00Z"
}
```

**Key Features**:

- Embedded score range configuration
- Boolean flag for reference requirement
- Current threshold for quick access
- Complete threshold history as embedded list
- Full audit trail with user attribution
- Atomic threshold updates

**Global Secondary Indexes**:

- `ActiveCasesIndex`: `active` (HASH) + `caseId` (RANGE) - enables filtering active cases

### 2. JudgesConfiguration

**Purpose**: Stores judge configurations with embedded LLM and evaluation parameters.

```json
{
  "partitionKey": "judge-uuid",
  "sortKey": "1.0",
  "active": "true",
  "judgeId": "judge-uuid",
  "name": "Hot Judge",
  "description": "High-temperature creative evaluator",
  "caseId": "case-uuid",
  "caseName": "consistency over news article summary",

  "metric": {
    "id": "metric-uuid",
    "type": "geval",
    "name": "Tako AI Eval Metric"
  },

  "llm": {
    "id": "model-uuid",
    "type": "openai",
    "model": "gpt-4o-2024-08-06",
    "temperature": 2.0,
    "maxTokens": 2500,
    "topP": 1.0,
    "frequencyPenalty": 0.0,
    "presencePenalty": 0.0
  },

  "evaluationParameters": {
    "nResponses": 10,
    "sleepTime": 0.0,
    "rateLimitSleep": 0.0,
    "retryAttempts": 3
  },

  "createdOn": "2024-01-01T00:00:00Z",
  "lastUpdatedOn": "2024-01-01T00:00:00Z"
}
```

**Key Features**:

- Embedded metric, LLM, and evaluation configurations
- Denormalized case information for efficient reads
- All numeric parameters stored as Decimal for DynamoDB compatibility
- Version control through sortKey
- Score range inherited from case configuration

**Global Secondary Indexes**:

- `ActiveJudgesIndex`: `active` (HASH) + `judgeId` (RANGE) - enables filtering active judges
- `JudgesByCaseIndex`: `caseId` (HASH) + `active` (RANGE) - enables getting judges for a specific case

### 3. EvaluationRuns

**Purpose**: Stores evaluation execution records with complete snapshots for reproducibility.

```json
{
  "partitionKey": "run-timestamp-uuid",
  "sortKey": "METADATA",
  "runId": 123456789,
  "judgeId": "judge-uuid",
  "status": "completed",
  "evaluationStatus": "pass",

  "evaluatedArtifact": {
    "evaluationTarget": "The actual text being evaluated...",
    "expectedOutput": "Optional reference text...",
    "metadata": {}
  },

  "judgeSnapshot": {
    "judgeId": "judge-uuid",
    "judgeName": "Hot Judge",
    "version": "1.0",
    "caseId": "case-uuid",
    "caseName": "consistency over news article summary",
    "metric": {
      "type": "geval",
      "name": "Tako AI Eval Metric"
    },
    "llm": {
      "type": "openai",
      "model": "gpt-4o-2024-08-06",
      "temperature": 2.0
    },
    "evaluationParameters": {
      "nResponses": 10
    },
    "scoreRange": {
      "min": 1,
      "max": 5
    }
  },

  "results": {
    "finalScore": 3.2,
    "finalScoreNormalized": 0.55,
    "allResponses": [3.0, 3.1, 3.5, 3.2, 3.0],
    "executionTimeSeconds": 12.5,
    "promptUsed": "Complete evaluation prompt text...",
    "tokenUsage": {
      "prompt": 800,
      "completion": 700,
      "total": 1500
    }
  },

  "runDate": "2024-01-01",
  "documentId": null,

  "timestamps": {
    "startedAt": "2024-01-01T00:00:00Z",
    "completedAt": "2024-01-01T00:00:00Z",
    "createdAt": "2024-01-01T00:00:00Z"
  }
}
```

**Key Features**:

- Complete judge configuration snapshot for reproducibility
- Embedded content in `evaluatedArtifact` eliminates need for separate documents table
- Comprehensive performance and token usage metrics
- Evaluation status calculated based on case threshold
- Null `documentId` when using direct content input

**Global Secondary Indexes**:

- `RunsByJudgeIndex`: `judgeId` (HASH) + `startedAt` (RANGE) - enables getting runs by specific judge
- `RunsByCaseIndex`: `caseId` (HASH) + `startedAt` (RANGE) - enables getting runs by specific case
- `RunsByDateIndex`: `runDate` (HASH) + `startedAt` (RANGE) - enables date-based filtering
- `RunsByStatusIndex`: `status` (HASH) + `startedAt` (RANGE) - enables status-based filtering

---

## Global Secondary Indexes (GSIs) Deep Dive

GSIs are **critical** for efficient DynamoDB access patterns. All deployed GSIs are **ACTIVE** and support the dashboard and API functionality.

### CasesConfiguration GSIs

| **Index Name**     | **Partition Key** | **Sort Key**      | **Purpose**                     | **API Usage**                                    |
| ------------------ | ----------------- | ----------------- | ------------------------------- | ------------------------------------------------ |
| `ActiveCasesIndex` | `active` (String) | `caseId` (String) | Filter active cases efficiently | Dashboard case filters, case selection dropdowns |

**Query Examples**:

```python
# Get all active cases (Dashboard)
response = dynamodb.query(
    IndexName='ActiveCasesIndex',
    KeyConditionExpression=Key('active').eq('true')
)
```

### JudgesConfiguration GSIs

| **Index Name**      | **Partition Key** | **Sort Key**       | **Purpose**                  | **API Usage**                       |
| ------------------- | ----------------- | ------------------ | ---------------------------- | ----------------------------------- |
| `ActiveJudgesIndex` | `active` (String) | `judgeId` (String) | Filter active judges         | Judge management, metrics filtering |
| `JudgesByCaseIndex` | `caseId` (String) | `active` (String)  | Get judges for specific case | Evaluation page judge selection     |

**Query Examples**:

```python
# Get all judges for a specific case (Evaluation page)
response = dynamodb.query(
    IndexName='JudgesByCaseIndex',
    KeyConditionExpression=Key('caseId').eq(case_id) & Key('active').eq('true')
)

# Get all active judges (Judge management)
response = dynamodb.query(
    IndexName='ActiveJudgesIndex',
    KeyConditionExpression=Key('active').eq('true')
)
```

### EvaluationRuns GSIs

| **Index Name**      | **Partition Key**  | **Sort Key**         | **Purpose**            | **API Usage**                               |
| ------------------- | ------------------ | -------------------- | ---------------------- | ------------------------------------------- |
| `RunsByJudgeIndex`  | `judgeId` (String) | `startedAt` (String) | Get runs by judge      | Dashboard analytics, judge-specific results |
| `RunsByCaseIndex`   | `caseId` (String)  | `startedAt` (String) | Get runs by case       | Case performance analytics                  |
| `RunsByDateIndex`   | `runDate` (String) | `startedAt` (String) | Date-based filtering   | Dashboard date filters                      |
| `RunsByStatusIndex` | `status` (String)  | `startedAt` (String) | Status-based filtering | Failed/completed run analysis               |

**Query Examples**:

```python
# Get all runs for a specific judge (Dashboard analytics)
response = dynamodb.query(
    IndexName='RunsByJudgeIndex',
    KeyConditionExpression=Key('judgeId').eq(judge_id)
)

# Get runs by date range (Dashboard filters)
response = dynamodb.query(
    IndexName='RunsByDateIndex',
    KeyConditionExpression=Key('runDate').eq('2024-01-01')
)

# Get failed runs for analysis
response = dynamodb.query(
    IndexName='RunsByStatusIndex',
    KeyConditionExpression=Key('status').eq('failed')
)
```

### GSI Implementation Requirements

**All GSIs Must**:

- Use `PAY_PER_REQUEST` billing mode (matches table)
- Have same projection type as main table
- Support eventual consistency for reads
- Handle sparse indexes gracefully (some items may not have GSI attributes)

**Critical for Dashboard Performance**:

- `JudgesByCaseIndex` enables **case → judges** filtering in evaluation UI
- `RunsByJudgeIndex` enables **judge-specific analytics** in dashboard
- `RunsByDateIndex` enables **date-based filtering** for large datasets
- `ActiveCasesIndex` & `ActiveJudgesIndex` enable **active-only** filtering

---

## API Context & Usage

_The following endpoints provide context for how the DynamoDB tables will be accessed. This information helps understand the query patterns needed for GSI design._

### Metrics Endpoints

| Endpoint        | Method | Purpose                         | Dashboard Usage                      |
| --------------- | ------ | ------------------------------- | ------------------------------------ |
| `/metrics`      | POST   | Create evaluation methodologies | Judges Management - metric selection |
| `/metrics`      | GET    | List all metrics                | Dashboard filters, Judges Management |
| `/metrics/{id}` | GET    | Get specific metric             | Judge details, validation            |

### Models Endpoints

| Endpoint       | Method | Purpose                   | Dashboard Usage                     |
| -------------- | ------ | ------------------------- | ----------------------------------- |
| `/models`      | POST   | Create LLM configurations | Models tab - model creation         |
| `/models`      | GET    | List all models           | Judges Management - model selection |
| `/models/{id}` | GET    | Get specific model        | Judge details, updates              |

### Cases Endpoints

| Endpoint      | Method | Purpose                   | Dashboard Usage                                 |
| ------------- | ------ | ------------------------- | ----------------------------------------------- |
| `/cases`      | POST   | Create evaluation cases   | Cases & Metrics - case creation                 |
| `/cases`      | GET    | List all cases            | Dashboard filters, Evaluations - case selection |
| `/cases/{id}` | GET    | Get specific case         | Case details, validation                        |
| `/cases/{id}` | PUT    | Update case configuration | Cases & Metrics - case updates                  |

### Judges Endpoints

| Endpoint       | Method | Purpose                   | Dashboard Usage                                    |
| -------------- | ------ | ------------------------- | -------------------------------------------------- |
| `/judges`      | POST   | Create specialized judges | Judges Management - judge creation                 |
| `/judges`      | GET    | List all judges           | Dashboard analytics, Evaluations - judge selection |
| `/judges/{id}` | PUT    | Update judge parameters   | Judges Management - parameter tuning               |

### Documents Endpoints

| Endpoint     | Method | Purpose                     | Dashboard Usage                  |
| ------------ | ------ | --------------------------- | -------------------------------- |
| `/documents` | POST   | Create evaluation documents | Documents tab - content storage  |
| `/documents` | GET    | List all documents          | Evaluations - document selection |

### Evaluation Endpoints

| Endpoint     | Method | Purpose                  | Dashboard Usage                      |
| ------------ | ------ | ------------------------ | ------------------------------------ |
| `/eval`      | POST   | Execute evaluations      | Evaluations - run assessments        |
| `/runs`      | GET    | List evaluation runs     | Dashboard analytics, Results viewing |
| `/runs/{id}` | GET    | Get specific run details | Detailed run analysis                |

### Thresholds Endpoints

| Endpoint                 | Method | Purpose                    | Dashboard Usage                                        |
| ------------------------ | ------ | -------------------------- | ------------------------------------------------------ |
| `/thresholds`            | GET    | List threshold records     | System monitoring                                      |
| `/cases/{id}/thresholds` | GET    | Get case threshold history | Cases & Metrics - threshold history (embedded in case) |

---

## Data Relationships

### Key Relationships

- **Cases ↔ Judges**: One-to-many (case can have multiple judges)
- **Judges ↔ Runs**: One-to-many (judge can have multiple runs)
- **Cases**: Contain embedded threshold history for audit trail
- **Runs**: Contain complete snapshots (judge + case + content)

### Query Patterns

**Primary Access Patterns**:

1. List all active cases → `CasesConfiguration` scan with `active=true`
2. Get judges for a case → `JudgesConfiguration.JudgesByCaseIndex` by `caseId`
3. List runs by judge → `EvaluationRuns.RunsByJudgeIndex` by `judgeId`
4. Get threshold history → `CasesConfiguration` direct item access (embedded `thresholdHistory` list)
5. Filter runs by status → `EvaluationRuns.RunsByStatusIndex` by `status`
6. Filter runs by date → `EvaluationRuns.RunsByDateIndex` by `runDate`
7. Get active judges → `JudgesConfiguration.ActiveJudgesIndex` by `active=true`
8. Get active cases → `CasesConfiguration.ActiveCasesIndex` by `active=true`

---

## CDK Implementation Requirements

### Table Configuration

**Required for all tables**:

- **Billing Mode**: `PAY_PER_REQUEST` for flexible scaling
- **Point-in-Time Recovery**: Enabled for data protection
- **Deletion Protection**: Enabled for production environments
- **Tags**: Apply consistent project tags

### Global Secondary Indexes (GSIs) Summary

**All GSIs are ACTIVE and critical for performance**:

**CasesConfiguration** (1 GSI):

- `ActiveCasesIndex`: `active` (HASH) + `caseId` (RANGE)

**JudgesConfiguration** (2 GSIs):

- `JudgesByCaseIndex`: `caseId` (HASH) + `active` (RANGE)
- `ActiveJudgesIndex`: `active` (HASH) + `judgeId` (RANGE)

**EvaluationRuns** (4 GSIs):

- `RunsByJudgeIndex`: `judgeId` (HASH) + `startedAt` (RANGE)
- `RunsByCaseIndex`: `caseId` (HASH) + `startedAt` (RANGE)
- `RunsByDateIndex`: `runDate` (HASH) + `startedAt` (RANGE)
- `RunsByStatusIndex`: `status` (HASH) + `startedAt` (RANGE)

**Total: 7 GSIs** across 3 tables enabling efficient dashboard and API operations.

### Data Types

**Critical**: All numeric fields must support `Decimal` type for DynamoDB compatibility:

**Decimal Fields**:

- `temperature`, `topP`, `frequencyPenalty`, `presencePenalty` → Decimal
- `finalScore`, `finalScoreNormalized`, `currentThreshold` → Decimal
- `executionTimeSeconds`, `sleepTime`, `rateLimitSleep` → Decimal
- All threshold history `score` values → Decimal

**Number Fields** (integers):

- `nResponses`, `maxTokens`, `minScore`, `maxScore` → Number
- Token usage: `prompt`, `completion`, `total` → Number

**String Fields**:

- `active` attribute → String ("true"/"false") for GSI compatibility
- All timestamp fields → ISO string format
- All UUID fields → String format

**Additional Fields**:

- `versionTag`: String ("1.0") - version control for configurations
- `runDate`: String (YYYY-MM-DD format) for date-based queries and partitioning
- `documentId`: Optional field, can be null for direct content evaluations (compatibility with legacy structure)

---

## Future Steps

### Phase 1: Infrastructure

1. **CDK Stack Creation**: Implement DynamoDB tables with TypeScript CDK
2. **IAM Policies**: Configure appropriate access permissions
3. **Monitoring Setup**: CloudWatch alarms and dashboards

### Phase 2: Application Integration

1. **Database Manager**: Connect existing FastAPI application to DynamoDB
2. **Data Migration**: Transfer existing evaluation data if needed
3. **Testing**: Validate all API endpoints work with new infrastructure

### Phase 3: Enhancement

1. **Performance Optimization**: Monitor and tune GSI usage
2. **Cost Optimization**: Analyze usage patterns and adjust billing
3. **Backup Strategy**: Implement comprehensive backup procedures
4. **Multi-Environment**: Deploy to staging/production environments

### Phase 4: Advanced Features

1. **DynamoDB Streams**: Real-time processing of evaluation events
2. **Analytics Pipeline**: Enhanced reporting and metrics collection
3. **Caching Layer**: ElastiCache integration for frequently accessed data
4. **Cross-Region Replication**: Global availability and disaster recovery

---

## Summary

This guide provides the complete specifications needed to implement the Tako AI Eval DynamoDB infrastructure using CDK/TypeScript. The focus is on creating three core tables with proper GSIs, billing configurations, and data type handling.

**Key Implementation Focus**:

- ✅ **Table Schemas**: Complete JSON structures for all three optimized tables
- ✅ **GSI Configuration**: 7 critical indexes across 3 tables for efficient query patterns
- ✅ **Data Types**: Critical Decimal handling for DynamoDB compatibility
- ✅ **Access Patterns**: Query patterns to guide GSI design
- ✅ **CDK Requirements**: Technical specifications for TypeScript implementation
- ✅ **Embedded Design**: Threshold history embedded in CasesConfiguration for optimal access patterns and atomic updates
- ✅ **Real Implementation**: Updated based on actual deployed infrastructure and tested data structures

The resulting infrastructure will support a scalable evaluation platform with full audit trails and historical data preservation using DynamoDB best practices.
