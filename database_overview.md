# G-Eval Database Schema Overview

## 📊 **System Architecture**

The G-Eval database is designed to support a comprehensive LLM evaluation system with historical tracking, specialized judges, and detailed telemetry. The schema follows a normalized approach to ensure data integrity, scalability, and maintainability.

## 🗂️ **Core Tables**

### **1. Cases Table**

**Purpose**: Defines evaluation scenarios/use cases that judges can evaluate.

```sql
CREATE TABLE cases (
    id TEXT PRIMARY KEY,                    -- UUID4 identifier
    name TEXT UNIQUE NOT NULL,              -- Lowercase unique name
    task_introduction TEXT NOT NULL,        -- Task description for LLM
    evaluation_criteria TEXT NOT NULL,      -- Specific evaluation criteria
    min_score INTEGER NOT NULL DEFAULT 1,  -- Minimum score range
    max_score INTEGER NOT NULL DEFAULT 5,  -- Maximum score range
    requires_reference BOOLEAN NOT NULL DEFAULT FALSE, -- Needs expected_output
    threshold_id TEXT NOT NULL REFERENCES thresholds(id), -- Current pass/fail threshold
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

**Business Logic**:

- **Cases are templates** for evaluation scenarios (e.g., "consistency", "relevance", "coherence")
- **Reusable**: Multiple judges can be created for the same case with different parameters
- **Reference flexibility**: Some cases need expected output for comparison, others don't
- **Score normalization**: Min/max scores define the evaluation scale for each case

---

### **2. Thresholds Table** ⭐ _NEW ARCHITECTURE_

**Purpose**: Historical tracking of pass/fail thresholds for cases.

```sql
CREATE TABLE thresholds (
    id TEXT PRIMARY KEY,                    -- UUID4 identifier
    score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0), -- Normalized threshold
    created_at TEXT NOT NULL               -- When threshold was set
);
```

**Business Logic**:

- **Historical tracking**: Every threshold change creates a new record
- **Audit trail**: Can track how evaluation standards evolved over time
- **Data integrity**: Never lose threshold history when standards change
- **Normalized scores**: All thresholds are 0.0-1.0 regardless of case score range

**Why Separate Table?**:

- **Future analytics**: Track threshold trends and changes over time
- **Compliance**: Maintain audit trail for evaluation standard changes
- **Flexibility**: Could support complex threshold strategies (time-based, conditional, etc.)

---

### **3. Metrics Table**

**Purpose**: Defines evaluation methodologies/algorithms.

```sql
CREATE TABLE metrics (
    id TEXT PRIMARY KEY,                    -- UUID4 identifier
    name TEXT UNIQUE NOT NULL,              -- Method name (e.g., "geval", "claude_eval")
    created_at TEXT NOT NULL
);
```

**Business Logic**:

- **Algorithm abstraction**: Separates evaluation method from implementation
- **Extensibility**: Easy to add new evaluation methods (GPT-4, Claude, custom metrics)
- **Consistency**: Same metric can be applied across different cases and judges

---

### **4. Models Table**

**Purpose**: LLM model configurations and provider information.

```sql
CREATE TABLE models (
    id TEXT PRIMARY KEY,                    -- UUID4 identifier
    name TEXT UNIQUE NOT NULL,              -- Model name (e.g., "gpt-4o-2024-08-06")
    provider TEXT NOT NULL,                 -- Provider (e.g., "openai", "anthropic")
    created_at TEXT NOT NULL
);
```

**Business Logic**:

- **Model management**: Centralized configuration for different LLMs
- **Provider tracking**: Know which provider/API to use for each model
- **Cost tracking**: Can analyze usage and costs per model
- **Experimentation**: Easy to compare performance across different models

---

### **5. Judges Table** 🏛️ _CORE ORCHESTRATION_

**Purpose**: Specialized evaluation configurations combining case, metric, and model.

```sql
CREATE TABLE judges (
    id TEXT PRIMARY KEY,                    -- UUID4 identifier
    name TEXT,                              -- Optional descriptive name
    model_id TEXT NOT NULL REFERENCES models(id),     -- Which LLM to use
    case_id TEXT NOT NULL REFERENCES cases(id),       -- What to evaluate
    metric_id TEXT NOT NULL REFERENCES metrics(id),   -- How to evaluate
    parameters TEXT NOT NULL,               -- JSON: evaluation parameters
    description TEXT,                       -- Optional description
    created_at TEXT NOT NULL
);
```

**Business Logic**:

- **Specialization**: Each judge is an expert in one case using one method
- **Parameter isolation**: Different judges can have different temperature, n_responses, etc.
- **Reusability**: Same case can have multiple judges with different configurations
- **Traceability**: Know exactly which model/method/parameters produced each result

**Why This Design?**:

- **Microservice pattern**: Each judge is a specialized evaluation service
- **A/B testing**: Compare different parameter configurations on same case
- **Scalability**: Can run multiple judges in parallel for same evaluation
- **Historical consistency**: Judge configuration is frozen in time

---

### **6. Eval_Documents Table**

**Purpose**: Text documents to be evaluated.

```sql
CREATE TABLE eval_documents (
    id TEXT PRIMARY KEY,                    -- UUID4 identifier
    actual_output TEXT NOT NULL,            -- Text being evaluated
    expected_output TEXT,                   -- Optional reference text
    metadata TEXT,                          -- JSON: additional metadata
    created_at TEXT NOT NULL
);
```

**Business Logic**:

- **Input management**: Store all evaluation inputs centrally
- **Reusability**: Same document can be evaluated by multiple judges
- **Reference flexibility**: Support both reference-based and reference-free evaluation
- **Metadata tracking**: Store source, context, or other relevant information

---

### **7. Runs Table** 🚀 _EXECUTION & TELEMETRY_

**Purpose**: Individual evaluation executions with complete telemetry.

```sql
CREATE TABLE runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Sequential ID
    judge_id TEXT NOT NULL REFERENCES judges(id),     -- Which judge ran
    document_id TEXT NOT NULL REFERENCES eval_documents(id), -- What was evaluated
    status TEXT NOT NULL,                   -- pending/running/completed/failed
    evaluation_status TEXT,                 -- pass/fail/pending (based on threshold)
    final_score REAL,                      -- Average score
    final_score_normalized REAL,          -- Normalized score (0-1)
    all_responses TEXT,                    -- JSON: All individual responses
    total_usage_tokens INTEGER,           -- Total tokens used
    prompt_tokens INTEGER,                -- Prompt tokens
    completion_tokens INTEGER,            -- Completion tokens
    execution_time_seconds REAL,          -- How long it took
    prompt_used TEXT,                     -- Actual prompt sent to LLM
    error_message TEXT,                   -- Error details if failed
    metadata TEXT,                        -- JSON: Additional execution metadata
    started_at TEXT NOT NULL,             -- When execution started
    completed_at TEXT,                    -- When execution finished
    created_at TEXT NOT NULL
);
```

**Business Logic**:

- **Complete telemetry**: Track every aspect of evaluation execution
- **Cost analysis**: Monitor token usage and execution time
- **Quality control**: Store all individual responses for analysis
- **Debugging**: Full execution context for troubleshooting
- **Compliance**: Complete audit trail of evaluation process

---

## 🔗 **Key Relationships**

### **1. Cases ↔ Thresholds (1:1 Current, 1:Many Historical)**

```
cases.threshold_id → thresholds.id
```

**Why**: Historical tracking of threshold changes while maintaining current threshold reference.

**Benefits**:

- Can analyze how evaluation standards evolved
- Maintain audit trail for compliance
- Support future complex threshold strategies

---

### **2. Judges ↔ Cases (Many:1)**

```
judges.case_id → cases.id
```

**Why**: Multiple judges can evaluate the same case with different configurations.

**Benefits**:

- A/B test different parameters on same evaluation scenario
- Specialist judges (e.g., "Conservative Consistency Judge", "Strict Consistency Judge")
- Gradual rollout of new evaluation approaches

---

### **3. Judges ↔ Metrics (Many:1)**

```
judges.metric_id → metrics.id
```

**Why**: Same evaluation method can be applied to different cases and configurations.

**Benefits**:

- Consistent evaluation methodology across cases
- Easy to migrate from one metric to another
- Compare different evaluation algorithms

---

### **4. Judges ↔ Models (Many:1)**

```
judges.model_id → models.id
```

**Why**: Same model can be used by multiple judges, and models can be upgraded/changed.

**Benefits**:

- Model performance comparison across evaluation tasks
- Easy model upgrades without losing judge configuration
- Cost optimization by model selection

---

### **5. Runs ↔ Judges (Many:1)**

```
runs.judge_id → judges.id
```

**Why**: Track which specific judge configuration produced each result.

**Benefits**:

- Performance analysis per judge
- Parameter optimization based on historical results
- Judge-specific quality metrics

---

### **6. Runs ↔ Documents (Many:1)**

```
runs.document_id → eval_documents.id
```

**Why**: Same document can be evaluated multiple times by different judges.

**Benefits**:

- Inter-judge reliability analysis
- Document difficulty scoring
- Judge comparison on same inputs

---

## 📈 **Data Flow & Evaluation Process**

### **Evaluation Execution Flow**:

1. **Document** created in `eval_documents`
2. **Judge** selected (which defines case, metric, model, parameters)
3. **Run** created in `runs` table (status: pending)
4. **Evaluation executed** using judge configuration
5. **Results stored** in `runs` with complete telemetry
6. **Evaluation status** calculated using case's current threshold

### **Threshold Evaluation Logic**:

```sql
-- Get threshold for evaluation status calculation
SELECT t.score
FROM runs r
JOIN judges j ON r.judge_id = j.id
JOIN cases c ON j.case_id = c.id
JOIN thresholds t ON c.threshold_id = t.id
WHERE r.id = ?
```

---

## 🎯 **Design Principles**

### **1. Separation of Concerns**

- **Cases**: What to evaluate
- **Metrics**: How to evaluate
- **Models**: Which LLM to use
- **Judges**: Specific configuration combining the above
- **Runs**: Execution results and telemetry

### **2. Historical Preservation**

- **Immutable records**: Judges, thresholds, and runs are never modified after creation
- **Audit trail**: Complete history of evaluation standards and configurations
- **Reproducibility**: Can recreate exact evaluation conditions from any point in time

### **3. Flexibility & Extensibility**

- **New metrics**: Add evaluation methods without schema changes
- **New models**: Support any LLM provider through model abstraction
- **Parameter evolution**: Judge parameters can be modified while preserving history
- **Threshold strategies**: Architecture supports complex threshold rules

### **4. Performance & Analytics**

- **Efficient queries**: Normalized structure supports complex analytics
- **Telemetry focus**: Rich data collection for optimization and debugging
- **Cost tracking**: Token usage and execution time monitoring
- **Quality metrics**: Multiple evaluation dimensions captured

---

## 🚀 **Usage Patterns**

### **Creating an Evaluation System**:

1. Define **metrics** (evaluation methods)
2. Create **models** (LLM configurations)
3. Design **cases** (evaluation scenarios)
4. Configure **judges** (specialized evaluators)
5. Upload **documents** (content to evaluate)
6. Execute **runs** (perform evaluations)

### **Analytics & Insights**:

- **Judge performance**: Compare accuracy, consistency, speed across judges
- **Document difficulty**: Identify challenging content based on score distributions
- **Model comparison**: Analyze performance differences between LLMs
- **Threshold optimization**: Historical analysis to find optimal pass/fail thresholds
- **Cost analysis**: Token usage and execution time patterns

### **Quality Assurance**:

- **Inter-judge reliability**: Multiple judges evaluating same documents
- **Threshold validation**: Historical threshold changes and their impact
- **Parameter optimization**: A/B testing different judge configurations
- **Error analysis**: Failed runs and their patterns

---

## 🔧 **Technical Notes**

### **Foreign Key Constraints**:

All relationships enforced at database level for data integrity.

### **JSON Fields**:

- `judges.parameters`: Evaluation configuration (temperature, n_responses, etc.)
- `runs.all_responses`: Individual evaluation scores from multiple API calls
- `runs.metadata`: Execution context and debugging information
- `eval_documents.metadata`: Document source and context information

### **UUID vs Auto-increment**:

- **UUIDs**: Used for business entities (cases, judges, models, etc.) for distributed system compatibility
- **Auto-increment**: Used for runs (sequential logging entity) for performance

### **Normalization Level**:

The schema is in **3NF (Third Normal Form)** with intentional denormalization in the `runs` table for telemetry and performance analysis.

---

_This database design supports a production-ready LLM evaluation system with comprehensive tracking, historical analysis, and extensible architecture for future evaluation methodologies._
