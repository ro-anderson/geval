#!/usr/bin/env python3
"""
Database creation script for G-Eval FastAPI application.

This script creates the SQLite database and all required tables for the evaluation system.
"""

import sqlite3
import uuid
from datetime import datetime
import json
import os


DATABASE_PATH = "geval_app.db"


def create_database():
    """Create the SQLite database and all required tables."""
    
    # Remove existing database for fresh start
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
        print(f"🗑️  Removed existing database: {DATABASE_PATH}")
    
    # Connect to SQLite database (creates file if doesn't exist)
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    print(f"📂 Creating database: {DATABASE_PATH}")
    
    # Enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # 1. METRICS TABLE - Store evaluation methodologies
    cursor.execute("""
        CREATE TABLE metrics (
            id TEXT PRIMARY KEY,  -- UUID4 as string
            name TEXT NOT NULL UNIQUE,  -- metric name (e.g., 'geval', 'claude_eval')
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # 2. MODELS TABLE - Store LLM model configurations
    cursor.execute("""
        CREATE TABLE models (
            id TEXT PRIMARY KEY,  -- UUID4 as string
            name TEXT NOT NULL UNIQUE,  -- model name (e.g., 'gpt-4o-2024-08-06')
            provider TEXT NOT NULL,  -- provider name (e.g., 'openai', 'anthropic')
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # 3. CASES TABLE - Store evaluation case definitions
    cursor.execute("""
        CREATE TABLE cases (
            id TEXT PRIMARY KEY,  -- UUID4 as string
            name TEXT NOT NULL UNIQUE,  -- lowercase case name (e.g., 'fluency', 'consistency')
            task_introduction TEXT NOT NULL,  -- Task description for LLM
            evaluation_criteria TEXT NOT NULL,  -- Specific evaluation criteria
            min_score INTEGER NOT NULL DEFAULT 1,  -- Minimum score in range
            max_score INTEGER NOT NULL DEFAULT 5,  -- Maximum score in range
            requires_reference BOOLEAN NOT NULL DEFAULT FALSE,  -- Whether case needs expected_output
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # 4. EVAL_DOCUMENTS TABLE - Store documents to be evaluated
    cursor.execute("""
        CREATE TABLE eval_documents (
            id TEXT PRIMARY KEY,  -- UUID4 as string
            actual_output TEXT NOT NULL,  -- The output being evaluated (required)
            expected_output TEXT,  -- Reference output for comparison (optional)
            metadata TEXT,  -- JSON field for additional document info
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # 5. JUDGES TABLE - Store specialized evaluation configurations
    cursor.execute("""
        CREATE TABLE judges (
            id TEXT PRIMARY KEY,  -- UUID4 as string
            name TEXT,  -- Optional judge name
            model_id TEXT NOT NULL,  -- Foreign key to models table
            case_id TEXT NOT NULL,  -- Foreign key to cases table (1:1 specialization)
            metric_id TEXT NOT NULL,  -- Foreign key to metrics table
            parameters TEXT NOT NULL,  -- JSON string with evaluation parameters
            description TEXT,  -- Optional judge description
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE,
            FOREIGN KEY (case_id) REFERENCES cases (id) ON DELETE CASCADE,
            FOREIGN KEY (metric_id) REFERENCES metrics (id) ON DELETE CASCADE
        );
    """)
    
    # 6. RUNS TABLE - Store execution results and telemetry
    cursor.execute("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Auto-increment for runs
            judge_id TEXT NOT NULL,  -- Foreign key to judges table
            document_id TEXT NOT NULL,  -- Foreign key to eval_documents table
            status TEXT NOT NULL DEFAULT 'pending',  -- 'pending', 'running', 'completed', 'failed'
            
            -- Core Results
            final_score REAL,  -- weighted_score_sum / linear_probs_sum
            final_score_normalized REAL,  -- normalized version (0-1 range)
            all_responses TEXT,  -- JSON array of all n_responses scores
            
            -- Telemetry Data
            total_usage_tokens INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            execution_time_seconds REAL,  -- Total execution time
            
            -- Additional Data
            prompt_used TEXT,  -- The actual prompt sent to LLM
            error_message TEXT,  -- Error details if status='failed'
            metadata TEXT,  -- JSON field for additional run info
            
            -- Timestamps
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (judge_id) REFERENCES judges (id) ON DELETE CASCADE,
            FOREIGN KEY (document_id) REFERENCES eval_documents (id) ON DELETE CASCADE
        );
    """)
    
    # Create indexes for better query performance
    cursor.execute("CREATE INDEX idx_runs_judge_id ON runs(judge_id);")
    cursor.execute("CREATE INDEX idx_runs_document_id ON runs(document_id);")
    cursor.execute("CREATE INDEX idx_runs_status ON runs(status);")
    cursor.execute("CREATE INDEX idx_judges_case_id ON judges(case_id);")
    cursor.execute("CREATE INDEX idx_judges_metric_id ON judges(metric_id);")
    cursor.execute("CREATE INDEX idx_judges_model_id ON judges(model_id);")
    cursor.execute("CREATE INDEX idx_cases_name ON cases(name);")
    cursor.execute("CREATE INDEX idx_metrics_name ON metrics(name);")
    cursor.execute("CREATE INDEX idx_models_name ON models(name);")
    cursor.execute("CREATE INDEX idx_models_provider ON models(provider);")
    
    # Commit changes
    conn.commit()
    
    print("✅ Database tables created successfully!")
    print("\n📋 Tables created:")
    print("   • metrics - Evaluation methodologies")
    print("   • models - LLM model configurations")
    print("   • cases - Evaluation case definitions")
    print("   • eval_documents - Documents to evaluate")
    print("   • judges - Specialized evaluation configurations")
    print("   • runs - Execution results and telemetry")
    
    return conn


def insert_sample_data(conn):
    """Insert sample data for testing."""
    cursor = conn.cursor()
    
    print("\n🧪 Inserting sample data...")
    
    # Sample metric (G-Eval)
    metric_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO metrics (id, name)
        VALUES (?, ?)
    """, (metric_id, "geval"))
    
    # Sample model (GPT-4o)
    model_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO models (id, name, provider)
        VALUES (?, ?, ?)
    """, (model_id, "gpt-4o-2024-08-06", "openai"))
    
    # Sample case (Fluency - no reference needed)
    case_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO cases (id, name, task_introduction, evaluation_criteria, min_score, max_score, requires_reference)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        case_id,
        "fluency",
        "You will be given an actual output to evaluate. Your task is to rate the actual output on one metric.",
        "Fluency (1-3): the quality of the actual output in terms of grammar, spelling, punctuation, word choice, and sentence structure. - 1: Poor. The output has many errors that make it hard to understand or sound unnatural. - 2: Fair. The output has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible. - 3: Good. The output has few or no errors and is easy to read and follow.",
        1,
        3,
        False  # Fluency doesn't need reference
    ))
    
    # Sample document
    doc_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO eval_documents (id, actual_output, metadata)
        VALUES (?, ?, ?)
    """, (
        doc_id,
        "This is a well-written summary with proper grammar and clear structure.",
        json.dumps({"source": "sample", "type": "test"})
    ))
    
    # Sample judge (specialized for fluency case using geval metric)
    judge_id = str(uuid.uuid4())
    parameters = {
        "temperature": 2.0,
        "max_tokens": 2500,
        "n_responses": 10
    }
    cursor.execute("""
        INSERT INTO judges (id, name, model_id, case_id, metric_id, parameters, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        judge_id,
        "Fluency Specialist Judge",
        model_id,
        case_id,
        metric_id,
        json.dumps(parameters),
        "Specialized judge for fluency evaluation using G-Eval methodology"
    ))
    
    conn.commit()
    print(f"✅ Sample data inserted:")
    print(f"   • Metric ID: {metric_id} (geval)")
    print(f"   • Model ID: {model_id} (gpt-4o-2024-08-06 - openai)")
    print(f"   • Case ID: {case_id} (fluency)")
    print(f"   • Document ID: {doc_id}")
    print(f"   • Judge ID: {judge_id} (fluency specialist)")
    
    return metric_id, model_id, case_id, doc_id, judge_id


def verify_database(conn):
    """Verify database structure and sample data."""
    cursor = conn.cursor()
    
    print("\n🔍 Verifying database structure...")
    
    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    expected_tables = ['metrics', 'models', 'cases', 'eval_documents', 'judges', 'runs']
    for table in expected_tables:
        if table in tables:
            print(f"   ✅ Table '{table}' exists")
        else:
            print(f"   ❌ Table '{table}' missing")
    
    # Check sample data
    cursor.execute("SELECT COUNT(*) FROM metrics;")
    metric_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM models;")
    model_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM cases;")
    case_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM eval_documents;")
    doc_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM judges;")
    judge_count = cursor.fetchone()[0]
    
    print(f"\n📊 Data counts:")
    print(f"   • Metrics: {metric_count}")
    print(f"   • Models: {model_count}")
    print(f"   • Cases: {case_count}")
    print(f"   • Documents: {doc_count}")
    print(f"   • Judges: {judge_count}")
    print(f"   • Runs: 0 (will be created via API)")


def main():
    """Main function to create database and insert sample data."""
    print("🚀 G-Eval Database Setup")
    print("=" * 50)
    
    # Create database and tables
    conn = create_database()
    
    # Insert sample data
    metric_id, model_id, case_id, doc_id, judge_id = insert_sample_data(conn)
    
    # Verify everything is working
    verify_database(conn)
    
    # Close connection
    conn.close()
    
    print("\n🎉 Database setup complete!")
    print(f"📂 Database file: {DATABASE_PATH}")
    print(f"🔗 Sample IDs for testing:")
    print(f"   • Metric ID: {metric_id} (geval)")
    print(f"   • Model ID: {model_id} (gpt-4o-2024-08-06 - openai)")
    print(f"   • Case ID: {case_id} (fluency)")
    print(f"   • Document ID: {doc_id}")
    print(f"   • Judge ID: {judge_id} (fluency specialist)")
    print("\n💡 Next steps:")
    print("   1. Run this script: python create_database.py")
    print("   2. Update the FastAPI app for new structure")
    print("   3. Test the endpoints with the sample IDs above")


if __name__ == "__main__":
    main()
