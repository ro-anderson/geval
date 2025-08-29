"""
Database utilities for G-Eval FastAPI application.

This module provides database connection and helper functions for working with SQLite.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager


DATABASE_PATH = "geval_app.db"


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
    conn.execute("PRAGMA foreign_keys = ON;")  # Enable foreign key constraints
    try:
        yield conn
    finally:
        conn.close()


def dict_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert SQLite Row to dictionary."""
    return {key: row[key] for key in row.keys()}


class DatabaseManager:
    """Database manager class for G-Eval operations."""
    
    # METRICS Operations
    @staticmethod
    def create_metric(name: str) -> str:
        """
        Create a new evaluation methodology.
        
        Returns:
            The UUID of the created metric
        """
        metric_id = str(uuid.uuid4())
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (id, name)
                VALUES (?, ?)
            """, (metric_id, name.lower()))
            conn.commit()
        
        return metric_id
    
    @staticmethod
    def get_metric(metric_id: str) -> Optional[Dict[str, Any]]:
        """Get metric by ID."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metrics WHERE id = ?", (metric_id,))
            row = cursor.fetchone()
            return dict_from_row(row) if row else None
    
    @staticmethod
    def list_metrics() -> List[Dict[str, Any]]:
        """List all metrics."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metrics ORDER BY created_at DESC")
            return [dict_from_row(row) for row in cursor.fetchall()]
    
    # MODELS Operations
    @staticmethod
    def create_model(name: str, provider: str) -> str:
        """
        Create a new LLM model configuration.
        
        Returns:
            The UUID of the created model
        """
        model_id = str(uuid.uuid4())
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO models (id, name, provider)
                VALUES (?, ?, ?)
            """, (model_id, name, provider.lower()))
            conn.commit()
        
        return model_id
    
    @staticmethod
    def get_model(model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models WHERE id = ?", (model_id,))
            row = cursor.fetchone()
            return dict_from_row(row) if row else None
    
    @staticmethod
    def list_models() -> List[Dict[str, Any]]:
        """List all models."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models ORDER BY provider, name")
            return [dict_from_row(row) for row in cursor.fetchall()]
    
    # CASES Operations  
    @staticmethod
    def create_case(
        name: str,
        task_introduction: str,
        evaluation_criteria: str,
        min_score: int = 1,
        max_score: int = 5,
        requires_reference: bool = False,
        score_threshold: float = 0.5
    ) -> str:
        """
        Create a new evaluation case.
        
        Args:
            score_threshold: Threshold for pass/fail evaluation (0.0-1.0, default 0.5)
        
        Returns:
            The UUID of the created case
        """
        case_id = str(uuid.uuid4())
        
        # Validate score_threshold
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cases (id, name, task_introduction, evaluation_criteria, 
                                 min_score, max_score, requires_reference, score_threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (case_id, name.lower(), task_introduction, evaluation_criteria, 
                  min_score, max_score, requires_reference, score_threshold))
            conn.commit()
        
        return case_id
    
    @staticmethod
    def get_case(case_id: str) -> Optional[Dict[str, Any]]:
        """Get case by ID."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
            row = cursor.fetchone()
            return dict_from_row(row) if row else None
    
    @staticmethod
    def list_cases() -> List[Dict[str, Any]]:
        """List all cases."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases ORDER BY created_at DESC")
            return [dict_from_row(row) for row in cursor.fetchall()]
    
    @staticmethod
    def create_document(
        actual_output: str,
        expected_output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new document for evaluation.
        
        Returns:
            The UUID of the created document
        """
        doc_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata) if metadata else None
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO eval_documents (id, actual_output, expected_output, metadata)
                VALUES (?, ?, ?, ?)
            """, (doc_id, actual_output, expected_output, metadata_json))
            conn.commit()
        
        return doc_id
    
    @staticmethod
    def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM eval_documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                doc = dict_from_row(row)
                # Parse metadata JSON
                if doc['metadata']:
                    doc['metadata'] = json.loads(doc['metadata'])
                return doc
            return None
    
    @staticmethod
    def list_documents() -> List[Dict[str, Any]]:
        """List all documents."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM eval_documents ORDER BY created_at DESC")
            documents = []
            for row in cursor.fetchall():
                doc = dict_from_row(row)
                # Parse metadata JSON
                if doc['metadata']:
                    doc['metadata'] = json.loads(doc['metadata'])
                documents.append(doc)
            return documents
    
    # JUDGES Operations
    @staticmethod
    def create_judge(
        name: Optional[str],
        model_id: str,
        case_id: str,
        metric_id: str,
        parameters: Dict[str, Any],
        description: Optional[str] = None
    ) -> str:
        """
        Create a new specialized judge.
        
        Returns:
            The UUID of the created judge
        """
        judge_id = str(uuid.uuid4())
        parameters_json = json.dumps(parameters)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO judges (id, name, model_id, case_id, metric_id, parameters, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (judge_id, name, model_id, case_id, metric_id, parameters_json, description))
            conn.commit()
        
        return judge_id
    
    @staticmethod
    def get_judge(judge_id: str) -> Optional[Dict[str, Any]]:
        """Get judge by ID with associated case, metric, and model data."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT j.*, c.name as case_name, c.task_introduction, c.evaluation_criteria,
                       c.min_score, c.max_score, c.requires_reference, m.name as metric_name,
                       mod.name as model_name, mod.provider as model_provider
                FROM judges j
                JOIN cases c ON j.case_id = c.id
                JOIN metrics m ON j.metric_id = m.id
                JOIN models mod ON j.model_id = mod.id
                WHERE j.id = ?
            """, (judge_id,))
            row = cursor.fetchone()
            if row:
                judge = dict_from_row(row)
                # Parse parameters JSON
                judge['parameters'] = json.loads(judge['parameters'])
                return judge
            return None
    
    @staticmethod
    def list_judges() -> List[Dict[str, Any]]:
        """List all judges with their case, metric, and model info."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT j.*, c.name as case_name, m.name as metric_name,
                       mod.name as model_name, mod.provider as model_provider
                FROM judges j
                JOIN cases c ON j.case_id = c.id
                JOIN metrics m ON j.metric_id = m.id
                JOIN models mod ON j.model_id = mod.id
                ORDER BY j.created_at DESC
            """)
            judges = []
            for row in cursor.fetchall():
                judge = dict_from_row(row)
                judge['parameters'] = json.loads(judge['parameters'])
                judges.append(judge)
            return judges
    
    @staticmethod
    def update_judge(
        judge_id: str,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> bool:
        """
        Update judge information (name, parameters, description, model_id).
        Case and metric relationships cannot be changed.
        
        Returns:
            True if update was successful, False if judge not found
        """
        # Build dynamic update query
        update_fields = []
        update_values = []
        
        if name is not None:
            update_fields.append("name = ?")
            update_values.append(name)
        
        if parameters is not None:
            update_fields.append("parameters = ?")
            update_values.append(json.dumps(parameters))
        
        if description is not None:
            update_fields.append("description = ?")
            update_values.append(description)
        
        if model_id is not None:
            update_fields.append("model_id = ?")
            update_values.append(model_id)
        
        if not update_fields:
            return True  # No updates needed
        
        update_values.append(judge_id)  # For WHERE clause
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = f"UPDATE judges SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, update_values)
            conn.commit()
            return cursor.rowcount > 0
    
    @staticmethod
    def update_case(
        case_id: str,
        name: Optional[str] = None,
        task_introduction: Optional[str] = None,
        evaluation_criteria: Optional[str] = None,
        min_score: Optional[int] = None,
        max_score: Optional[int] = None,
        requires_reference: Optional[bool] = None,
        score_threshold: Optional[float] = None
    ) -> bool:
        """
        Update case information.
        
        Returns:
            True if update was successful, False if case not found
        """
        # Build dynamic update query
        update_fields = []
        update_values = []
        
        if name is not None:
            update_fields.append("name = ?")
            update_values.append(name.lower())
        
        if task_introduction is not None:
            update_fields.append("task_introduction = ?")
            update_values.append(task_introduction)
        
        if evaluation_criteria is not None:
            update_fields.append("evaluation_criteria = ?")
            update_values.append(evaluation_criteria)
        
        if min_score is not None:
            update_fields.append("min_score = ?")
            update_values.append(min_score)
        
        if max_score is not None:
            update_fields.append("max_score = ?")
            update_values.append(max_score)
        
        if requires_reference is not None:
            update_fields.append("requires_reference = ?")
            update_values.append(requires_reference)
        
        if score_threshold is not None:
            update_fields.append("score_threshold = ?")
            update_values.append(score_threshold)
        
        if not update_fields:
            return True  # No updates needed
        
        update_values.append(case_id)  # For WHERE clause
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = f"UPDATE cases SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, update_values)
            conn.commit()
            return cursor.rowcount > 0
    
    @staticmethod
    def create_run(
        judge_id: str,
        document_id: str
    ) -> int:
        """
        Create a new run record.
        
        Returns:
            The ID of the created run
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO runs (judge_id, document_id, status, started_at)
                VALUES (?, ?, 'pending', ?)
            """, (judge_id, document_id, datetime.utcnow().isoformat()))
            conn.commit()
            return cursor.lastrowid
    
    @staticmethod
    def update_run_status(run_id: int, status: str, error_message: Optional[str] = None):
        """Update run status."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE runs SET status = ?, error_message = ? WHERE id = ?
            """, (status, error_message, run_id))
            conn.commit()
    
    @staticmethod
    def complete_run(
        run_id: int,
        final_score: float,
        final_score_normalized: float,
        all_responses: List[float],
        total_usage_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        execution_time_seconds: float,
        prompt_used: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Complete a run with results and calculate evaluation status."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get the score threshold from the case via judge
            cursor.execute("""
                SELECT c.score_threshold 
                FROM runs r 
                JOIN judges j ON r.judge_id = j.id 
                JOIN cases c ON j.case_id = c.id 
                WHERE r.id = ?
            """, (run_id,))
            
            threshold_row = cursor.fetchone()
            if threshold_row:
                score_threshold = threshold_row[0]
                # Determine evaluation status based on threshold
                evaluation_status = 'pass' if final_score_normalized >= score_threshold else 'fail'
            else:
                # Fallback if threshold not found (shouldn't happen)
                evaluation_status = 'pending'
            
            cursor.execute("""
                UPDATE runs SET 
                    status = 'completed',
                    final_score = ?,
                    final_score_normalized = ?,
                    all_responses = ?,
                    total_usage_tokens = ?,
                    prompt_tokens = ?,
                    completion_tokens = ?,
                    execution_time_seconds = ?,
                    prompt_used = ?,
                    metadata = ?,
                    evaluation_status = ?,
                    completed_at = ?
                WHERE id = ?
            """, (
                final_score, final_score_normalized, json.dumps(all_responses),
                total_usage_tokens, prompt_tokens, completion_tokens,
                execution_time_seconds, prompt_used,
                json.dumps(metadata) if metadata else None,
                evaluation_status,
                datetime.utcnow().isoformat(),
                run_id
            ))
            conn.commit()
    
    @staticmethod
    def get_run(run_id: int) -> Optional[Dict[str, Any]]:
        """Get run by ID with full details."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.*, j.name as judge_name, mod.name as model_name, mod.provider as model_provider,
                       c.name as case_name, m.name as metric_name,
                       d.actual_output, d.expected_output
                FROM runs r
                JOIN judges j ON r.judge_id = j.id
                JOIN cases c ON j.case_id = c.id
                JOIN metrics m ON j.metric_id = m.id
                JOIN models mod ON j.model_id = mod.id
                JOIN eval_documents d ON r.document_id = d.id
                WHERE r.id = ?
            """, (run_id,))
            row = cursor.fetchone()
            if row:
                run = dict_from_row(row)
                # Parse JSON fields
                if run['all_responses']:
                    run['all_responses'] = json.loads(run['all_responses'])
                if run['metadata']:
                    run['metadata'] = json.loads(run['metadata'])
                return run
            return None
    
    @staticmethod
    def list_runs(judge_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List runs, optionally filtered by judge."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if judge_id:
                cursor.execute("""
                    SELECT r.*, j.name as judge_name, c.name as case_name, m.name as metric_name,
                           mod.name as model_name, mod.provider as model_provider,
                           d.actual_output, d.expected_output
                    FROM runs r
                    JOIN judges j ON r.judge_id = j.id
                    JOIN cases c ON j.case_id = c.id
                    JOIN metrics m ON j.metric_id = m.id
                    JOIN models mod ON j.model_id = mod.id
                    JOIN eval_documents d ON r.document_id = d.id
                    WHERE r.judge_id = ?
                    ORDER BY r.created_at DESC
                    LIMIT ?
                """, (judge_id, limit))
            else:
                cursor.execute("""
                    SELECT r.*, j.name as judge_name, c.name as case_name, m.name as metric_name,
                           mod.name as model_name, mod.provider as model_provider,
                           d.actual_output, d.expected_output
                    FROM runs r
                    JOIN judges j ON r.judge_id = j.id
                    JOIN cases c ON j.case_id = c.id
                    JOIN metrics m ON j.metric_id = m.id
                    JOIN models mod ON j.model_id = mod.id
                    JOIN eval_documents d ON r.document_id = d.id
                    ORDER BY r.created_at DESC
                    LIMIT ?
                """, (limit,))
            
            runs = []
            for row in cursor.fetchall():
                run = dict_from_row(row)
                # Parse JSON fields
                if run['all_responses']:
                    run['all_responses'] = json.loads(run['all_responses'])
                if run['metadata']:
                    run['metadata'] = json.loads(run['metadata'])
                runs.append(run)
            
            return runs
