#!/usr/bin/env python3
"""
G-Eval FastAPI Application

A REST API for managing and executing G-Eval evaluations with comprehensive telemetry.
Provides endpoints for creating evaluation metrics, cases, judges, and running evaluations.
"""

import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from database import DatabaseManager
from geval import GEval
from openai import OpenAI
from settings import config


# Initialize FastAPI app
app = FastAPI(
    title="G-Eval API",
    description="A REST API for managing and executing G-Eval evaluations with telemetry",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================
# Pydantic Models
# ===============================

class MetricCreateRequest(BaseModel):
    """Request model for creating evaluation methodologies."""
    name: str = Field(..., description="Unique name for the evaluation methodology (e.g., 'geval')")
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError('name cannot be empty')
        return v.strip().lower()

class MetricResponse(BaseModel):
    """Response model for evaluation methodologies."""
    id: str
    name: str
    created_at: str

class ModelCreateRequest(BaseModel):
    """Request model for creating LLM model configurations."""
    name: str = Field(..., description="Model name (e.g., 'gpt-4o-2024-08-06')")
    provider: str = Field(..., description="Provider name (e.g., 'openai', 'anthropic')")
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError('name cannot be empty')
        return v.strip()
    
    @validator('provider')
    def provider_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError('provider cannot be empty')
        return v.strip().lower()

class ModelResponse(BaseModel):
    """Response model for LLM model configurations."""
    id: str
    name: str
    provider: str
    created_at: str

class CaseCreateRequest(BaseModel):
    """Request model for creating evaluation cases."""
    name: str = Field(..., description="Unique name for the evaluation case (will be lowercased)")
    task_introduction: str = Field(..., description="Task description for the LLM")
    evaluation_criteria: str = Field(..., description="Specific evaluation criteria")
    min_score: int = Field(1, ge=1, description="Minimum score in the evaluation range")
    max_score: int = Field(5, ge=1, description="Maximum score in the evaluation range")
    requires_reference: bool = Field(False, description="Whether this case requires expected_output for comparison")
    score_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Pass/fail threshold for normalized scores (0.0-1.0)")
    
    @validator('max_score')
    def max_score_must_be_greater_than_min(cls, v, values):
        if 'min_score' in values and v <= values['min_score']:
            raise ValueError('max_score must be greater than min_score')
        return v
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError('name cannot be empty')
        return v.strip().lower()

class CaseResponse(BaseModel):
    """Response model for evaluation cases."""
    id: str
    name: str
    task_introduction: str
    evaluation_criteria: str
    min_score: int
    max_score: int
    requires_reference: bool
    score_threshold: float
    created_at: str

class JudgeCreateRequest(BaseModel):
    """Request model for creating specialized judges."""
    name: Optional[str] = Field(None, description="Optional name for the judge")
    model_id: str = Field(..., description="ID of the LLM model to use")
    case_id: str = Field(..., description="ID of the evaluation case to judge")
    metric_id: str = Field(..., description="ID of the evaluation methodology to use")
    parameters: Dict[str, Any] = Field(..., description="Evaluation parameters")
    description: Optional[str] = Field(None, description="Optional judge description")
    
    @validator('case_id', 'metric_id', 'model_id')
    def ids_must_be_uuid(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('ID must be a valid UUID')
        return v

class JudgeResponse(BaseModel):
    """Response model for specialized judges."""
    id: str
    name: Optional[str]
    model_id: str
    model_name: str
    model_provider: str
    case_id: str
    metric_id: str
    case_name: str
    metric_name: str
    parameters: Dict[str, Any]
    description: Optional[str]
    created_at: str

class JudgeUpdateRequest(BaseModel):
    """Request model for updating judge configuration."""
    name: Optional[str] = Field(None, description="Updated name for the judge")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Updated evaluation parameters")
    description: Optional[str] = Field(None, description="Updated judge description")
    model_id: Optional[str] = Field(None, description="Updated model ID for the judge")

class DocumentCreateRequest(BaseModel):
    """Request model for creating evaluation documents."""
    actual_output: str = Field(..., description="The output being evaluated")
    expected_output: Optional[str] = Field(None, description="Reference output for comparison (optional)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional document metadata")

class DocumentResponse(BaseModel):
    """Response model for evaluation documents."""
    id: str
    actual_output: str
    expected_output: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: str

class EvaluationRequest(BaseModel):
    """Request model for running evaluations."""
    judge_id: str = Field(..., description="ID of the judge to use for evaluation")
    document_id: str = Field(..., description="ID of the document to evaluate")
    
    @validator('judge_id', 'document_id')
    def ids_must_be_uuid(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('ID must be a valid UUID')
        return v

class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    run_id: int
    judge_name: Optional[str]
    case_name: str
    metric_name: str
    status: str
    final_score: Optional[float]
    final_score_normalized: Optional[float]
    total_usage_tokens: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    execution_time_seconds: Optional[float]
    started_at: str
    completed_at: Optional[str]

class RunResponse(BaseModel):
    """Response model for detailed run information."""
    id: int
    judge_id: str
    document_id: str
    judge_name: Optional[str]
    case_name: str
    metric_name: str
    model_name: Optional[str]
    model_provider: Optional[str]
    status: str
    evaluation_status: Optional[str]
    final_score: Optional[float]
    final_score_normalized: Optional[float]
    all_responses: Optional[List[float]]
    total_usage_tokens: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    execution_time_seconds: Optional[float]
    prompt_used: Optional[str]
    error_message: Optional[str]
    metadata: Optional[Dict[str, Any]]
    actual_output: str
    expected_output: Optional[str]
    started_at: str
    completed_at: Optional[str]
    created_at: str


# ===============================
# API Endpoints
# ===============================

@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "G-Eval API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# ===============================
# Metrics Endpoints
# ===============================

@app.post("/metrics", response_model=MetricResponse, tags=["Metrics"])
async def create_metric(request: MetricCreateRequest):
    """Create a new evaluation methodology."""
    try:
        metric_id = DatabaseManager.create_metric(name=request.name)
        metric = DatabaseManager.get_metric(metric_id)
        
        if not metric:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create metric"
            )
        
        return MetricResponse(**metric)
    
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Metric with name '{request.name}' already exists"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create metric: {str(e)}"
        )

@app.get("/metrics", response_model=List[MetricResponse], tags=["Metrics"])
async def list_metrics():
    """List all evaluation methodologies."""
    try:
        metrics = DatabaseManager.list_metrics()
        return [MetricResponse(**metric) for metric in metrics]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )

@app.get("/metrics/{metric_id}", response_model=MetricResponse, tags=["Metrics"])
async def get_metric(metric_id: str):
    """Get a specific evaluation methodology by ID."""
    try:
        metric = DatabaseManager.get_metric(metric_id)
        if not metric:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Metric not found"
            )
        return MetricResponse(**metric)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metric: {str(e)}"
        )

# ===============================
# Models Endpoints
# ===============================

@app.post("/models", response_model=ModelResponse, tags=["Models"])
async def create_model(request: ModelCreateRequest):
    """Create a new LLM model configuration."""
    try:
        model_id = DatabaseManager.create_model(name=request.name, provider=request.provider)
        model = DatabaseManager.get_model(model_id)
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create model"
            )
        
        return ModelResponse(**model)
    
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model with name '{request.name}' already exists"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create model: {str(e)}"
        )

@app.get("/models", response_model=List[ModelResponse], tags=["Models"])
async def list_models():
    """List all LLM model configurations."""
    try:
        models = DatabaseManager.list_models()
        return [ModelResponse(**model) for model in models]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models: {str(e)}"
        )

@app.get("/models/{model_id}", response_model=ModelResponse, tags=["Models"])
async def get_model(model_id: str):
    """Get a specific LLM model configuration by ID."""
    try:
        model = DatabaseManager.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        return ModelResponse(**model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model: {str(e)}"
        )

# ===============================
# Cases Endpoints
# ===============================

@app.post("/cases", response_model=CaseResponse, tags=["Cases"])
async def create_case(request: CaseCreateRequest):
    """Create a new evaluation case."""
    try:
        case_id = DatabaseManager.create_case(
            name=request.name,
            task_introduction=request.task_introduction,
            evaluation_criteria=request.evaluation_criteria,
            min_score=request.min_score,
            max_score=request.max_score,
            requires_reference=request.requires_reference,
            score_threshold=request.score_threshold
        )
        
        case = DatabaseManager.get_case(case_id)
        if not case:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create case"
            )
        
        return CaseResponse(**case)
    
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Case with name '{request.name}' already exists"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create case: {str(e)}"
        )

@app.get("/cases", response_model=List[CaseResponse], tags=["Cases"])
async def list_cases():
    """List all evaluation cases."""
    try:
        cases = DatabaseManager.list_cases()
        return [CaseResponse(**case) for case in cases]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cases: {str(e)}"
        )

@app.get("/cases/{case_id}", response_model=CaseResponse, tags=["Cases"])
async def get_case(case_id: str):
    """Get a specific evaluation case by ID."""
    try:
        case = DatabaseManager.get_case(case_id)
        if not case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found"
            )
        return CaseResponse(**case)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve case: {str(e)}"
        )

# ===============================
# Documents Endpoints
# ===============================

@app.post("/documents", response_model=DocumentResponse, tags=["Documents"])
async def create_document(request: DocumentCreateRequest):
    """Create a new evaluation document."""
    try:
        doc_id = DatabaseManager.create_document(
            actual_output=request.actual_output,
            expected_output=request.expected_output,
            metadata=request.metadata
        )
        
        document = DatabaseManager.get_document(doc_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create document"
            )
        
        return DocumentResponse(**document)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create document: {str(e)}"
        )

@app.get("/documents", response_model=List[DocumentResponse], tags=["Documents"])
async def list_documents():
    """List all evaluation documents."""
    try:
        documents = DatabaseManager.list_documents()
        return [DocumentResponse(**doc) for doc in documents]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )

# ===============================
# Judges Endpoints
# ===============================

@app.post("/judges", response_model=JudgeResponse, tags=["Judges"])
async def create_judge(request: JudgeCreateRequest):
    """Create a new specialized judge."""
    try:
        # Verify case, metric, and model exist
        case = DatabaseManager.get_case(request.case_id)
        if not case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found"
            )
        
        metric = DatabaseManager.get_metric(request.metric_id)
        if not metric:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Metric not found"
            )
        
        model = DatabaseManager.get_model(request.model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        judge_id = DatabaseManager.create_judge(
            name=request.name,
            model_id=request.model_id,
            case_id=request.case_id,
            metric_id=request.metric_id,
            parameters=request.parameters,
            description=request.description
        )
        
        judge = DatabaseManager.get_judge(judge_id)
        if not judge:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create judge"
            )
        
        return JudgeResponse(**judge)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create judge: {str(e)}"
        )

@app.get("/judges", response_model=List[JudgeResponse], tags=["Judges"])
async def list_judges():
    """List all specialized judges."""
    try:
        judges = DatabaseManager.list_judges()
        return [JudgeResponse(**judge) for judge in judges]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve judges: {str(e)}"
        )

@app.put("/judges/{judge_id}", response_model=JudgeResponse, tags=["Judges"])
async def update_judge(judge_id: str, request: JudgeUpdateRequest):
    """Update judge configuration (name, parameters, description, model)."""
    try:
        # Validate judge_id format
        try:
            uuid.UUID(judge_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid judge ID format"
            )
        
        # Check if judge exists
        existing_judge = DatabaseManager.get_judge(judge_id)
        if not existing_judge:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Judge not found"
            )
        
        # Validate model_id if provided
        if request.model_id:
            try:
                uuid.UUID(request.model_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid model ID format"
                )
            
            # Check if model exists
            existing_model = DatabaseManager.get_model(request.model_id)
            if not existing_model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Model not found"
                )
        
        # Perform update
        updated = DatabaseManager.update_judge(
            judge_id=judge_id,
            name=request.name,
            parameters=request.parameters,
            description=request.description,
            model_id=request.model_id
        )
        
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Judge not found"
            )
        
        # Return updated judge
        judge = DatabaseManager.get_judge(judge_id)
        if not judge:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve updated judge"
            )
        
        return JudgeResponse(**judge)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update judge: {str(e)}"
        )

# ===============================
# Evaluation Endpoint
# ===============================

@app.post("/eval", response_model=EvaluationResponse, tags=["Evaluation"])
async def run_evaluation(request: EvaluationRequest):
    """Execute an evaluation using a specialized judge."""
    start_time = time.time()
    
    try:
        # Get judge with full case and metric information
        judge = DatabaseManager.get_judge(request.judge_id)
        if not judge:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Judge not found"
            )
        
        # Get document
        document = DatabaseManager.get_document(request.document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Check metric type and handle accordingly
        if judge['metric_name'].lower() != 'geval':
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Metric '{judge['metric_name']}' not yet implemented. Currently only 'geval' is supported."
            )
        
        # Create run record
        run_id = DatabaseManager.create_run(
            judge_id=request.judge_id,
            document_id=request.document_id
        )
        
        # Update run to running status
        DatabaseManager.update_run_status(run_id, "running")
        
        try:
            # Initialize GEval with judge parameters
            geval_params = {
                'name': judge['case_name'],
                'task_introduction': judge['task_introduction'],
                'evaluation_criteria': judge['evaluation_criteria'],
                'min_score': judge['min_score'],
                'max_score': judge['max_score'],
                'client': OpenAI(api_key=config.openai_api_key),  # Add OpenAI client
                **judge['parameters']
            }
            
            # Filter out non-serializable objects for metadata storage
            serializable_params = {k: v for k, v in geval_params.items() if k not in ['client']}
            
            # Create GEval instance
            geval = GEval(**geval_params)
            
            # Prepare evaluation inputs based on requires_reference
            actual_output = document['actual_output']
            expected_output = document['expected_output'] if judge['requires_reference'] else None
            
            # Execute evaluation
            if judge['requires_reference'] and expected_output:
                result = geval.evaluate(actual_output=actual_output, expected_output=expected_output)
            else:
                result = geval.evaluate(actual_output=actual_output)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Get token usage
            token_usage = geval.get_token_usage_summary()
            
            # Calculate final score from all responses (average)
            final_score = sum(result['all_responses']) / len(result['all_responses']) if result['all_responses'] else 0
            
            # Calculate normalized score
            score_range = judge['max_score'] - judge['min_score']
            final_score_normalized = (final_score - judge['min_score']) / score_range if score_range > 0 else 0
            
            # Complete the run with results
            DatabaseManager.complete_run(
                run_id=run_id,
                final_score=final_score,
                final_score_normalized=final_score_normalized,
                all_responses=result['all_responses'],
                total_usage_tokens=token_usage['total_tokens'],
                prompt_tokens=token_usage['prompt_tokens'],
                completion_tokens=token_usage['completion_tokens'],
                execution_time_seconds=execution_time,
                prompt_used=result.get('prompt', ''),
                metadata={
                    'judge_parameters': serializable_params,
                    'geval_version': '1.0.0',
                    'evaluation_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Return evaluation response
            return EvaluationResponse(
                run_id=run_id,
                judge_name=judge['name'],
                case_name=judge['case_name'],
                metric_name=judge['metric_name'],
                status="completed",
                final_score=final_score,
                final_score_normalized=final_score_normalized,
                total_usage_tokens=token_usage['total_tokens'],
                prompt_tokens=token_usage['prompt_tokens'],
                completion_tokens=token_usage['completion_tokens'],
                execution_time_seconds=execution_time,
                started_at=datetime.utcnow().isoformat(),
                completed_at=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            # Mark run as failed
            error_msg = f"Evaluation failed: {str(e)}\n{traceback.format_exc()}"
            DatabaseManager.update_run_status(run_id, "failed", error_msg)
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Evaluation execution failed: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run evaluation: {str(e)}"
        )

# ===============================
# Runs Endpoints
# ===============================

@app.get("/runs/{run_id}", response_model=RunResponse, tags=["Runs"])
async def get_run(run_id: int):
    """Get detailed information about a specific run."""
    try:
        run = DatabaseManager.get_run(run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Run not found"
            )
        return RunResponse(**run)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve run: {str(e)}"
        )

@app.get("/runs", response_model=List[RunResponse], tags=["Runs"])
async def list_runs(judge_id: Optional[str] = None, limit: int = 100):
    """List runs, optionally filtered by judge."""
    try:
        runs = DatabaseManager.list_runs(judge_id=judge_id, limit=limit)
        return [RunResponse(**run) for run in runs]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve runs: {str(e)}"
        )


# ===============================
# Server Startup
# ===============================

if __name__ == "__main__":
    print("🚀 Starting G-Eval API Server...")
    print("📋 Available endpoints:")
    print("   • GET  /          - Health check")
    print("   • POST /metrics   - Create evaluation methodologies") 
    print("   • GET  /metrics   - List evaluation methodologies")
    print("   • POST /models    - Create LLM model configurations")
    print("   • GET  /models    - List LLM model configurations")
    print("   • POST /cases     - Create evaluation cases")
    print("   • GET  /cases     - List evaluation cases")
    print("   • POST /documents - Create documents")
    print("   • GET  /documents - List documents")
    print("   • POST /judges    - Create specialized judges")
    print("   • GET  /judges    - List specialized judges")
    print("   • PUT  /judges/{id} - Update judge parameters")
    print("   • POST /eval      - Run evaluations")
    print("   • GET  /runs      - List runs")
    print("   • GET  /docs      - API documentation")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )