"""
FastAPI Application with two endpoints for file processing.
Production-ready with proper validation, error handling, and security features.
"""

import os
import json
import logging
from typing import Optional
from pathlib import Path

import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, field_validator
from Interview_question_generation import extract_text_from_pdf, interview_question
from Interview_report import evaluate_interview_transcript


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============= EXTERNAL API =============
EXTERNAL_API_URL = "https://aiinterviewagent.bestworks.cloud/analysis/ai/"

# Initialize FastAPI app
app = FastAPI(
    title="AI Interview API",
    description="API for processing resumes and job descriptions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Production middleware configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)


# ============= PYDANTIC MODELS =============

class ProcessPDFResponse(BaseModel):
    success: bool
    message: str
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    jd_length: Optional[int] = None
    response: Optional[list] = None


class ProcessTextResponse(BaseModel):
    token: str
    analysis: Optional[dict] = None

    @field_validator("analysis", mode="before")
    @classmethod
    def parse_analysis(cls, v):
        """
        Handles 3 possible cases:
        1. Already a dict        → return as-is
        2. A JSON string         → parse and return dict
        3. A Pydantic model      → call .model_dump()
        """
        if v is None:
            return v
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                cleaned = v.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                return json.loads(cleaned.strip())
            except json.JSONDecodeError as e:
                raise ValueError(f"analysis string is not valid JSON: {e}")
        if hasattr(v, "model_dump"):
            return v.model_dump()
        raise ValueError(f"Cannot convert analysis of type {type(v)} to dict")


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail: Optional[str] = None


# ============= CONSTANTS =============

ALLOWED_PDF_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


# ============= HELPERS =============

def validate_file_size(file, max_size: int = MAX_FILE_SIZE) -> None:
    """Validate file size before processing."""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {max_size / (1024*1024):.1f}MB"
        )
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty"
        )


def validate_file_extension(filename: str, allowed_extensions: set) -> None:
    """Validate file extension."""
    file_ext = Path(filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )


async def forward_to_external_api(token: str, analysis: dict):
    """
    Forwards the evaluation result to the external API endpoint.
    Payload: { "token": "...", "analysis": { ... } }
    """
    payload = {
        "token": token,
        "analysis": json.dumps(analysis)  # Java backend expects analysis as a JSON string
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                EXTERNAL_API_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info(f"External API call successful: status={response.status_code}")
            return response.json()

    except httpx.TimeoutException:
        logger.error("External API call timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="External API timed out"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"External API returned error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"External API error: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Unexpected error calling external API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to reach external API: {str(e)}"
        )


# ============= ENDPOINTS =============

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint for health check."""
    return {
        "status": "healthy",
        "message": "AI Interview API is running",
        "version": "1.0.0"
    }


@app.post(
    "/api/v1/process-pdf",
    response_model=ProcessPDFResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        413: {"model": ErrorResponse, "description": "Payload Too Large"},
        415: {"model": ErrorResponse, "description": "Unsupported Media Type"},
        422: {"model": ErrorResponse, "description": "Validation Error"}
    },
    tags=["PDF Processing"],
    summary="Process PDF with Job Description",
    description="Upload a PDF file (resume) along with a job description text for processing"
)
async def process_pdf_with_jd(
    pdf_file: UploadFile = File(
        ...,
        description="PDF file to be processed",
        media_type="application/pdf"
    ),
    jd_text: str = Form(
        ...,
        description="Job Description text",
        min_length=1,
        max_length=10000
    ),
    Experience: str = Form(
        ...,
        description="Experience text",
        min_length=1,
        max_length=10000
    ),
    Mandatory_skills: str = Form(
        ...,
        description="Mandatory skills text",
        min_length=1,
        max_length=10000
    ),
    Nice_to_have_skills: str = Form(
        ...,
        description="Nice to have skills text",
        min_length=1,
        max_length=10000
    ),
) -> ProcessPDFResponse:
    """
    Endpoint to process a PDF file along with job description text.

    - **pdf_file**: PDF file upload (required)
    - **jd_text**: Job description as plain text (required, 1-10000 characters)

    Returns processing status and metadata.
    """
    try:
        validate_file_extension(pdf_file.filename, ALLOWED_PDF_EXTENSIONS)
        validate_file_size(pdf_file.file)

        content = await pdf_file.read()
        file_size = len(content)

        logger.info(
            f"Processing PDF: {pdf_file.filename}, "
            f"Size: {file_size} bytes, "
            f"JD length: {len(jd_text)} characters"
        )

        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, pdf_file.filename)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(content)

        data = await extract_text_from_pdf(temp_file_path)
        result = await interview_question(data, jd_text, Experience, Mandatory_skills, Nice_to_have_skills)

        os.remove(temp_file_path)

        return ProcessPDFResponse(
            success=True,
            message="PDF and JD processed successfully",
            file_name=pdf_file.filename,
            file_size=file_size,
            jd_length=len(jd_text),
            response=[question.all_question for question in result.all_question]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the PDF"
        )


@app.post(
    "/api/v1/process-text",
    response_model=ProcessTextResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        413: {"model": ErrorResponse, "description": "Payload Too Large"},
        415: {"model": ErrorResponse, "description": "Unsupported Media Type"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        502: {"model": ErrorResponse, "description": "External API Error"},
        504: {"model": ErrorResponse, "description": "External API Timeout"},
    },
    tags=["Text Processing"],
    summary="Process Text File",
    description="Upload a text file for processing"
)
async def process_text_file(
    text: str = Form(
        ...,
        description="Transcript text",
        min_length=1,
        max_length=100000
    ),
    Experience: str = Form(
        ...,
        description="Experience text",
        min_length=1,
        max_length=10000
    ),
    Mandatory_skills: str = Form(
        ...,
        description="Mandatory skills text",
        min_length=1,
        max_length=10000
    ),
    Nice_to_have_skills: str = Form(
        ...,
        description="Nice to have skills text",
        min_length=1,
        max_length=10000
    ),
    user_id: str = Form(
        ...,
        description="User id text",
        min_length=1,
        max_length=10000
    )
) -> ProcessTextResponse:
    """
    Endpoint to process a transcript text.

    - **text**: Interview transcript text (required)
    - **Experience**: Experience level (required)
    - **Mandatory_skills**: Must-have skills (required)
    - **Nice_to_have_skills**: Preferred skills (required)
    - **user_id**: User identifier returned as token (required)

    Returns evaluation result and forwards it to the external API.
    """
    try:
        # Step 1: Evaluate the transcript
        result = await evaluate_interview_transcript(
            text,
            Experience,
            Mandatory_skills,
            Nice_to_have_skills
        )

        # Step 2: Normalize result to a clean dict
        if hasattr(result, "model_dump"):
            analysis_data = result.model_dump()
        elif isinstance(result, dict):
            analysis_data = result
        else:
            analysis_data = result  # validator will handle conversion

        # Step 3: Forward result to external API
        await forward_to_external_api(
            token=user_id,
            analysis=analysis_data
        )

        # Step 4: Return same response back to caller
        return ProcessTextResponse(
            token=user_id,
            analysis=analysis_data
        )

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File encoding must be UTF-8"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the text file"
        )


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "service": "ai-interview-api",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info",
        access_log=True
    )