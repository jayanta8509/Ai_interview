"""
FastAPI Application with two endpoints for file processing.
Production-ready with proper validation, error handling, and security features.
"""

# import imp
import os
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from urllib3 import request
from Interview_question_generation import extract_text_from_pdf,interview_question
from Interview_report import evaluate_interview_transcript


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


# Pydantic models for request/response
class ProcessPDFResponse(BaseModel):
    success: bool
    message: str
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    jd_length: Optional[int] = None
    response: Optional[list] = None


class ProcessTextResponse(BaseModel):
    success: bool
    message: str
    file_name: str
    file_size: int
    content_preview: Optional[str] = None
    response: Optional[dict] = None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    detail: Optional[str] = None


# Allowed file extensions
ALLOWED_PDF_EXTENSIONS = {".pdf"}
ALLOWED_TEXT_EXTENSIONS = {".txt", ".text"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def validate_file_size(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> None:
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
    )
) -> ProcessPDFResponse:
    """
    Endpoint to process a PDF file along with job description text.

    - **pdf_file**: PDF file upload (required)
    - **jd_text**: Job description as plain text (required, 1-10000 characters)

    Returns processing status and metadata.
    """
    try:
        # Validate PDF file extension
        validate_file_extension(pdf_file.filename, ALLOWED_PDF_EXTENSIONS)

        # Validate file size
        validate_file_size(pdf_file.file)

        # Read file content (for demonstration)
        content = await pdf_file.read()
        file_size = len(content)

        # Log the processing
        logger.info(
            f"Processing PDF: {pdf_file.filename}, "
            f"Size: {file_size} bytes, "
            f"JD length: {len(jd_text)} characters"
        )

        # TODO: Add your actual PDF processing logic here
        # Example: extract text from PDF, match with JD, etc.

        # Save uploaded file to temp directory for processing
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, pdf_file.filename)

        # Write the uploaded file to disk
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(content)

        # Extract text from PDF using the file path
        data = await extract_text_from_pdf(temp_file_path)

        # Generate interview questions
        result = await interview_question(data, jd_text)

        # Clean up temp file
        os.remove(temp_file_path)

        return ProcessPDFResponse(
            success=True,
            message="PDF and JD processed successfully",
            file_name=pdf_file.filename,
            file_size=file_size,
            jd_length=len(jd_text),
            response= [question.all_question for question in result.all_question]
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
        422: {"model": ErrorResponse, "description": "Validation Error"}
    },
    tags=["Text Processing"],
    summary="Process Text File",
    description="Upload a text file for processing"
)
async def process_text_file(
    text_file: UploadFile = File(
        ...,
        description="Text file to be processed",
        media_type="text/plain"
    )
) -> ProcessTextResponse:
    """
    Endpoint to process a text file.

    - **text_file**: Text file upload (.txt or .text) (required)

    Returns processing status and content preview.
    """
    try:
        # Validate text file extension
        validate_file_extension(text_file.filename, ALLOWED_TEXT_EXTENSIONS)

        # Validate file size
        validate_file_size(text_file.file)

        # Read file content
        content_bytes = await text_file.read()
        content = content_bytes.decode('utf-8')
        file_size = len(content_bytes)

        # Evaluate interview transcript using the content string
        result = await evaluate_interview_transcript(content)

        # Log the processing
        logger.info(
            f"Processing text file: {text_file.filename}, "
            f"Size: {file_size} bytes"
        )

        # TODO: Add your actual text processing logic here

        return ProcessTextResponse(
            success=True,
            message="Text file processed successfully",
            file_name=text_file.filename,
            file_size=file_size,
            response=result.model_dump() if result else None
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

    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8004,
        reload=True,  # Set to False in production
        log_level="info",
        access_log=True
    )
