# AI Interview API

A production-ready FastAPI application for generating interview questions from resumes and evaluating interview transcripts using AI.

## Features

- **Interview Question Generation**: Upload a resume (PDF) along with a Job Description to generate tailored interview questions
- **Interview Transcript Evaluation**: Upload an interview transcript (text file) to get comprehensive evaluation and hiring recommendations
- **Production-Ready**: Built with FastAPI, includes input validation, error handling, logging, and security middleware
- **AI-Powered**: Uses OpenAI GPT-4o-mini for intelligent analysis and question generation

## API Endpoints

### 1. Generate Interview Questions
**POST** `/api/v1/process-pdf`

Upload a PDF resume and job description to generate targeted interview questions.

**Request:**
- `pdf_file`: PDF file (resume)
- `jd_text`: Job description as text (1-10,000 characters)

**Response:**
```json
{
  "success": true,
  "message": "PDF and JD processed successfully",
  "file_name": "resume.pdf",
  "file_size": 646078,
  "jd_length": 2004,
  "response": [
    "Question 1...",
    "Question 2...",
    ...
  ]
}
```

### 2. Evaluate Interview Transcript
**POST** `/api/v1/process-text`

Upload an interview transcript to get comprehensive evaluation with scores, mistakes, and hiring recommendation.

**Request:**
- `text_file`: Text file (.txt) containing interview transcript with Q&A pairs

**Response:**
```json
{
  "success": true,
  "message": "Text file processed successfully",
  "file_name": "transcript.txt",
  "file_size": 1234,
  "response": {
    "overall_score": 45,
    "category_scores": {
      "technical": 12,
      "communication": 8,
      "experience": 10,
      "culture_fit": 8,
      "critical_thinking": 7
    },
    "top_strengths": [...],
    "mistakes": {
      "critical": [...],
      "medium": [...],
      "minor": [...]
    },
    "key_concerns": [...],
    "question_scores": [...],
    "recommendation": "No Hire",
    "recommendation_rationale": "...",
    "risk_assessment": {...}
  }
}
```

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. Clone the repository
```bash
git clone <repository-url>
cd Ai_interview
```

2. Create virtual environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running the Server

```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Project Structure

```
Ai_interview/
├── app.py                           # Main FastAPI application
├── Interview_question_generation.py   # PDF text extraction & question generation
├── Interview_report.py               # Transcript evaluation logic
├── requirements.txt                  # Python dependencies
├── test_transcript.txt              # Sample transcript for testing
├── .env                            # Environment variables (create this)
└── temp_uploads/                   # Temporary file storage (auto-created)
```

## Evaluation Criteria

The interview transcript evaluator scores candidates out of 100 points:

| Category | Points | Description |
|----------|--------|-------------|
| Technical | 30 | Accuracy, depth, problem-solving |
| Communication | 20 | Clarity, structure, relevance |
| Experience | 20 | Specific examples, STAR format, results |
| Culture Fit | 15 | Teamwork, adaptability, attitude |
| Critical Thinking | 15 | Logic, structured approach, trade-offs |

**Scoring Guide:**
- 90-100: Exceptional
- 80-89: Strong hire
- 70-79: Good hire
- 60-69: Marginal
- Below 60: Do not hire

## File Upload Limits

- **Max file size**: 10 MB
- **Accepted formats**:
  - PDF endpoint: `.pdf` only
  - Text endpoint: `.txt`, `.text`
- **Encoding**: UTF-8 for text files

## Testing

### Test the PDF Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/process-pdf" \
  -F "pdf_file=@path/to/resume.pdf" \
  -F "jd_text=Job description text here..."
```

### Test the Transcript Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/process-text" \
  -F "text_file=@test_transcript.txt"
```

Or use the provided test file:
```bash
curl -X POST "http://localhost:8000/api/v1/process-text" \
  -F "text_file=@test_transcript.txt"
```

## Configuration

### Production Settings

Update the following in `app.py` for production:

```python
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TrustedHost middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["yourdomain.com"]  # Specify allowed hosts
)
```

### Disable Auto Reload

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Dependencies

- **FastAPI** - Modern web framework for building APIs
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation using Python type annotations
- **LangChain** - Framework for building AI applications
- **OpenAI** - GPT-4o-mini model for AI processing
- **PyMuPDF (fitz)** - PDF text extraction
- **python-dotenv** - Environment variable management

## Security Considerations

- File size validation prevents DoS attacks
- File extension validation prevents malicious uploads
- Temporary files are cleaned up after processing
- CORS and TrustedHost middleware for additional security
- Input sanitization via Pydantic models

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]
