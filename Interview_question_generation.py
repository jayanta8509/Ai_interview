import os
import asyncio
import json
from re import S
import fitz
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel , Field
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class question(BaseModel):
    all_question: str= Field(description="The interview question")

class Question(BaseModel):
    all_question: list[question]



async def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
# OPTIMIZATION 1: Use faster model with optimized parameters
model = ChatOpenAI(
    model="gpt-4o-mini",  # Much faster than gpt-5-mini
    # # temperature=0.1,        # very low for deterministic output
    # max_tokens=800,    # Set explicit limit to prevent over-generation
    # request_timeout=25,           # Add timeout to prevent hanging
    # # streaming=False,      # Disable streaming for batch processing
    # # max_retries=2
)

SYSTEM_PROMPT = """You are conducting a real job interview. Speak naturally and conversationally - exactly how a professional interviewer would talk to a candidate face-to-face.

Your approach:
- Read the job description carefully - these are the requirements YOU need to assess
- Review the resume to understand what this candidate claims they can do
- Ask questions that naturally bridge between "what the job needs" and "what this person says they've done"

How real interviewers speak:
- Start with context: "I see on your resume..." or "This role really needs..." or "Looking at your background..."
- Be conversational: "Tell me about..." "Walk me through..." "How did you handle..." "What was your approach to..."
- Reference specific resume claims: "You mentioned you built X - can you elaborate on..."
- Connect to job needs: "Since this position requires a lot of Y, I'd love to hear..."
- Use follow-ups naturally: "Interesting - and then what happened?" "How did that work out?"

Question types to include (15-20 questions total):

1. TECHNICAL DEEP DIVES (about 6-8 questions):
   - "I notice you've worked with [specific tech from resume]. This role uses that heavily - can you walk me through a challenging situation where you had to..."
   - "The job description emphasizes [requirement from JD]. Your resume mentions [related experience]. Tell me about how you'd apply that here..."

2. EXPERIENCE VERIFICATION (about 4-5 questions):
   - "You wrote that you [achievement from resume]. That sounds impressive - can you break down what YOUR specific contribution was?"
   - "Looking at your time at [company], what would you say was your most meaningful project and why?"

3. ADDRESSING GAPS OR CONCERNS (about 2-3 questions):
   - If missing skill: "This job does require [missing skill]. I don't see that directly in your background - how have you approached similar challenges?"
   - If career switch: "I see you moved from [old field] to [new field]. What drove that decision and how has the transition been?"

4. BEHAVIORAL SITUATIONS (about 2-3 questions):
   - "Tell me about a time when you had to..."
   - "Give me an example of when you..."

5. CLOSING/MOTIVATION (about 1-2 questions):
   - "What specifically about this role excites you?"
   - "Where do you see yourself fitting in with what we're trying to build here?"

CRITICAL: Make every question sound like it's coming from a human interviewer who has actually read both documents and is making real-time connections between them. Avoid robotic, template-style questions."""


async def interview_question(resume: str, JD: str):
    agent = create_agent(model,
            response_format=ToolStrategy(Question),
            system_prompt=SYSTEM_PROMPT)

    context_message = f"""USER INPUT:
Job Description:
{JD}

Candidate Resume:
{resume}

Please generate tailored interview questions."""
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    ans = result["structured_response"]
    return ans

# if __name__ == "__main__":
#     resume = """Jayanta Roy 
# AI/ML Engineer — LLM & Multi-Agent Systems Specialist 
# 8017021283 | jayantameslova@gmail.com | LinkedIn | GitHub | Kolkata, India 
# PROFESSIONAL SUMMARY 
# Results-driven AI/ML Engineer with 2+ years building production AI systems serving 10K+ users. Specialized in LLM applications, multi-agent 
# architectures, and end-to-end ML pipelines using GPT-4, LangChain, and modern frameworks. Proven expertise delivering 95%+ model accuracy 
# and 40% efficiency improvements through scalable solutions with FastAPI, AWS, and GPU infrastructure. 
# EXPERIENCE 
# AI/ML  Engineer 
# Iksen India Pvt Ltd 
# Jul 2024 – Present 
# Kolkata, India 
# • Architected AI Question Generation System using GPT-4O mini, OpenCV, and Swarm framework with AWS S3 integration, 
# implementing pattern recognition and context-aware question generation following SDLC best practices. 
# • Built production-grade Virtual Try-On AI platform using Fooocus model and FastAPI, deployed on RunPod GPU servers, 
# generating photorealistic fashion visualizations with 95%+ accuracy for e-commerce applications. 
# • Engineered end-to-end AI video generation pipeline integrating GPT-4O for scripting, Eleven Labs for voice synthesis, 
# Stable Diffusion for image creation, and WAN 2.1 for animation, deployed on scalable RunPod infrastructure. 
# • Developed enterprise recruitment intelligence platform with three specialized GPT-4o AI agents for resume parsing, JD analysis, 
# and candidate matching, delivering 40% faster screening via FastAPI REST backend. 
# • Created multi-agent Resume Maker Tool with GPT-4o-mini, aggregating data from LinkedIn, GitHub, and portfolios to 
# generate ATS-optimized resumes with 95%+ compliance through async processing and structured JSON outputs. 
# Machine Learning Engineer 
# Paythrough Softwares and Solutions Pvt Ltd 
# Jun 2023 – Jun 2024 
# Kolkata, India 
# • Deployed AI Financial Advisor Platform integrating LangChain, CrewAI, AutoGen agents with OpenAI-ada-002 
# embeddings, Pinecone vector DB, IBM Watson transcription API, and Twilio REST API for real-time 
# advisor-client communication. 
# • Fine-tuned Mistral-7B on e-commerce FAQ dataset using PEFT with LoRA and Supervised Fine-tuning Trainer, achieving 30% 
# improvement in query understanding and response accuracy for customer support automation. 
# • Built production loan prediction and repayment models using SGD algorithm, NumPy, Pandas with comprehensive EDA and 
# feature engineering, achieving 85%+ accuracy in credit risk assessment. 
# • Designed dual-mode recommendation engine using SVD algorithm, delivering personalized product suggestions for 10K+ users 
# with selection sort optimization for new and existing customer segments. 
# TECHNICAL SKILLS 
# Languages: Python, C/C++, SQL 
# AI/ML Frameworks: LangChain, RAG, LangGraph, CrewAI, AutoGen, Swarm, Pydantic AI, MLOps, Scikit-Learn, TensorFlow, 
# Keras, PyTorch 
# LLM & Models: OpenAI GPT-4/4o, Gemini, DeepSeek, Anthropic Claude, Grok, Mistral, BERT, T5, Stable Diffusion, Hugging 
# Face 
# Databases: PostgreSQL, MySQL, Redis, Pinecone, FAISS, Chroma, Qdrant (Vector DBs) 
# DevOps & Cloud: Docker, Git, AWS (SageMaker, EC2, Lambda, S3, LightSail), RunPod GPU Servers, CI/CD Pipelines 
# APIs & Web: FastAPI, Flask, Quart, Django REST, RESTful APIs, OpenCV, Beautiful Soup 
# ML Techniques: NLP, Computer Vision, Supervised/Unsupervised Learning, Feature Engineering, XGBoost, SGD, PEFT, LoRA, 
# Model Fine-tuning 
# EDUCATION 
# Narula Institute of Technology (MAKAUT) 
# Bachelor of Technology in Computer Science and Engineering; CGPA: 8.10/10.0 
# South Calcutta Polytechnic (WBSCTE) 
# Diploma in Computer Science and Technology; Percentage: 71.90% 
# KEY ACHIEVEMENTS 
# Kolkata, India 
# Graduated 2023 
# Kolkata, India 
# Graduated 2020 
# • Deployed 8+ production AI systems serving 10K+ users with 95%+ model accuracy and ATS compliance 
# • Expertise in multi-agent AI architectures, LLM fine-tuning, and scalable ML pipeline development 
# • Proficient in full-stack AI deployment: model training, API development, cloud infrastructure, and CI/CD automation """

#     jd = """Job Title: Generative AI Engineer (Agentic AI)
# Location: Hyderabad (Work from Office)
# Experience: 4+ years
# Employment Type: Full-Time


# About the Role:

# We are seeking a highly skilled Generative AI Engineer to design, develop, and deploy intelligent AI systems, including LLMs, Generative AI, and Agentic AI applications. The ideal candidate will work on cutting-edge projects, collaborate with cross-functional teams, and drive innovation in AI solutions leveraging AWS tools and modern AI frameworks.


# Key Responsibilities:

# Design, develop, and maintain Python-based applications and AI services.
# Develop autonomous AI agents capable of reasoning, task execution, and multi-step decision-making.
# Work with LLMs, Generative AI models, and integrate with AWS services like Lambda, Lex, SageMaker, and S3.
# Collaborate with business and technical teams to understand requirements, propose solutions, and deliver high-quality implementations.
# Research and implement the latest AI libraries, tools, and frameworks to enhance functionality and performance.
# Troubleshoot, debug, and optimize applications and models for reliability and efficiency.
# Write clean, maintainable, and well-tested code; contribute to CI/CD pipelines and GitHub version control.
# Stay up-to-date with AI/ML trends, best practices, and Agentic AI developments.


# Qualifications & Skills:

# Bachelor’s degree in computer science, Engineering, or a related field.
# 4+ years of hands-on Python development experience for AI or backend applications.
# Strong experience with Generative AI, LLMs, and Agentic AI frameworks (LangChain, Llama Index, or similar).
# Proficiency in data manipulation and analysis using Pandas, NumPy, and database integrations.
# Knowledge of API integrations, multithreading, and structured programming.
# Experience with AWS AI/ML services and cloud-native application deployment.
# Familiarity with GitHub, CI/CD, and software development best practices.
# Excellent problem-solving, analytical, and communication skills"""

#     output = asyncio.run(interview_question(resume, jd))
#     print("=" * 50)
#     print("MISSING SKILLS (Required by JD but not in resume):")
#     print("=" * 50)
#     for ms in output.all_question:
#         print(f"- {ms.all_question}")
