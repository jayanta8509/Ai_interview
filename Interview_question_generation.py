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
- Focus on MANDATORY SKILLS - these are non-negotiable requirements
- Consider NICE-TO-HAVE SKILLS as bonus areas to explore if time permits
- Match the depth of questions to the candidate's EXPERIENCE level
- Ask questions that naturally bridge between "what the job needs" and "what this person says they've done"

How real interviewers speak:
- Start with context: "I see on your resume..." or "This role really needs..." or "Looking at your background..."
- Be conversational: "Tell me about..." "Walk me through..." "How did you handle..." "What was your approach to..."
- Reference specific resume claims: "You mentioned you built X - can you elaborate on..."
- Connect to job needs: "Since this position requires a lot of Y, I'd love to hear..."
- Use follow-ups naturally: "Interesting - and then what happened?" "How did that work out?"

**IMPORTANT: Generate questions for a 1-HOUR interview approximately 12-15 questions MAX**

Question distribution (total 12-15 questions for 1 hour):

1. INTRODUCTION & EXPERIENCE OVERVIEW (1-2 questions - 5 minutes):
   - "Walk me through your background and how it led you to this role."
   - Based on their EXPERIENCE level, ask about their career journey.

2. MANDATORY SKILLS ASSESSMENT (5-7 questions - 30-35 minutes):
   - PRIORITIZE these heavily - every mandatory skill MUST be covered
   - "I see you've worked with [mandatory skill]. This role requires strong proficiency here. Can you walk me through a challenging project where you used..."
   - "The job description emphasizes [mandatory skill]. Your resume mentions [related experience]. Tell me about how you'd apply that here..."
   - If missing a mandatory skill: "This role requires [mandatory skill]. I don't see that directly in your background. Can you share how you've approached similar challenges or your plan to get up to speed?"

3. NICE-TO-HAVE SKILLS EXPLORATION (2-3 questions - 10-12 minutes):
   - Only if candidate has these on their resume OR if they seem promising
   - "I noticed you have experience with [nice-to-have skill]. That's a bonus for this role - tell me about..."
   - "We're also looking for [nice-to-have skill]. Have you had any exposure to this?"

4. EXPERIENCE VERIFICATION (2-3 questions - 8-10 minutes):
   - Based on stated EXPERIENCE, ask depth-appropriate questions:
   - For senior roles: "You mentioned you [achievement]. That sounds impressive - can you break down your specific contribution and the technical decisions you made?"
   - For mid-level roles: "Looking at your time at [company], what would you say was your most meaningful project and why?"
   - For junior roles: "Which project from your resume did you learn the most from?"

5. BEHAVIORAL SITUATIONS (1-2 questions - 5-6 minutes):
   - "Tell me about a time when you had to..."
   - "Give me an example of when you..."

6. CLOSING (1 question - 3-4 minutes):
   - "What specifically about this role excites you?"
   - "Do you have any questions for me about the position or team?"

CRITICAL: Make every question sound like it's coming from a human interviewer who has actually read both documents and is making real-time connections between them. Avoid robotic, template-style questions."""


async def interview_question(
    resume_data: str,
    Job_Description: str,
    Experience: str,
    Mandatory_skills: str,
    Nice_to_have_skills: str
):
    """
    Generate tailored interview questions based on resume, job description, and skill requirements.

    Args:
        resume_data: Candidate's resume text
        Job_Description: Full job description
        Experience: Required experience level (e.g., "4+ years", "Senior", "Mid-level")
        Mandatory_skills: Skills that are required for the role
        Nice_to_have_skills: Bonus skills that are preferred but not required
    """
    agent = create_agent(model,
            response_format=ToolStrategy(Question),
            system_prompt=SYSTEM_PROMPT)

    context_message = f"""USER INPUT:

JOB DESCRIPTION:
{Job_Description}

CANDIDATE EXPERIENCE LEVEL:
{Experience}

MANDATORY SKILLS (Must assess these thoroughly):
{Mandatory_skills}

NICE-TO-HAVE SKILLS (Explore if time permits/candidate has experience):
{Nice_to_have_skills}

CANDIDATE RESUME:
{resume_data}

Please generate 12-15 tailored interview questions suitable for a 1-hour interview session.
Prioritize assessment of MANDATORY SKILLS while covering all relevant aspects of the candidate's background."""
    
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

#     # Extract parameters from JD
#     experience = "4+ years"

#     mandatory_skills = """• 4+ years of hands-on Python development experience for AI or backend applications
# • Strong experience with Generative AI, LLMs, and Agentic AI frameworks (LangChain, Llama Index, or similar)
# • Proficiency in data manipulation and analysis using Pandas, NumPy, and database integrations
# • Knowledge of API integrations, multithreading, and structured programming
# • Experience with AWS AI/ML services (Lambda, Lex, SageMaker, S3) and cloud-native application deployment
# • Bachelor's degree in Computer Science, Engineering, or related field"""

#     nice_to_have_skills = """• CI/CD pipelines and GitHub
# • Docker and containerization
# • Vector databases (Pinecone, FAISS, Chroma)
# • Additional cloud platforms beyond AWS
# • MLOps practices and tools"""

#     output = asyncio.run(interview_question(
#         resume_data=resume,
#         Job_Description=jd,
#         Experience=experience,
#         Mandatory_skills=mandatory_skills,
#         Nice_to_have_skills=nice_to_have_skills
#     ))
#     print("=" * 60)
#     print("INTERVIEW QUESTIONS (12-15 questions for 1 hour):")
#     print("=" * 60)
#     for i, q in enumerate(output.all_question, 1):
#         print(f"\nQ{i}: {q.all_question}")
