import os
import asyncio
from typing import List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# ============= RESPONSE MODELS FOR TRANSCRIPT EVALUATOR =============

class SkillEvaluation(BaseModel):
    """Individual skill evaluation with percentage score"""
    skill_name: str = Field(description="Name of the skill (e.g., 'Python', 'SQL', 'Communication Skills')")
    description: str = Field(description="Detailed evaluation of the candidate's proficiency in this skill based on the interview")
    percentage: int = Field(description="Skill proficiency score out of 100", ge=0, le=100)


class QuestionAnalysis(BaseModel):
    """Analysis of individual question-response pair"""
    question: str = Field(description="The interview question asked")
    response_summary: str = Field(description="Summary of the candidate's response to this question")
    score: int = Field(description="Score for this specific response out of 100", ge=0, le=100)
    key_insights: List[str] = Field(description="Key positive points, strengths, or good observations from the response (3-5 points)")
    missed_opportunities: List[str] = Field(description="What the candidate missed or could have improved (2-4 points)")


class CategoryScore(BaseModel):
    """Score for each evaluation category - all out of 100"""
    technical_area: int = Field(description="Technical competence score out of 100", ge=0, le=100)
    communication_skills: int = Field(description="Communication skills score out of 100", ge=0, le=100)
    project_experience: int = Field(description="Project experience/expertise score out of 100", ge=0, le=100)
    behavioral_fit: int = Field(description="Behavioral fit score out of 100", ge=0, le=100)
    critical_thinking: int = Field(description="Critical thinking score out of 100", ge=0, le=100)


class AreaForImprovement(BaseModel):
    """Area where candidate needs to improve"""
    area: str = Field(description="The specific area that needs improvement")
    description: str = Field(description="Brief explanation of what needs work and why")


class InterviewEvaluationResponse(BaseModel):
    """Complete structured response for interview evaluation"""
    overall_score: int = Field(description="Total score out of 100", ge=0, le=100)
    must_to_have_skills: List[SkillEvaluation] = Field(description="Evaluation of mandatory skills from the job description")
    good_to_have_skills: List[SkillEvaluation] = Field(description="Evaluation of nice-to-have skills from the job description")
    interview_questions_responses: List[QuestionAnalysis] = Field(description="Question-wise analysis with scores, insights, and missed opportunities")
    category_scores: CategoryScore
    top_strengths: List[str] = Field(description="Top 3-5 strengths with specific examples from the interview", min_length=3, max_length=5)
    areas_for_improvement: List[AreaForImprovement] = Field(description="List of areas that need improvement (simple list, no severity labels)")
    overall_ai_summary: str = Field(description="Overall AI interview summary in approximately 300 words, written in a natural, professional tone like a real interviewer's summary")


# ============= MODEL & PROMPT =============

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=3000
    # request_timeout=40
)

EVALUATOR_SYSTEM_PROMPT = """You are an experienced interviewer who has just finished conducting a job interview. You need to provide a comprehensive evaluation of the candidate - exactly like you would write in your interview notes or share with the hiring team.

Your evaluation should feel like it was written by a real human interviewer who paid close attention to the conversation.

IMPORTANT CONTEXT:
- You have been provided with MANDATORY SKILLS (Must-to-Have) that are required for this role
- You have been provided with NICE-TO-HAVE SKILLS (Good-to-Have) that are preferred but not required
- The EXPERIENCE LEVEL indicates the expected depth for this role
- Evaluate candidates based on the EXPERIENCE LEVEL provided

EVALUATION STRUCTURE:

1. MUST-TO-HAVE SKILLS EVALUATION:
   - Evaluate EACH mandatory skill from the provided list
   - For each skill, provide:
     * skill_name: Exact name of the skill
     * description: Detailed assessment of candidate's proficiency based ONLY on interview responses. What did they demonstrate? What was missing? Reference specific quotes or examples from the transcript.
     * percentage: Score 0-100 based on demonstrated proficiency relative to the experience level
   - Be thorough and reference actual interview content

2. GOOD-TO-HAVE SKILLS EVALUATION:
   - Evaluate EACH nice-to-have skill from the provided list
   - Same structure as above: skill_name, description, percentage
   - These are bonus skills - lower scores are acceptable if candidate demonstrates strong mandatory skills

3. QUESTION-WISE ANALYSIS:
   For EACH question-response pair in the transcript, provide:
   - question: The exact question asked
   - response_summary: Concise summary of what the candidate said
   - score: 0-100 based on quality, depth, and relevance of the response
   - key_insights: 3-5 positive points - what did they do well? What stood out? Specific strengths demonstrated
   - missed_opportunities: 2-4 points - what could they have added? What details were missing? What would have made this answer stronger?

4. CATEGORY SCORES (all out of 100):
   - Technical Area: Technical knowledge, depth, problem-solving approach, clarity in explaining concepts
   - Communication Skills: Clarity, structure, conciseness, ability to explain complex things simply
   - Project Experience: Substantiality of experience, specific examples with outcomes, authenticity
   - Behavioral Fit: Team fit, adaptability, ownership, accountability, handling of challenges/conflicts
   - Critical Thinking: Logical reasoning, trade-off discussions, systematic problem solving

5. AREAS FOR IMPROVEMENT:
   - Simple list (no severity labels)
   - Focus on genuine gaps based on actual responses
   - Be fair and specific

6. TOP STRENGTHS:
   - 3-5 genuine strengths with specific examples from the interview

7. OVERALL AI SUMMARY (approximately 300 words):
   Write a natural, professional summary like a real interviewer would provide. Include:
   - Overall impression of the candidate
   - Technical capability highlights and concerns
   - Communication style and clarity
   - Fit for the role based on mandatory skills
   - Key strengths that stood out
   - Significant gaps or concerns
   - Final hiring recommendation (hire/no hire/maybe)

Write like you're talking to a colleague - not like a robot scoring an exam. Use phrases like:
- "What stood out to me was..."
- "I was particularly impressed when..."
- "That gave me pause because..."
- "The moment that really sold me was..."
- "I left the interview feeling..."

BASE YOUR EVALUATION ONLY ON WHAT WAS SAID IN THE INTERVIEW TRANSCRIPT. Do not assume skills or qualities that weren't demonstrated."""


# ============= EVALUATOR FUNCTION =============

async def evaluate_interview_transcript(
    transcript_data: str,
    experience: str,
    mandatory_skills: str,
    nice_to_have_skills: str
) -> InterviewEvaluationResponse:
    """
    Evaluate interview transcript and provide comprehensive assessment

    Args:
        transcript_data: Full interview transcript with Q&A pairs
        experience: Required experience level (e.g., "4+ years", "Senior", "Mid-level")
        mandatory_skills: Skills that are required for the role
        nice_to_have_skills: Bonus skills that are preferred but not required

    Returns:
        InterviewEvaluationResponse with skills evaluation, question analysis, scores, and summary
    """
    agent = create_agent(
        model,
        response_format=ToolStrategy(InterviewEvaluationResponse),
        system_prompt=EVALUATOR_SYSTEM_PROMPT
    )

    context_message = f"""INTERVIEW TRANSCRIPT:
{transcript_data}

JOB REQUIREMENTS:

EXPECTED EXPERIENCE LEVEL:
{experience}

MANDATORY SKILLS (Must-to-Have - Critical for this role):
{mandatory_skills}

NICE-TO-HAVE SKILLS (Good-to-Have - Preferred but not required):
{nice_to_have_skills}

Analyze this interview comprehensively. Evaluate each mandatory skill, each nice-to-have skill,
analyze each question-response pair, and provide an overall assessment."""
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    
    return result["structured_response"]


# ============= EXAMPLE USAGE =============

# async def main():
#     # Example transcript
#     sample_transcript = """
# Q1: Can you explain how you implemented the microservices architecture at your previous company?
# A: Well, we used microservices. The team decided to use Docker and Kubernetes. It was good.

# Q2: You mentioned improving system performance by 40%. How did you measure this?
# A: We just noticed things were faster. The users were happy. Everyone said it was better.

# Q3: Tell me about a time you had to resolve a conflict with a team member.
# A: I don't really have conflicts. I get along with everyone. We always agree on things.

# Q4: What interests you about this role?
# A: I need a job and the salary seems good. I heard your company has good benefits.

# Q5: Can you walk through your approach to debugging a production issue?
# A: I check the logs, see what's wrong, and fix it. Pretty straightforward.
# """

#     # Job requirements
#     experience = "4+ years"
#     mandatory_skills = """• Python development for AI/backend applications
# • Generative AI and LLMs (GPT-4, Claude, etc.)
# • LangChain or similar frameworks
# • Pandas and NumPy for data manipulation
# • AWS AI/ML services (Lambda, SageMaker, S3)"""

#     nice_to_have_skills = """• CI/CD pipelines and GitHub
# • Docker and containerization
# • Vector databases (Pinecone, FAISS, Chroma)
# • MLOps practices"""

#     # Evaluate transcript
#     evaluation = await evaluate_interview_transcript(
#         transcript_data=sample_transcript,
#         experience=experience,
#         mandatory_skills=mandatory_skills,
#         nice_to_have_skills=nice_to_have_skills
#     )
#     print(evaluation)

#     # Print formatted results
#     print(f"\n{'='*60}")
#     print(f"INTERVIEW EVALUATION REPORT")
#     print(f"{'='*60}")
#     print(f"\nOVERALL SCORE: {evaluation.overall_score}/100")

#     print(f"\n{'='*60}")
#     print("MUST-TO-HAVE SKILLS")
#     print(f"{'='*60}")
#     for skill in evaluation.must_to_have_skills:
#         print(f"\n{skill.skill_name}: {skill.percentage}%")
#         print(f"  {skill.description}")

#     print(f"\n{'='*60}")
#     print("GOOD-TO-HAVE SKILLS")
#     print(f"{'='*60}")
#     for skill in evaluation.good_to_have_skills:
#         print(f"\n{skill.skill_name}: {skill.percentage}%")
#         print(f"  {skill.description}")

#     print(f"\n{'='*60}")
#     print("QUESTION-WISE ANALYSIS")
#     print(f"{'='*60}")
#     for i, qa in enumerate(evaluation.interview_questions_responses, 1):
#         print(f"\nQ{i}: {qa.question[:80]}...")
#         print(f"Score: {qa.score}/100")
#         print(f"Summary: {qa.response_summary}")
#         print(f"Key Insights: {', '.join(qa.key_insights[:2])}")
#         print(f"Missed: {', '.join(qa.missed_opportunities[:2])}")

#     print(f"\n--- CATEGORY SCORES ---")
#     print(f"Technical Area:        {evaluation.category_scores.technical_area}/100")
#     print(f"Communication Skills:  {evaluation.category_scores.communication_skills}/100")
#     print(f"Project Experience:     {evaluation.category_scores.project_experience}/100")
#     print(f"Behavioral Fit:         {evaluation.category_scores.behavioral_fit}/100")
#     print(f"Critical Thinking:      {evaluation.category_scores.critical_thinking}/100")

#     print(f"\n--- TOP STRENGTHS ---")
#     for i, strength in enumerate(evaluation.top_strengths, 1):
#         print(f"{i}. {strength}")

#     print(f"\n--- AREAS FOR IMPROVEMENT ---")
#     for area in evaluation.areas_for_improvement:
#         print(f"\n• {area.area}")
#         print(f"  {area.description}")

#     print(f"\n{'='*60}")
#     print("OVERALL AI SUMMARY")
#     print(f"{'='*60}")
#     print(evaluation.overall_ai_summary)


# if __name__ == "__main__":
#     asyncio.run(main())