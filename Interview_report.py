import os
import asyncio
from typing import List, Optional
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# ============= RESPONSE MODELS FOR TRANSCRIPT EVALUATOR =============

class CategoryScore(BaseModel):
    """Score for each evaluation category"""
    technical: int = Field(description="Technical competence score out of 30", ge=0, le=30)
    communication: int = Field(description="Communication skills score out of 20", ge=0, le=20)
    experience: int = Field(description="Experience validation score out of 20", ge=0, le=20)
    culture_fit: int = Field(description="Cultural fit score out of 15", ge=0, le=15)
    critical_thinking: int = Field(description="Critical thinking score out of 15", ge=0, le=15)


class Mistake(BaseModel):
    """Individual mistake or issue identified"""
    question_number: Optional[int] = Field(default=None, description="Which question this relates to")
    mistake: str = Field(description="Description of the mistake")
    what_they_said: str = Field(description="Quote or paraphrase of problematic response")
    why_problematic: str = Field(description="Why this is an issue")
    correct_approach: str = Field(description="What they should have done/said")


class MistakesByseverity(BaseModel):
    """Mistakes categorized by severity"""
    critical: List[Mistake] = Field(description="High impact issues")
    medium: List[Mistake] = Field(description="Medium impact issues")
    minor: List[Mistake] = Field(description="Minor issues")


class QuestionScore(BaseModel):
    """Score and feedback for individual question"""
    question_number: int = Field(description="Question number")
    question_text: str = Field(description="The question asked")
    answer_summary: str = Field(description="Brief summary of candidate's answer")
    score: int = Field(description="Score out of 10", ge=0, le=10)
    feedback: str = Field(description="Issues identified or 'Strong answer'")


class RiskAssessment(BaseModel):
    """Hiring risk levels"""
    technical_risk: str = Field(description="Low, Medium, or High")
    culture_risk: str = Field(description="Low, Medium, or High")
    performance_risk: str = Field(description="Low, Medium, or High")


class InterviewEvaluationResponse(BaseModel):
    """Complete structured response for interview evaluation"""
    overall_score: int = Field(description="Total score out of 100", ge=0, le=100)
    category_scores: CategoryScore
    top_strengths: List[str] = Field(description="Top 3-4 strengths with specific examples", min_length=3, max_length=4)
    mistakes: MistakesByseverity
    key_concerns: List[str] = Field(description="Patterns or recurring issues", max_length=3)
    question_scores: List[QuestionScore] = Field(description="Detailed analysis of each Q&A")
    recommendation: str = Field(description="Strong Hire, Hire, Maybe, No Hire, or Strong No Hire")
    recommendation_rationale: str = Field(description="2-3 sentences explaining the decision")
    risk_assessment: RiskAssessment


# ============= MODEL & PROMPT =============

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=3000,
    request_timeout=40
)

EVALUATOR_SYSTEM_PROMPT = """Analyze interview transcript. Score candidate and identify all mistakes.

Score Breakdown (Total: 100):
  Technical (30): Accuracy, depth, problem-solving
  Communication (20): Clarity, structure, relevance
  Experience (20): Specific examples, STAR format, results
  Culture Fit (15): Teamwork, adaptability, attitude
  Critical Thinking (15): Logic, structured approach, trade-offs

Red Flags to Catch:
  Vague answers ("we/team" without "I"), can't explain claimed skills, contradictions, blame-shifting, rambling, not answering question, exaggeration, poor attitude, no ownership of failures

Scoring Guide:
  90-100: Exceptional
  80-89: Strong hire
  70-79: Good hire
  60-69: Marginal
  Below 60: Do not hire

Quote exact problematic statements. Be specific and direct."""


# ============= EVALUATOR FUNCTION =============

async def evaluate_interview_transcript(transcript: str) -> InterviewEvaluationResponse:
    """
    Evaluate interview transcript and provide comprehensive assessment
    
    Args:
        transcript: Full interview transcript with Q&A pairs
        
    Returns:
        InterviewEvaluationResponse with scores, mistakes, and recommendation
    """
    agent = create_agent(
        model,
        response_format=ToolStrategy(InterviewEvaluationResponse),
        system_prompt=EVALUATOR_SYSTEM_PROMPT
    )

    context_message = f"""Interview Transcript:

{transcript}

Analyze this interview and provide comprehensive evaluation."""
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": context_message}]}
    )
    
    return result["structured_response"]


# ============= EXAMPLE USAGE =============

async def main():
    # Example transcript
    sample_transcript = """
Q1: Can you explain how you implemented the microservices architecture at your previous company?
A: Well, we used microservices. The team decided to use Docker and Kubernetes. It was good.

Q2: You mentioned improving system performance by 40%. How did you measure this?
A: We just noticed things were faster. The users were happy. Everyone said it was better.

Q3: Tell me about a time you had to resolve a conflict with a team member.
A: I don't really have conflicts. I get along with everyone. We always agree on things.

Q4: What interests you about this role?
A: I need a job and the salary seems good. I heard your company has good benefits.

Q5: Can you walk through your approach to debugging a production issue?
A: I check the logs, see what's wrong, and fix it. Pretty straightforward.
"""
    
    # Evaluate transcript
    evaluation = await evaluate_interview_transcript(sample_transcript)
    print(evaluation)
    
    # print(f"=== OVERALL SCORE: {evaluation.overall_score}/100 ===\n")
    
    # print("=== CATEGORY SCORES ===")
    # print(f"Technical: {evaluation.category_scores.technical}/30")
    # print(f"Communication: {evaluation.category_scores.communication}/20")
    # print(f"Experience: {evaluation.category_scores.experience}/20")
    # print(f"Culture Fit: {evaluation.category_scores.culture_fit}/15")
    # print(f"Critical Thinking: {evaluation.category_scores.critical_thinking}/15")
    
    # print("\n=== TOP STRENGTHS ===")
    # for i, strength in enumerate(evaluation.top_strengths, 1):
    #     print(f"{i}. {strength}")
    
    # print("\n=== CRITICAL MISTAKES ===")
    # for mistake in evaluation.mistakes.critical:
    #     print(f"\nQ{mistake.question_number}: {mistake.mistake}")
    #     print(f"  Said: \"{mistake.what_they_said}\"")
    #     print(f"  Problem: {mistake.why_problematic}")
    #     print(f"  Should be: {mistake.correct_approach}")
    
    # print("\n=== KEY CONCERNS ===")
    # for concern in evaluation.key_concerns:
    #     print(f"  - {concern}")
    
    # print("\n=== QUESTION-BY-QUESTION SCORES ===")
    # for qs in evaluation.question_scores:
    #     print(f"\nQ{qs.question_number}: {qs.question_text}")
    #     print(f"  Score: {qs.score}/10")
    #     print(f"  Feedback: {qs.feedback}")
    
    # print(f"\n=== RECOMMENDATION: {evaluation.recommendation} ===")
    # print(f"{evaluation.recommendation_rationale}")
    
    # print("\n=== RISK ASSESSMENT ===")
    # print(f"Technical Risk: {evaluation.risk_assessment.technical_risk}")
    # print(f"Culture Risk: {evaluation.risk_assessment.culture_risk}")
    # print(f"Performance Risk: {evaluation.risk_assessment.performance_risk}")


if __name__ == "__main__":
    asyncio.run(main())