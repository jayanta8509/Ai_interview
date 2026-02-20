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
    category_scores: CategoryScore
    top_strengths: List[str] = Field(description="Top 3-5 strengths with specific examples from the interview", min_length=3, max_length=5)
    areas_for_improvement: List[AreaForImprovement] = Field(description="List of areas that need improvement (simple list, no severity labels)")
    recommendation_rationale: str = Field(description="Detailed recommendation rationale in natural human tone, 700-800 words, written like how a real interviewer would provide feedback after an interview")


# ============= MODEL & PROMPT =============

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=3000,
    request_timeout=40
)

EVALUATOR_SYSTEM_PROMPT = """You are an experienced interviewer who has just finished conducting a job interview. You need to provide a comprehensive evaluation of the candidate - exactly like you would write in your interview notes or share with the hiring team.

Your evaluation should feel like it was written by a real human interviewer who paid close attention to the conversation.

SCORING (all out of 100):

1. Technical Area: Assess their technical knowledge, depth of understanding, problem-solving approach, and ability to explain technical concepts clearly. Did they demonstrate genuine expertise or was it surface-level?

2. Communication Skills: How clearly did they express themselves? Did they structure their answers well? Were they concise or rambling? Could they explain complex things simply? Did they listen well and answer what was actually asked?

3. Project Experience/Expertise: How substantial and relevant is their actual project experience? Did they provide specific examples with real outcomes? Could they speak authentically about their contributions? Was there depth in their project discussions?

4. Behavioral Fit: How well would they fit the team and culture? Did they demonstrate teamwork, adaptability, ownership, accountability? How did they handle questions about challenges, conflicts, or failures?

5. Critical Thinking: Did they show logical reasoning? Could they discuss trade-offs? Did they think through problems systematically? Could they handle "what if" scenarios thoughtfully?

AREAS FOR IMPROVEMENT:
- Create a simple list (no severity labels like "critical" or "minor")
- Focus on genuine gaps or concerns based on their actual responses
- Be fair and specific - reference what they said that indicates the need for improvement
- Only list real issues, not nitpicks

TOP STRENGTHS:
- Identify 3-5 genuine strengths based on their interview performance
- Reference specific things they said or demonstrated
- Be authentic - not every candidate has 5 strengths, but most have at least 2-3

FINAL RECOMMENDATION RATIONALE (700-800 words):
This is the most important part. Write it like a real interviewer providing thoughtful feedback to the hiring manager. Use a natural, conversational human tone.

Structure your rationale like this:

1. OPENING (50-100 words): Start with your overall impression. "Having interviewed [candidate name] for [position], I came away with..." or "My conversation with [candidate] left me feeling..."

2. TECHNICAL ASSESSMENT (150-200 words): Discuss their technical capabilities naturally. "When we dug into [specific topic], I was impressed by..." or "I had some concerns when they couldn't explain..." Reference actual exchanges from the interview.

3. EXPERIENCE & EXPERTISE (100-150 words): Talk about their project experience authentically. "Their work on [project] stood out because..." or "I struggled to get concrete details about..." Mention what felt genuine vs. what felt exaggerated.

4. COMMUNICATION & PRESENCE (100-150 words): Describe how they came across. "They communicated with confidence and clarity when..." or "I noticed they tended to ramble when..." Be honest about how they would present to stakeholders or clients.

5. BEHAVIORAL/CULTURAL FIT (100-150 words): Discuss team fit. "I could see them fitting in well because..." or "I have some concerns about..." Reference behavioral questions and their responses.

6. CLOSING & RECOMMENDATION (100-150 words): Sum up with a clear hiring recommendation. "Overall, I would [recommend/not recommend] moving forward because..." Be decisive but nuanced.

Write like you're talking to a colleague - not like a robot scoring an exam. Use phrases like:
- "What stood out to me was..."
- "I was particularly impressed when..."
- "That gave me pause because..."
- "The moment that really sold me was..."
- "I left the interview feeling..."

BASE YOUR EVALUATION ONLY ON WHAT WAS SAID IN THE INTERVIEW TRANSCRIPT. Do not assume skills or qualities that weren't demonstrated."""


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
    
#     # Evaluate transcript
#     evaluation = await evaluate_interview_transcript(sample_transcript)
#     print(evaluation)

#     # Print formatted results
#     print(f"\n{'='*60}")
#     print(f"INTERVIEW EVALUATION REPORT")
#     print(f"{'='*60}")
#     print(f"\nOVERALL SCORE: {evaluation.overall_score}/100")
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
#     print("FINAL RECOMMENDATION RATIONALE")
#     print(f"{'='*60}")
#     print(evaluation.recommendation_rationale)


# if __name__ == "__main__":
#     asyncio.run(main())