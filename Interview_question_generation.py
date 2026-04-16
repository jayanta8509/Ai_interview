"""
Interview Question Generation Module
-------------------------------------
Generates calibrated interview questions based on:
  - Candidate's resume
  - Job description (including company context)
  - Required experience level
  - Mandatory and nice-to-have skills

FIXES APPLIED:
  1. Replaced synchronous create_agent().invoke() (which blocks the async event
     loop) with model.with_structured_output() + asyncio.to_thread() — the
     correct async-safe pattern for LangChain structured output.
  2. Completely rewrote SYSTEM_PROMPT with explicit, rules-based calibration:
       - Experience tier -> question depth/complexity matrix
       - Company-type detection from JD -> interview style/culture calibration
       - Mandatory vs nice-to-have skill weighting
  3. Restructured context_message so the LLM sees calibration directives FIRST,
     before the content — higher priority parsing.
"""

import os
import asyncio

import fitz  # PyMuPDF
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import ChatOpenAI


# ----------------------------------------------------------
#  Pydantic models for structured LLM output
# ----------------------------------------------------------

class question(BaseModel):
    all_question: str = Field(description="A single interview question")


class Question(BaseModel):
    all_question: list[question] = Field(
        description="List of 12-15 calibrated interview questions"
    )


# ----------------------------------------------------------
#  PDF extraction
# ----------------------------------------------------------

async def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from every page of a PDF."""
    def _extract():
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    return await asyncio.to_thread(_extract)


# ----------------------------------------------------------
#  LLM — gpt-4o-mini with structured output
# ----------------------------------------------------------

_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,  # slight creativity while staying focused
)

# Bind the structured output schema once; reuse across all requests
_structured_model = _model.with_structured_output(Question)


# ----------------------------------------------------------
#  System prompt with explicit calibration rules
# ----------------------------------------------------------

SYSTEM_PROMPT = """
You are a senior technical interviewer generating a set of calibrated interview questions.
Your primary job is NOT to ask generic questions. It is to craft questions that match
the EXACT LEVEL of this candidate for THIS specific company and role.

=================================================================
 STEP 1 -- DETERMINE THE EXPERIENCE TIER
=================================================================
Read the EXPERIENCE field in the user message and classify the candidate:

  TIER 1 -- JUNIOR (0-2 years / fresher / intern / entry-level)
    - Focus on: fundamentals, conceptual understanding, learning ability
    - Depth: "What is X?", "How does X work?", "Have you used X?"
    - Avoid: system design, architecture decisions, team leadership topics

  TIER 2 -- MID-LEVEL (2-5 years / associate / software engineer II)
    - Focus on: ownership, debugging, feature delivery, trade-off awareness
    - Depth: "Walk me through how you built X", "What would you do differently?"
    - Include: some design questions, cross-team collaboration scenarios

  TIER 3 -- SENIOR (5-10 years / senior / lead / staff)
    - Focus on: architecture, scale, technical leadership, mentorship impact
    - Depth: "Design a system that...", "How would you migrate X to Y at scale?"
    - Include: system design, org-level decisions, technical strategy

  TIER 4 -- PRINCIPAL / DIRECTOR / VP (10+ years / staff+ / principal / director)
    - Focus on: org-wide strategy, cross-functional influence, build-vs-buy decisions
    - Depth: "How would you align an engineering roadmap with business goals?"
    - Include: platform strategy, technical vision, stakeholder management

IF EXPERIENCE IS AMBIGUOUS OR NOT PROVIDED -- infer tier from the resume's highest role title and total years.

=================================================================
 STEP 2 -- DETERMINE COMPANY TYPE FROM THE JD
=================================================================
Read the JD carefully. Infer the company type and calibrate question style:

  STARTUP / EARLY-STAGE
    - Signals: "fast-paced", "wear many hats", "ownership", "early team", "Series A/B"
    - Style: practical, breadth-first, "how would you build X from scratch with limited resources?",
             emphasize initiative and adaptability, less process / more execution

  MID-SIZE TECH COMPANY
    - Signals: "growing team", "scale", "cross-functional", moderate headcount
    - Style: balance of depth and breadth, process maturity, team collaboration,
             some system design at appropriate tier

  LARGE ENTERPRISE / FAANG-TIER
    - Signals: well-known brand, "at scale", "millions of users", "distributed systems",
               structured interviews, "bar raiser"
    - Style: deep system design (Tier 3+), coding depth, behavioral STAR format,
             "tell me about a time you influenced without authority"

  TRADITIONAL / NON-TECH INDUSTRY (finance, healthcare, manufacturing, government)
    - Signals: domain-specific language, regulatory compliance, legacy systems
    - Style: blend of technical + domain knowledge, reliability over novelty,
             "how have you worked with legacy systems or compliance requirements?"

  IF UNCLEAR -- default to Mid-Size Tech Company style.

=================================================================
 STEP 3 -- SKILL COVERAGE RULES
=================================================================
  MANDATORY SKILLS: Cover EVERY mandatory skill with at least one question.
    These are non-negotiable. If the candidate lacks a mandatory skill, ask:
    "This role requires [skill]. I don't see that in your background.
     How would you approach learning it or handling that gap?"

  NICE-TO-HAVE SKILLS: Ask about these ONLY if the candidate lists them on the resume.
    Do NOT ask about nice-to-have skills the candidate has zero exposure to.

=================================================================
 STEP 4 -- QUESTION DISTRIBUTION (12-15 total)
=================================================================
  1. INTRO / BACKGROUND           1-2 questions  (calibrated to tier)
  2. MANDATORY SKILL DEPTH        5-7 questions  (at least one per mandatory skill)
  3. NICE-TO-HAVE EXPLORATION     2-3 questions  (only if candidate has these)
  4. EXPERIENCE VERIFICATION      2-3 questions  (depth matched to tier + company type)
  5. BEHAVIORAL / SITUATIONAL     1-2 questions  (STAR format, tier-appropriate stakes)
  6. CLOSING                      1 question     ("What excites you about this role?")

=================================================================
 STYLE RULES
=================================================================
  - Sound like a human who has READ both documents
  - Reference specific resume items: "I see you worked on X at Company Y..."
  - Connect to the JD: "This role requires a lot of Z, so I would like to explore..."
  - Scale complexity: Tier 1 gets "what/how" questions; Tier 4 gets "why/strategy" questions
  - NO generic template questions like "What is your greatest weakness?"
  - NO questions that are answered by simply reading the resume
  - Generate EXACTLY 12-15 questions total — no more, no fewer
"""


# ----------------------------------------------------------
#  Main generation function
# ----------------------------------------------------------

async def interview_question(
    resume_data: str,
    Job_Description: str,
    Experience: str,
    Mandatory_skills: str,
    Nice_to_have_skills: str,
) -> Question:
    """
    Generate calibrated interview questions.

    Args:
        resume_data:         Full text of the candidate's resume
        Job_Description:     Full text of the job description
        Experience:          Experience requirement (e.g. "4+ years", "Senior", "0-1 year")
        Mandatory_skills:    Comma/newline-separated required skills
        Nice_to_have_skills: Comma/newline-separated preferred skills

    Returns:
        Question pydantic model containing a list of question objects
    """

    # Calibration directives come FIRST so the LLM processes them at the
    # highest priority before reading the content blocks below.
    context_message = f"""
CALIBRATION DIRECTIVES (read these before generating questions)
================================================================

EXPERIENCE LEVEL:
{Experience.strip() if Experience.strip() else "Not explicitly provided -- infer from resume"}

MANDATORY SKILLS (must cover every skill with at least one question):
{Mandatory_skills.strip() if Mandatory_skills.strip() else "Not provided -- infer required skills from the JD below"}

NICE-TO-HAVE SKILLS (ask only if the candidate lists these on their resume):
{Nice_to_have_skills.strip() if Nice_to_have_skills.strip() else "Not provided -- skip this section"}

Using the rules in your instructions:
  1. Identify the EXPERIENCE TIER from the experience field above.
  2. Infer the COMPANY TYPE from the job description below.
  3. Generate 12-15 questions calibrated to BOTH the tier and company type.

================================================================
JOB DESCRIPTION
================================================================
{Job_Description}

================================================================
CANDIDATE RESUME
================================================================
{resume_data}

================================================================
REMINDER FOR YOUR OUTPUT
================================================================
- Generate exactly 12-15 interview questions.
- Every mandatory skill listed above MUST appear in at least one question.
- Calibrate complexity strictly to the experience tier and company type you identified.
- Reference specific items from the resume and JD — do not ask generic questions.
"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context_message),
    ]

    # Run the synchronous LangChain call in a thread pool executor so it does
    # not block the FastAPI async event loop.
    result: Question = await asyncio.to_thread(_structured_model.invoke, messages)
    return result
