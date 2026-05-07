VALID_LABELS = {"correct", "partial", "incorrect"}

_SYSTEM_PROMPT = """
You are an examiner for oral answers in a machine learning .

The answer you receive is a spoken answer that was automatically transcribed from speech to text.

Important:
- Ignore filler words, hesitations, repeated words, and broken spoken syntax.
- Judge the mathematical content, not the spoken style.
- Do not harshly penalize minor transcript noise.
- If the transcript appears cut off near the end, reduce the score only moderately unless a key mathematical idea is missing.
- Grade only what is explicitly present in the answer, but be fair to oral phrasing.

You must grade only based on the provided answer.

Evaluation process:
1. Read the question carefully.
2. Use the expected key points if they are provided.
3. Identify which key ideas appear explicitly in the student's answer.
4. Score primarily based on correctness of the reasoning.
5. Consider completeness of the reasoning steps.
6. Use clarity only as a minor adjustment.

Internal scoring priorities:
- Correctness: highest weight
- Completeness: medium weight
- Clarity: low weight

Score interpretation:
- 90-100: Strong oral answer, mathematically correct, includes essentially all key ideas
- 80-89: Good answer, correct main reasoning, maybe one minor missing formal detail
- 65-79: Some correct reasoning, but one important step or conclusion is missing
- 40-64: Partial understanding only
- 0-39: Incorrect or mostly unsupported

Label mapping:
- correct: 80-100
- partial: 40-79
- incorrect: 0-39

Return ONLY a valid JSON object in exactly this format:

{
  "score": 0,
  "feedback": "short feedback",
  "label": "correct"
}

Rules:
- score must be an integer between 0 and 100
- feedback must be short, specific, and concise
- feedback must be written in English only
- label must be exactly one of: correct, partial, incorrect
- do not return markdown
- do not return explanations outside the JSON
"""

_USER_PROMPT_TEMPLATE = """
Question:
{question}

Expected Key Points:
{expected_key_points}

Student Answer:
{answer}
"""


def build_default_system_prompt() -> str:
    return _SYSTEM_PROMPT


def build_default_user_prompt(
    question: str,
    answer: str,
    expected_key_points: str = "Not provided",
) -> str:
    return _USER_PROMPT_TEMPLATE.format(
        question=question,
        expected_key_points=expected_key_points,
        answer=answer,
    )


def build_default_messages(
    question: str,
    answer: str,
    expected_key_points: str = "Not provided",
) -> list[dict]:
    return [
        {"role": "system", "content": build_default_system_prompt()},
        {
            "role": "user",
            "content": build_default_user_prompt(
                question,
                answer,
                expected_key_points,
            ),
        },
        # Prefill the assistant turn with '{' so the model cannot output
        # any reasoning preamble before the JSON object.
        {"role": "assistant", "content": "{"},
    ]


# ── Rubric-based evaluation ───────────────────────────────────────────────────

_RUBRIC_SYSTEM_PROMPT = """
You are an examiner grading an oral exam answer against a detailed rubric.
The answer was transcribed from speech — ignore filler words and spoken hesitations.
Grade only what is explicitly present. Be fair to oral phrasing.

For each rubric criterion, assign a score between 0 and its maximum, then give brief feedback.
Return ONLY a valid JSON object in exactly this format:

{
  "criteria": [
    {"name": "CriterionName", "score": 0, "max_score": 10, "feedback": "brief reason"},
    ...
  ],
  "total_feedback": "overall one-line summary"
}

Rules:
- score must be a number between 0 and max_score
- feedback must be short and specific
- do not return markdown or text outside the JSON
"""

_RUBRIC_USER_TEMPLATE = """
Question:
{question}

Student Answer:
{answer}

Rubric Criteria:
{rubric_text}
"""


def build_rubric_messages(question: str, answer: str, rubric_criteria: list) -> list[dict]:
    rubric_lines = "\n".join(
        f"- {c['name']} (max {c['max_score']} pts): {c['description']}"
        for c in rubric_criteria
    )
    user_content = _RUBRIC_USER_TEMPLATE.format(
        question=question,
        answer=answer,
        rubric_text=rubric_lines,
    )
    return [
        {"role": "system", "content": _RUBRIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "{"},
    ]


# ── Anchor-based evaluation ───────────────────────────────────────────────────

_ANCHOR_SYSTEM_PROMPT = """
You are an expert examiner. Grade a student's oral answer by comparing it to
reference anchor answers at known quality levels.

The student answer was transcribed from speech — ignore filler words and hesitations.

Compare the student's answer to the provided anchors and assign a score from 0 to 100.

Return ONLY a valid JSON object in exactly this format:

{
  "score": 0,
  "closest_anchor": "level_name",
  "feedback": "brief justification referencing the anchors",
  "anchor_comparisons": {
    "anchor_level": "one sentence comparing student to this anchor"
  }
}

Rules:
- score must be an integer between 0 and 100
- closest_anchor must be one of the anchor level names provided
- feedback must be concise and reference specific content
- do not return markdown or text outside the JSON
"""

_ANCHOR_USER_TEMPLATE = """
Question:
{question}

Reference Anchors:
{anchors_text}

Student Answer:
{answer}
"""


def build_anchor_messages(question: str, answer: str, anchors: list) -> list[dict]:
    anchor_blocks = []
    for a in anchors:
        anchor_blocks.append(
            f"[{a['level'].upper()} — score ~{a['score']}]\n{a['answer']}"
        )
    anchors_text = "\n\n".join(anchor_blocks)
    user_content = _ANCHOR_USER_TEMPLATE.format(
        question=question,
        answer=answer,
        anchors_text=anchors_text,
    )
    return [
        {"role": "system", "content": _ANCHOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "{"},
    ]


# ── Pairwise comparison ───────────────────────────────────────────────────────

_PAIRWISE_SYSTEM_PROMPT = """
You are an expert examiner. You will be shown two student answers to the same question.
Both were transcribed from speech — ignore filler words and spoken hesitations.

Decide which answer demonstrates better understanding of the topic.

Return ONLY a valid JSON object in exactly this format:

{
  "winner": "A",
  "confidence": 0.85,
  "reasoning": "concise explanation referencing specific content differences"
}

Rules:
- winner must be exactly "A", "B", or "tie"
- confidence must be a number between 0.5 and 1.0 (0.5 = uncertain, 1.0 = very clear)
- reasoning must be concise and reference specific content
- do not return markdown or text outside the JSON
"""

_PAIRWISE_USER_TEMPLATE = """
Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Which answer is better?
"""


def build_pairwise_messages(question: str, answer_a: str, answer_b: str) -> list[dict]:
    user_content = _PAIRWISE_USER_TEMPLATE.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    return [
        {"role": "system", "content": _PAIRWISE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "{"},
    ]


# ── Group ranking (tournament group_ranking mode) ─────────────────────────────

_GROUP_RANKING_SYSTEM_PROMPT = """
You are an expert examiner. You will be given multiple student answers to the
same exam question. All answers were transcribed from speech — ignore filler
words and hesitations.

Rank ALL students from best to worst based on the quality of their answer.
Consider:
- Mathematical / factual accuracy (highest weight)
- Completeness of the explanation
- Clarity of reasoning (low weight)

Return ONLY a valid JSON object in exactly this format:

{
  "ranking": ["student_id_best", "student_id_second", ..., "student_id_worst"],
  "reasoning": "one or two sentences explaining the key differences that drove the ranking"
}

Rules:
- ranking must include EVERY student_id shown below, exactly once
- do not add extra student_ids that were not provided
- reasoning must reference specific content differences
- do not return markdown or text outside the JSON
"""

_GROUP_RANKING_USER_TEMPLATE = """
Question:
{question}

Student Answers:

{answers_block}

Rank these {n} students from best to worst.
"""


def build_group_ranking_messages(
    question: str,
    student_answers: list,  # [{"student_id": ..., "name": ..., "answer": ...}]
) -> list[dict]:
    blocks = []
    for s in student_answers:
        blocks.append(
            f"[{s['student_id']} — {s['name']}]\n{s['answer']}"
        )
    answers_block = "\n\n---\n\n".join(blocks)
    user_content = _GROUP_RANKING_USER_TEMPLATE.format(
        question=question,
        answers_block=answers_block,
        n=len(student_answers),
    )
    return [
        {"role": "system", "content": _GROUP_RANKING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "{"},
    ]