G_EVAL_COT_TEMPLATE = """
*** TASK:
Based on the following task description and evaluation criteria,
generate a detailed Chain of Thought (CoT) that outlines the necessary Evaluation Steps
to assess the solution. The CoT should clarify the reasoning process for each step of evaluation.

*** INPUT:

TASK INTRODUCTION:
{task_introduction}

EVALUATION CRITERIA:
{evaluation_criteria}

FINAL SCORE:
return the score in the range of starting from {min_score} to {max_score} inclusive.
SCORE VALUE MUST BE AN INTEGER.
"""


G_EVAL_QUERY_TEMPLATE = """
*** TASK INTRODUCTION:
{task_introduction}

*** EVALUATION CRITERIA:
{evaluation_criteria}

{chain_of_thought}

*** INPUT:
{input}

*** OUTPUT:
Return the output in a JSON format with the keys "score" and "reason".
"""


SUMMEVAL_TEMPLATE = """
{task_introduction}

{evaluation_criteria}

Source Text:

{document}

Summary:

{summary}

Evaluation Form (scores ONLY):

- Score ({min_score}-{max_score}):
"""


def format_summeval_prompt(
    task_introduction: str,
    evaluation_criteria: str,
    document: str = "",
    summary: str = "",
    min_score: int = 1,
    max_score: int = 5
) -> str:
    """
    Format a SummEval prompt using the template.
    
    Args:
        task_introduction: Task description
        evaluation_criteria: Evaluation criteria
        document: Source document (optional)
        summary: Summary to evaluate
        min_score: Minimum score in range
        max_score: Maximum score in range
        
    Returns:
        Formatted prompt string
    """
    return SUMMEVAL_TEMPLATE.format(
        task_introduction=task_introduction,
        evaluation_criteria=evaluation_criteria,
        document=document,
        summary=summary,
        min_score=min_score,
        max_score=max_score
    )