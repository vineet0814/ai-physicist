import json

def heuristic_reward(messages: list) -> float:
    from collections import Counter
    score = 0.0
    last_response = messages[-1]['content'].lower()

    if "therefore" in last_response or "thus" in last_response:
        score += 1.0
    if any(k in last_response for k in ["solve", "equation", "integrate"]):
        score += 1.0

    tools_used = [msg['name'] for msg in messages if msg['role'] == 'tool']
    tool_counts = Counter(tools_used)
    score += 1.0 if tools_used else 0.0

    redundant_penalty = sum(count - 1 for count in tool_counts.values() if count > 1)
    score -= redundant_penalty * 0.2

    return max(score, 0.0)

def evaluator_llm(llm, hypothesis: str, eval_strategy: str, tool_outputs: list) -> dict:
    """
    Use an Evaluator LLM to assess the quality of the evaluation strategy.

    Args:
        hypothesis (str): The hypothesis being evaluated.
        eval_strategy (str): The evaluation strategy proposed by the model.
        tool_outputs (list): A list of outputs from tools used during evaluation.

    Returns:
        dict: A dictionary containing the evaluation score and feedback.
    """
    # Step 1: Prepare the input prompt for the Evaluator LLM
    prompt = f"""
    Hypothesis: {hypothesis}

    Evaluation Strategy:
    {eval_strategy}

    Tool Outputs:
    {', '.join(tool_outputs)}

    Instructions:
    - Assess the logical coherence of the evaluation strategy.
    - Check if the evaluation strategy aligns with the hypothesis.
    - Evaluate the completeness of the strategy (e.g., use of tools, evidence quality).
    - Provide a score between 0 and 1 based on the overall quality.
    - Provide feedback on how the strategy can be improved.
    """

    # Step 2: Send the prompt to the Evaluator LLM
    # (Pseudo-code for LLM interaction)
    response = llm.generate(prompt, max_tokens=200)

    # Step 3: Parse the response
    try:
        # Assume the LLM returns a JSON-like response
        evaluation = json.loads(response)
        score = evaluation.get("score", 0.0)
        feedback = evaluation.get("feedback", "No feedback provided.")
    except Exception:
        # Handle parsing errors
        score = 0.0
        feedback = "Error parsing LLM response."

    # Step 4: Return the evaluation results
    return {
        "score": score,
        "feedback": feedback
    }