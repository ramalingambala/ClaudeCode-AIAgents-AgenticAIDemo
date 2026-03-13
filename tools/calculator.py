def calculate(expression: str) -> dict:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression string (e.g., "2 + 3 * 4").

    Returns:
        A dictionary with the expression and its result.
    """
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return {"expression": expression, "error": "Invalid characters in expression"}

    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}
