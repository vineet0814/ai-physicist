def math_solver(equation: str) -> str:
    import sympy as sp
    try:
        expr = sp.sympify(equation)
        sol = sp.solve(expr)
        return str(sol)
    except Exception as e:
        return f"Math error: {str(e)}"

def code_executor(code: str) -> str:
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return str(exec_globals)
    except Exception as e:
        return f"Code error: {str(e)}"

import arxiv

def retrieve_arxiv_documents(query: str, max_results: int = 3, sort_by: str = "relevance") -> List[str]:
    """
    Retrieve relevant documents from arXiv.

    Args:
        query (str): The input query for retrieval.
        max_results (int): Number of top documents to retrieve.
        sort_by (str): Sorting mode. (choices: relevance, lastUpdatedDate)

    Returns:
        List[str]: A list of document titles and abstracts.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion[sort_by]
    )
    results = []
    for result in client.results(search):
        results.append(f"Title: {result.title}\\nAbstract: {result.summary}")
    return results

TOOLS = [math_solver, code_executor, retrieve_arxiv_documents]