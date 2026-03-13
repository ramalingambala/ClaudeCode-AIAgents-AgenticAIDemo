KNOWLEDGE_BASE = [
    {
        "id": 1,
        "title": "Getting Started with Python",
        "content": "Python is a versatile programming language used for web development, data science, automation, and more.",
        "tags": ["python", "programming", "beginner"],
    },
    {
        "id": 2,
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
        "tags": ["ml", "ai", "data science"],
    },
    {
        "id": 3,
        "title": "REST API Best Practices",
        "content": "RESTful APIs should use proper HTTP methods, status codes, and follow resource-based URL naming conventions.",
        "tags": ["api", "rest", "web development"],
    },
    {
        "id": 4,
        "title": "Claude Agent SDK Overview",
        "content": "The Claude Agent SDK allows developers to build agentic applications where Claude can use tools, reason through problems, and complete multi-step tasks.",
        "tags": ["claude", "agent", "sdk", "anthropic"],
    },
    {
        "id": 5,
        "title": "Docker Fundamentals",
        "content": "Docker containers package applications with their dependencies, ensuring consistent behavior across different environments.",
        "tags": ["docker", "devops", "containers"],
    },
]


def search_knowledge_base(query: str) -> dict:
    """Search a knowledge base for articles matching the query.

    Args:
        query: The search query string.

    Returns:
        A dictionary with matching results.
    """
    query_lower = query.lower()
    results = []

    for article in KNOWLEDGE_BASE:
        title_match = query_lower in article["title"].lower()
        content_match = query_lower in article["content"].lower()
        tag_match = any(query_lower in tag for tag in article["tags"])

        if title_match or content_match or tag_match:
            results.append(article)

    return {
        "query": query,
        "total_results": len(results),
        "results": results,
    }
