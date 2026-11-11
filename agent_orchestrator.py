"""agent_orchestrator.py - Tool-calling agent on local Qwen2.5 (Ollama).

The agent decides on its own which tool to call:
  * calculator          - safe arithmetic
  * get_order_status    - looks up a real SQLite database (orders.db)
  * search_company_docs - searches the Chroma index built by rag_chain.py

Setup: same as rag_chain.py (Ollama running, qwen2.5 + nomic-embed-text pulled).
Run  : python agent_orchestrator.py --docs ./docs
"""
from __future__ import annotations

import argparse
import ast
import operator
import sqlite3

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from rag_chain import LLM_MODEL, build_vectorstore, format_docs

DB_PATH = "orders.db"

SYSTEM_PROMPT = (
    "You are a customer support agent for Example Store. "
    "Use the tools to get facts instead of guessing: "
    "get_order_status for order questions, search_company_docs for policies "
    "(refunds, shipping, warranty, FAQs), and calculator for any arithmetic. "
    "For ANY question about the company or its products, ALWAYS call "
    "search_company_docs first, even if you think the answer is not there. "
    "Answer only from tool results. If the tools return nothing relevant, reply "
    "exactly: \"I don't have that information.\" Never suggest websites, "
    "sections or contacts that are not in the tool results. Keep answers short."
)


# --------------------------------------------------------------------------- #
# Database (the "database lookup" tool from the guide)
# --------------------------------------------------------------------------- #
def init_db() -> None:
    """Create orders.db with sample data on first run."""
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS orders ("
        "order_id TEXT PRIMARY KEY, customer TEXT, item TEXT, status TEXT, amount REAL)"
    )
    if con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0:
        con.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
            [
                ("A100", "Asha", "Wireless Headphones", "Shipped", 2499.0),
                ("A101", "Ravi", "Smart Watch", "Processing", 5999.0),
                ("A102", "Meera", "Phone Charger", "Delivered", 799.0),
            ],
        )
    con.commit()
    con.close()


# --------------------------------------------------------------------------- #
# Tools. The docstring is what the model reads to decide when to use each one.
# --------------------------------------------------------------------------- #
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression such as '2499 * 3 + 50'. Supports + - * / ** % and parentheses."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:  # report errors to the model instead of crashing
        return f"Could not calculate: {exc}"


@tool
def get_order_status(order_id: str) -> str:
    """Look up an order by its ID (for example A100). Returns the item, status and amount."""
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT item, status, amount FROM orders WHERE order_id = ?",  # parameterized
        (order_id.strip().upper(),),
    ).fetchone()
    con.close()
    if row is None:
        return f"No order found with ID {order_id}."
    return f"Order {order_id.upper()}: item={row[0]}, status={row[1]}, amount={row[2]} rupees."


def make_docs_tool(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    @tool
    def search_company_docs(query: str) -> str:
        """Search company documents for policies: refunds, shipping, warranty, FAQs, company info."""
        return format_docs(retriever.invoke(query)) or "No relevant documents found."

    return search_company_docs


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
def build_agent(docs_dir: str):
    init_db()
    tools = [calculator, get_order_status, make_docs_tool(build_vectorstore(docs_dir))]
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    return create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),  # per-thread conversation memory
    )


def tools_used(messages) -> list[str]:
    """Names of tools called during the latest turn (for visibility while learning)."""
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return [
        f"{call['name']}({call['args']})"
        for m in messages[last_human:]
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5 tool-calling agent")
    parser.add_argument("--docs", default="./docs", help="folder of .txt/.md files")
    args = parser.parse_args()

    agent = build_agent(args.docs)
    config = {"configurable": {"thread_id": "default"}}

    print("Ask anything (type 'exit' to quit). Try: 'Status of order A100?'")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        result = agent.invoke({"messages": [("user", question)]}, config=config)
        for call in tools_used(result["messages"]):
            print(f"  [tool] {call}")
        print("Agent:", result["messages"][-1].content)


if __name__ == "__main__":
    main()
