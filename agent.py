"""
Web Research Agent using LangGraph + Tavily Search.

Searches the web for a given topic, synthesizes findings, and returns
a structured research report.

Usage:
    python agent.py
    python agent.py --query "latest advances in quantum computing"
"""

import argparse
import os
import time
import difflib
from typing import Annotated, TypedDict, Optional, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# Enable a full mock mode for offline/testing without API calls.
# Set environment variable `USE_MOCK=1` or `USE_MOCK=true` to enable.
USE_MOCK = os.getenv("USE_MOCK", "false").lower() in ("1", "true", "yes")


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    search_results: list[dict]
    report: str


def search_web(state: ResearchState) -> ResearchState:
    if USE_MOCK:
        # Return mocked search results for offline testing
        return {
            "search_results": [
                {
                    "url": "https://example.com/mock-1",
                    "title": "Mock result 1",
                    "content": "This is mocked content for testing the agent."
                },
                {
                    "url": "https://example.com/mock-2",
                    "title": "Mock result 2",
                    "content": "Additional mocked content to exercise the synthesizer."
                },
            ]
        }

    tool = TavilySearch(max_results=5)
    raw_results = tool.invoke(state["query"])
    if isinstance(raw_results, dict):
        results = raw_results.get("results", [])
    elif isinstance(raw_results, list):
        results = raw_results
    else:
        results = []
    return {"search_results": results}


def synthesize_report(state: ResearchState) -> ResearchState:
    temperature = float(state.get("temperature", 0))
    previous_report = state.get("previous_report")

    if USE_MOCK:
        # Return a deterministic mocked report for offline testing
        sources = "\n".join(f"- {r.get('url', 'N/A')}" for r in state.get("search_results", []))
        report = (
            f"MOCK RESEARCH REPORT for: {state.get('query', '')}\n\n"
            "Summary:\nThis is a mocked summary generated for testing purposes.\n\n"
            "Key Findings:\n- Mock finding A\n- Mock finding B\n\n"
            "Sources:\n" + sources
        )
        return {"report": report, "messages": []}

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temperature)

    # Number sources for easier citation and provenance
    sources: List[str] = []
    numbered_results = []
    for i, r in enumerate(state.get("searat ch_results", []), start=1):
        url = r.get("url", "N/A")
        title = r.get("title", "N/A")
        content = r.get("content", "")[:1000]
        numbered_results.append(f"[{i}] Source: {url}\nTitle: {title}\nContent: {content}")
        sources.append(f"[{i}] {url} - {title}")

    results_text = "\n\n".join(numbered_results)

    # If there's a previous report, ask the model to improve it using the new sources
    if previous_report:
        human_prompt = (
            f"Research query: {state['query']}\n\n"
            f"Previous report:\n{previous_report}\n\n"
            f"Use the following search results to improve the report, correct any errors, add missing key findings, and include a Sources section with numbered citations.\n\n"
            f"Search results:\n{results_text}"
        )
    else:
        human_prompt = (
            f"Research query: {state['query']}\n\n"
            f"Synthesize a clear, structured report (Summary, Key Findings, and Sources) from these search results. Include a Sources section with numbered citations.\n\n"
            f"Search results:\n{results_text}"
        )

    messages = [
        SystemMessage(content="You are a research analyst. Provide a concise, well-structured research report. Cite sources by number."),
        HumanMessage(content=human_prompt),
    ]

    response = llm.invoke(messages)

    # Append a clean Sources section if the model didn't include one
    report = response.content
    if "Sources:" not in report:
        report = report.strip() + "\n\nSources:\n" + "\n".join(sources)

    return {"report": report, "messages": [response]}


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)
    graph.add_node("search", search_web)
    graph.add_node("synthesize", synthesize_report)
    graph.set_entry_point("search")
    graph.add_edge("search", "synthesize")
    graph.set_finish_point("synthesize")
    return graph.compile()


def main():
    parser = argparse.ArgumentParser(description="Web Research Agent")
    parser.add_argument("--query", default="latest advances in AI agents 2024", help="Research query")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum refinement iterations")
    parser.add_argument("--temperature", type=float, default=0, help="LLM temperature")
    args = parser.parse_args()

    print(f"\n🔍 Researching: {args.query}\n")

    # Iterative refinement loop (Retrieve -> Generate -> Evaluate)
    query = args.query
    max_iters = args.max_iterations
    temperature = args.temperature

    graph = build_graph()
    state = {
        "query": query,
        "messages": [],
        "previous_report": None,
        "temperature": temperature,
    }

    prev_report = ""
    final_report = ""

    for iteration in range(1, max_iters + 1):
        print(f"\n--- Iteration {iteration} ---")
        if prev_report:
            state["previous_report"] = prev_report

        synth_out = graph.invoke(state)
        report = synth_out.get("report", "")
        state.update(synth_out)

        # Simple evaluator: check change vs previous and count cited sources
        def evaluate(prev: str, curr: str) -> float:
            if not prev:
                return 0.0
            seq = difflib.SequenceMatcher(a=prev, b=curr)
            change_score = 1.0 - seq.ratio()  # larger when report changed
            # reward having multiple citations
            citations = curr.count("[")
            citation_score = min(1.0, citations / 5.0)
            # weighted combination
            return 100.0 * (0.6 * change_score + 0.4 * citation_score)

        score = evaluate(prev_report, report)
        print(f"Evaluation score (0-100): {score:.1f}")

        prev_report = report
        final_report = report
        state["report"] = report

        # Stopping criteria: no meaningful change
        if score < 1.0 and iteration > 1:
            print("No meaningful improvements; stopping early.")
            break

        # small delay to avoid hitting rate limits in quick loops
        time.sleep(0.5)

    print("=" * 60)
    print("📄 FINAL RESEARCH REPORT")
    print("=" * 60)
    print(final_report)


if __name__ == "__main__":
    main()
