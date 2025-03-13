from typing import Dict, List, Any, Optional, TypedDict, Callable
import time
import os
from datetime import datetime

from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchRun
import traceback

# Configuration parameters
CONFIG = {
    "model": "qwen2.5:latest",
    "max_iterations": 20,
    "max_search_results_length": 4000,
    "progress_bar": True,
    "debug": False,
    "visualization_enabled": False,
    "recursion_limit": 50,  # Added recursion limit parameter
}

# Inicializar o LLM
llm = ChatOllama(model=CONFIG["model"])

# Iniciar a search tool
search_tool = DuckDuckGoSearchRun()


class AgentState(TypedDict):
    question: str
    sub_questions: List[str]
    research_findings: Dict[str, str]
    current_sub_question: Optional[str]
    final_answer: Optional[str]
    evaluation: Optional[str]
    messages: List[Any]
    iteration_count: int
    errors: List[str]
    start_time: float


def safe_llm_call(messages, max_retries=2):
    """Safely call the LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                error_msg = f"LLM call failed after {max_retries} attempts: {str(e)}"
                print(f"ERROR: {error_msg}")
                return AIMessage(content=f"Error retrieving information: {str(e)}")
            time.sleep(2)  # Wait before retrying


def breakdown_question(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content="You are a research expert skilled at breaking complex questions into specific sub-questions. Your job is to analyze the provided research question and break it down into 3-5 clear, focused sub-questions."),
        HumanMessage(
            content=f"Please break down the following research question into 3-5 sub-questions:\n\n{state['question']}\n\nFormat your response as a list of sub-questions, one per line, without numbering or bullets.")
    ]

    try:
        response = safe_llm_call(messages)
        sub_questions = [q.strip()
                         for q in response.content.split('\n') if q.strip()]

        if CONFIG["debug"]:
            print(f"Generated {len(sub_questions)} sub-questions")

        return {
            **state,
            "sub_questions": sub_questions,
            "research_findings": {},
            "messages": state["messages"] + [
                HumanMessage(
                    content=f"Breaking down the question: {state['question']}"),
                AIMessage(
                    content=f"I've broken down your question into these sub-questions:\n{response.content}")
            ]
        }
    except Exception as e:
        error_msg = f"Error in breakdown_question: {str(e)}"
        print(f"ERROR: {error_msg}")
        return {
            **state,
            "errors": state.get("errors", []) + [error_msg],
            # Fallback simple question
            "sub_questions": ["What is " + state["question"] + "?"],
        }


def research_with_search(state: AgentState) -> AgentState:
    """Research a sub-question using web search and update state."""
    # Update iteration count in state
    state = {**state, "iteration_count": state.get("iteration_count", 0) + 1}

    # Determine the next sub-question to research
    researched_questions = set(state["research_findings"].keys())
    remaining_questions = [
        q for q in state["sub_questions"] if q not in researched_questions]

    if remaining_questions:
        current_sub_question = remaining_questions[0]
    else:
        # No remaining questions, this shouldn't happen due to our routing
        # But handle it gracefully anyway
        print("\nWarning: No remaining questions but still in research phase")
        return state

    try:
        # Track time per question
        question_start_time = time.time()

        search_results = search_tool.invoke(current_sub_question)
        messages = [
            SystemMessage(
                content="You are a thorough research assistant. Your task is to provide a detailed answer to a specific research question based on the provided search results."),
            HumanMessage(
                content=f"Question: {current_sub_question}\n\nSearch results: {search_results[:CONFIG['max_search_results_length']]}\n\nPlease analyze these search results and provide a detailed, accurate answer to the question.")
        ]
        response = safe_llm_call(messages)

        research_findings = state["research_findings"].copy()
        research_findings[current_sub_question] = response.content

        # Calculate progress
        total_questions = len(state["sub_questions"])
        completed_questions = len(research_findings)
        progress_percentage = (completed_questions / total_questions) * 100

        # Calculate elapsed time and time per question
        elapsed = time.time() - question_start_time
        total_elapsed = time.time() - state["start_time"]

        # Enhanced progress reporting
        if CONFIG["progress_bar"]:
            bar_length = 20
            filled_length = int(
                bar_length * completed_questions // total_questions)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r[{bar}] {progress_percentage:.1f}% | Q: {completed_questions}/{total_questions} | Time: {elapsed:.1f}s/q | Total: {total_elapsed:.1f}s", end="")
        else:
            print(
                f"Research progress: {completed_questions}/{total_questions} questions ({progress_percentage:.1f}%) - {elapsed:.1f}s")

        return {
            **state,
            "current_sub_question": current_sub_question,
            "research_findings": research_findings,
            "messages": state["messages"] + [
                HumanMessage(
                    content=f"Researching with web search: {current_sub_question}"),
                AIMessage(
                    content=f"Research findings: {response.content[:100]}...")
            ]
        }
    except Exception as e:
        error_msg = f"Error in research for '{current_sub_question}': {str(e)}"
        print(f"\nERROR: {error_msg}")

        # Create a fallback response
        research_findings = state["research_findings"].copy()
        research_findings[current_sub_question] = f"Error during research: {str(e)}"

        return {
            **state,
            "current_sub_question": current_sub_question,
            "research_findings": research_findings,
            "errors": state.get("errors", []) + [error_msg]
        }


def select_next_step(state: AgentState) -> str:
    """Determine the next step in the research process.
    This function only returns the routing decision, not state updates."""

    # Safety limit check - use the existing iteration_count if it exists
    iteration_count = state.get("iteration_count", 0) + 1

    # Check if we've reached maximum iterations
    if iteration_count >= CONFIG["max_iterations"]:
        print(
            f"\nReached maximum iterations ({iteration_count}/{CONFIG['max_iterations']}), proceeding to synthesis...")
        return "synthesize"

    # Check which sub-questions still need research
    researched_questions = set(state["research_findings"].keys())
    remaining_questions = [
        q for q in state["sub_questions"] if q not in researched_questions]

    # Log progress if debug is enabled
    if CONFIG["debug"]:
        print(
            f"\nIteration {iteration_count}: {len(remaining_questions)} questions remaining")

    # Return the next step based on whether there are remaining questions
    if remaining_questions:
        return "research"
    else:
        print("\nAll questions researched, proceeding to synthesis...")
        return "synthesize"


def synthesize_findings(state: AgentState) -> AgentState:
    findings_text = "\n\n".join([
        f"Sub-question: {q}\nFindings: {findings}"
        for q, findings in state["research_findings"].items()
    ])

    print("\nSynthesizing research findings...")

    messages = [
        SystemMessage(content="You are a research synthesis expert. Your task is to combine separate research findings into a cohesive, well-structured answer to the original question."),
        HumanMessage(
            content=f"Original question: {state['question']}\n\nResearch findings:\n{findings_text}\n\nPlease synthesize these findings into a comprehensive answer to the original question.")
    ]

    response = safe_llm_call(messages)

    return {
        **state,
        "final_answer": response.content,
        "messages": state["messages"] + [
            HumanMessage(content="Synthesizing research findings..."),
            AIMessage(
                content=f"I've synthesized the findings into a comprehensive answer: {response.content[:100]}...")
        ]
    }


def evaluate_answer(state: AgentState) -> AgentState:
    """Evaluate the quality of the synthesized answer."""
    print("Evaluating answer quality...")

    messages = [
        SystemMessage(content="You are a critical evaluator of research answers. Your task is to identify any gaps, logical flaws, or areas where the answer could be improved."),
        HumanMessage(
            content=f"Original question: {state['question']}\n\nSynthesized answer: {state['final_answer']}\n\nPlease evaluate this answer, identifying any weaknesses or areas for improvement.")
    ]

    response = safe_llm_call(messages)

    return {
        **state,
        "evaluation": response.content,
        "messages": state["messages"] + [
            HumanMessage(content="Evaluating the answer quality..."),
            AIMessage(content=f"Evaluation: {response.content[:100]}...")
        ]
    }


def needs_refinement(state: AgentState) -> str:
    if not state["evaluation"]:
        return "complete"

    evaluation_lower = state["evaluation"].lower()
    positive_indicators = ["excellent", "adequate",
                           "good", "sufficient", "complete", "comprehensive"]

    if any(indicator in evaluation_lower for indicator in positive_indicators):
        return "complete"
    else:
        print("Answer needs refinement based on evaluation.")
        return "refine"


def refine_answer(state: AgentState) -> AgentState:
    print("Refining answer...")

    messages = [
        SystemMessage(
            content="You are an expert at refining research answers. Your task is to improve an answer based on specific evaluation feedback."),
        HumanMessage(
            content=f"Original question: {state['question']}\n\nCurrent answer: {state['final_answer']}\n\nEvaluation feedback: {state['evaluation']}\n\nPlease provide an improved version of the answer that addresses the weaknesses identified in the evaluation.")
    ]

    response = safe_llm_call(messages)

    return {
        **state,
        "final_answer": response.content,
        "messages": state["messages"] + [
            HumanMessage(content="Refining the answer based on evaluation..."),
            AIMessage(content=f"Refined answer: {response.content[:100]}...")
        ]
    }


def create_research_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("breakdown", breakdown_question)
    graph.add_node("research", research_with_search)
    graph.add_node("synthesize", synthesize_findings)
    graph.add_node("evaluate", evaluate_answer)
    graph.add_node("refine", refine_answer)

    # Add edges
    graph.add_edge("breakdown", "research")
    graph.add_conditional_edges(
        "research",
        select_next_step,  # Use the simplified select_next_step directly
        {
            "research": "research",
            "synthesize": "synthesize"
        }
    )
    graph.add_edge("synthesize", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        needs_refinement,
        {
            "refine": "refine",
            "complete": END
        }
    )
    graph.add_edge("refine", "evaluate")

    # Set entry point
    graph.set_entry_point("breakdown")

    # Compile the graph - removed the unsupported recursion_limit parameter
    return graph.compile()


# Create the research agent
research_agent = create_research_graph()

# Optional: Visualization function


def visualize_graph():
    if CONFIG["visualization_enabled"]:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.graph import StateGraph, START, END

            graph = StateGraph(AgentState)
            graph.add_node("breakdown", breakdown_question)
            graph.add_node("research", research_with_search)
            graph.add_node("synthesize", synthesize_findings)
            graph.add_node("evaluate", evaluate_answer)
            graph.add_node("refine", refine_answer)
            graph.add_edge("breakdown", "research")
            graph.add_edge("synthesize", "evaluate")
            graph.add_edge("refine", "evaluate")
            graph.set_entry_point("breakdown")

            # Export the graph visualization
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"research_graph_{timestamp}"
            graph.save_graph(f"{filename}.png")
            print(f"\nGraph visualization saved to {filename}.png")
        except Exception as e:
            print(f"\nVisualization failed: {str(e)}")


# UI with improved error handling and reporting
def run_research_agent(question: str, config_updates=None):
    # Update configuration if provided
    if config_updates and isinstance(config_updates, dict):
        CONFIG.update(config_updates)

    # Optional: Generate visualization
    if CONFIG["visualization_enabled"]:
        visualize_graph()

    # Prepare initial state
    initial_state = {
        "question": question,
        "sub_questions": [],
        "research_findings": {},
        "current_sub_question": None,
        "final_answer": None,
        "evaluation": None,
        "messages": [],
        "iteration_count": 0,
        "errors": [],
        "start_time": time.time()
    }

    print(f"\n{'='*50}")
    print(f"RESEARCH QUESTION: {question}")
    print(f"{'='*50}\n")
    print(
        f"Starting research at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using model: {CONFIG['model']}\n")

    try:
        final_state = research_agent.invoke(initial_state)

        # Calculate total research time
        total_time = time.time() - final_state["start_time"]

        print(f"\n{'='*50}")
        print(f"RESEARCH COMPLETED in {total_time:.1f} seconds")
        print(f"{'='*50}")

        print("\n=== Sub-Questions ===")
        for i, q in enumerate(final_state["sub_questions"], 1):
            print(f"{i}. {q}")

        print("\n=== Final Answer ===")
        print(final_state["final_answer"])

        print("\n=== Evaluation ===")
        print(final_state["evaluation"])

        if final_state.get("errors", []):
            print("\n=== Errors Encountered ===")
            for error in final_state["errors"]:
                print(f"- {error}")

        return final_state

    except Exception as e:
        print(f"\n{'!'*50}")
        print(f"ERROR: Research process failed: {str(e)}")
        print(traceback.format_exc())
        print(f"{'!'*50}\n")
        return None


if __name__ == "__main__":
    # Example configuration override
    custom_config = {
        "debug": True,
        "max_iterations": 10,  # Use a lower value to prevent any recursion issues
        "visualization_enabled": False,  # Set to True if needed
        "recursion_limit": 50  # Keep this for documentation but it won't be used in compile()
    }

    # Add additional safety to prevent too many iterations
    import sys
    # Check current recursion limit and set a reasonable one
    current_limit = sys.getrecursionlimit()
    print(f"Current Python recursion limit: {current_limit}")
    # Only increase if needed, don't decrease
    if current_limit < 2000:
        new_limit = 2000  # Set a reasonable limit
        sys.setrecursionlimit(new_limit)
        print(f"Increased Python recursion limit to {new_limit}")

    question = "Porque a cidade de Curitiba tem esse nome?"
    run_research_agent(question, custom_config)
