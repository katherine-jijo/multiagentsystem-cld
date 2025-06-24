from langgraph.graph import StateGraph
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import TypedDict, List, Tuple, Optional
from dotenv import load_dotenv
import os

load_dotenv()

# 💾 Shared state
class AgentState(TypedDict):
    input_text: str
    task: Optional[str]         # "extract", "summarize", "reject"
    relationships: List[Tuple[str, str, str]]
    diagram: str
    summary: str
    message: str                # Final output message

# Model used 
llm = ChatOpenAI(temperature=0, model="azure.gpt-4o")

# 🤖 Orchestrator Agent
def orchestrator_agent(state: AgentState) -> AgentState:
    prompt = PromptTemplate.from_template("""
Decide what to do with the following user input. Respond with only one word:
"extract" - if it contains clear cause-effect relationships,
"summarize" - if it's insightful but not causal,
"reject" - if it is not useful for analysis.

Text:
{input_text}
""")
    response = (prompt | llm).invoke({"input_text": state["input_text"]})
    task = response.strip().lower()
    return {**state, "task": task}

# 🔍 Extractor Agent
def extractor_agent(state: AgentState) -> AgentState:
    prompt = PromptTemplate.from_template("""
Extract causal relationships from this text in the format:
[("Cause", "Effect", "positive" or "negative")]

Text:
{input_text}
""")
    response = (prompt | llm).invoke({"input_text": state["input_text"]})
    try:
        extracted = eval(response.strip())
    except:
        extracted = []
    return {**state, "relationships": extracted}

# 📊 Formatter Agent
def formatter_agent(state: AgentState) -> AgentState:
    lines = ["graph TD"]
    for cause, effect, polarity in state["relationships"]:
        arrow = "+" if polarity.lower() == "positive" else "−"
        lines.append(f'{cause} -->|{arrow}| {effect}')
    return {**state, "diagram": "\n".join(lines)}

# 🧾 Summarizer Agent
def summarizer_agent(state: AgentState) -> AgentState:
    prompt = PromptTemplate.from_template("""
Summarize the main ideas from this text in 2–3 sentences:

Text:
{input_text}
""")
    summary = (prompt | llm).invoke({"input_text": state["input_text"]})
    return {**state, "summary": summary.strip()}

# ❌ Reject Agent
def reject_agent(state: AgentState) -> AgentState:
    return {**state, "message": "Sorry, this input doesn't contain anything we can work with."}

# 🧠 Decide what next based on the task
def router(state: AgentState) -> str:
    task = state.get("task", "reject")
    if task == "extract":
        return "extractor"
    elif task == "summarize":
        return "summarizer"
    else:
        return "reject"

# 🧱 Build LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", orchestrator_agent)
workflow.add_node("extractor", extractor_agent)
workflow.add_node("formatter", formatter_agent)
workflow.add_node("summarizer", summarizer_agent)
workflow.add_node("reject", reject_agent)

workflow.set_entry_point("orchestrator")
workflow.add_conditional_edges("orchestrator", router)
workflow.add_edge("extractor", "formatter")

graph = workflow.compile()

# 🧪 Run
def run_orchestrated_pipeline(text: str) -> AgentState:
    initial_state = {
        "input_text": text,
        "task": None,
        "relationships": [],
        "diagram": "",
        "summary": "",
        "message": ""
    }
    return graph.invoke(initial_state)
