"""Minimal local Streamlit chat UI for the Deep Researcher graph."""

import asyncio
import logging
import time
import uuid

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from open_deep_research.configuration import Configuration
from open_deep_research.deep_researcher import deep_researcher_builder

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("open_deep_research.deep_researcher").setLevel(logging.INFO)


@st.cache_resource
def get_graph():
    return deep_researcher_builder.compile(checkpointer=InMemorySaver())


PHASES = [
    ("clarify_with_user", "Clarify"),
    ("write_research_brief", "Brief"),
    ("research_supervisor", "Research"),
    ("final_report_generation", "Report"),
]


def render(phase: int, topics: list[dict], placeholder) -> None:
    """Draw the phase line and the topic/search checklist."""
    marks = ["[>]" if i == phase else "[x]" if i < phase else "[ ]" for i in range(len(PHASES))]
    lines = ["  >  ".join(f"{m} {label}" for m, (_, label) in zip(marks, PHASES)), ""]
    for topic in topics:
        lines.append(f"- {'[x]' if topic['done'] else '[>]'} {topic['title']}")
        lines += [f"    - search: {q}" for q in topic["queries"]]
    placeholder.markdown("\n".join(lines))


async def tick(start: float, placeholder) -> None:
    while True:
        placeholder.markdown(f"Elapsed: {time.monotonic() - start:.0f}s")
        await asyncio.sleep(1)


async def run_turn(graph, text: str, config: dict, placeholder, timer) -> str:
    start = time.monotonic()
    ticker = asyncio.create_task(tick(start, timer))
    try:
        return await research(graph, text, config, placeholder)
    finally:
        ticker.cancel()
        timer.markdown(f"Total time: {time.monotonic() - start:.0f}s")


async def research(graph, text: str, config: dict, placeholder) -> str:
    max_units = Configuration.from_runnable_config().max_concurrent_research_units
    phase, topics = 0, []
    render(phase, topics, placeholder)
    async for ns, update in graph.astream(
        {"messages": [{"role": "user", "content": text}]},
        config,
        stream_mode="updates",
        subgraphs=True,
    ):
        for node, output in update.items():
            output = output or {}
            if node in dict(PHASES):
                phase = [n for n, _ in PHASES].index(node) + 1
            elif node == "supervisor":
                plan = output.get("research_plan", [])
                if plan and not topics:
                    topics = [
                        {
                            "title": item["requirement"],
                            "queries": item.get("queries", []),
                            "done": item["status"] in {"covered", "unavailable"},
                        }
                        for item in plan
                    ]
                messages = output.get("supervisor_messages", [])
                last_message = messages[-1] if messages else None
                calls = getattr(last_message, "tool_calls", [])
                titles = [c["args"]["research_topic"] for c in calls if c["name"] == "ConductResearch"]
                known_titles = {topic["title"] for topic in topics}
                topics += [
                    {"title": t, "queries": [], "done": False}
                    for t in titles[:max_units]
                    if t not in known_titles
                ]
            elif node == "researcher" and topics:
                for message in output.get("researcher_messages", []):
                    for call in getattr(message, "tool_calls", []):
                        topics[-1]["queries"] += call["args"].get("queries", [])
            elif node == "compress_research" and topics:
                topics[-1]["done"] = True
            render(phase, topics, placeholder)
    state = await graph.aget_state(config)
    return state.values["messages"][-1].content


st.title("Open Deep Research (local)")
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []

for role, content in st.session_state.history:
    st.chat_message(role).markdown(content)

if prompt := st.chat_input("Ask a research question"):
    st.chat_message("user").markdown(prompt)
    st.session_state.history.append(("user", prompt))
    with st.chat_message("assistant"):
        with st.status("Researching...", expanded=True) as status:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            timer = st.empty()
            answer = asyncio.run(run_turn(get_graph(), prompt, config, st.empty(), timer))
            status.update(label="Done", state="complete")
        st.markdown(answer)
    st.session_state.history.append(("assistant", answer))
