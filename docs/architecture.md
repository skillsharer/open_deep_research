# Architecture

All paths are relative to the repo root.

```mermaid
flowchart TB
    subgraph Launcher["run.sh"]
        RUN["run.sh: starts SearXNG (Docker), Ollama, exports env, launches Streamlit; trap cleanup stops containers"]
    end

    subgraph UI["app.py (Streamlit)"]
        GG["get_graph() - compiles graph with in-memory checkpointer"]
        RT["run_turn() - streams graph updates"]
        RS["research() - phase line + topic checklist"]
        RD["render() / tick() - progress and timer"]
        GG --> RT --> RS --> RD
    end

    subgraph Graph["src/open_deep_research/deep_researcher.py - deep_researcher_builder"]
        C["clarify_with_user()"]
        B["write_research_brief()"]
        subgraph Sup["supervisor_subgraph (supervisor_builder)"]
            S["supervisor()"]
            ST["supervisor_tools()"]
            S --> ST --> S
        end
        subgraph Res["researcher_subgraph (researcher_builder), one per topic"]
            R["researcher()"]
            RTL["researcher_tools()<br/>execute_tool_safely()"]
            CR["compress_research()"]
            R --> RTL --> R
            RTL --> CR
        end
        F["final_report_generation()"]
        C --> B --> S
        ST -- "ConductResearch (parallel, limited by max_concurrent_research_units)" --> R
        CR -- "notes" --> ST
        ST -- "ResearchComplete" --> F
    end

    subgraph Utils["src/open_deep_research/utils.py"]
        GAT["get_all_tools()"]
        GST["get_search_tool()"]
        SX["searxng_search() - SEARXNG_LANGUAGES, SEARXNG_MAX_RESULTS, source IDs"]
        FH["fetch_html() - archive fallback on 401/402/403/429"]
        TT["think_tool()"]
        MCP["load_mcp_tools() / get_tokens() / set_tokens() (optional MCP + Supabase)"]
        FC["finalize_citations() - [S id] to [n], Sources built from retrieval metadata"]
        GN["get_notes_from_tool_calls()"]
        TL["is_token_limit_exceeded() / get_model_token_limit()"]
    end

    subgraph Cfg["Configuration and prompts"]
        CF["configuration.py: Configuration, SearchAPI, MCPConfig"]
        ST2["state.py: AgentState, SupervisorState, ResearcherState, ConductResearch, ResearchComplete"]
        PR["prompts.py: clarify / brief / lead_researcher / research_system / compress_research / final_report_generation prompts"]
    end

    subgraph Ext["Local services"]
        SXNG[("SearXNG container :8080")]
        OLL[("Ollama :11434 (gemma4)")]
        WEB[("Target web pages")]
        IA[("Internet Archive (Wayback)")]
    end

    RUN --> UI
    RUN --> SXNG
    RUN --> OLL
    RT --> C
    R --> GAT --> GST --> SX
    GAT --> TT
    GAT --> MCP
    SX -- "queries (sequential, 2s apart)" --> SXNG
    SX --> FH
    FH -- "fetch + trafilatura extract" --> WEB
    FH -- "if blocked" --> IA
    C & B & S & R & CR & F -- "init_chat_model" --> OLL
    ST --> GN
    F --> FC
    F --> TL
    Graph --- Cfg
```

## Flow

1. [run.sh](../run.sh) starts SearXNG and Ollama, sets env vars, and runs [app.py](../app.py).
2. [`run_turn`](../app.py) streams the graph; [`research`](../app.py) maps node names to the phases Clarify, Brief, Research, Report.
3. [`clarify_with_user`](../src/open_deep_research/deep_researcher.py) may ask a question, otherwise [`write_research_brief`](../src/open_deep_research/deep_researcher.py) creates the brief.
4. [`supervisor`](../src/open_deep_research/deep_researcher.py) plans topics; [`supervisor_tools`](../src/open_deep_research/deep_researcher.py) runs a researcher subgraph per `ConductResearch` call.
5. [`researcher`](../src/open_deep_research/deep_researcher.py) calls [`searxng_search`](../src/open_deep_research/utils.py) and `think_tool`; [`compress_research`](../src/open_deep_research/deep_researcher.py) returns a note that cites sources by ID (`[S3f9a1c]`).
6. [`final_report_generation`](../src/open_deep_research/deep_researcher.py) writes the report with source IDs, then [`finalize_citations`](../src/open_deep_research/utils.py) numbers them by first appearance and builds the Sources list from the retrieved metadata (unknown IDs are dropped).

## Tests

- [tests/test_live_search.py](../tests/test_live_search.py): SearXNG and Trafilatura connectivity.
- [tests/test_live_research.py](../tests/test_live_research.py): end-to-end research quality over several questions in `CASES` (facts retrieved and reported, grounded citations, claims supported by cited sources). Run with `uv run pytest -m live`.
