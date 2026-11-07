# 🔬 Open Deep Research

<img width="1388" height="298" alt="full_diagram" src="https://github.com/user-attachments/assets/12a2371b-8be2-4219-9b48-90503eb43c69" />

Deep research has broken out as one of the most popular agent applications. This is a simple, configurable, fully open source deep research agent that works across many model providers, search tools, and MCP servers. It's performance is on par with many popular deep research agents ([see Deep Research Bench leaderboard](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)).

<img width="817" height="666" alt="Screenshot 2025-07-13 at 11 21 12 PM" src="https://github.com/user-attachments/assets/052f2ed3-c664-4a4f-8ec2-074349dcaa3f" />

### 🔥 Recent Updates

**August 14, 2025**: See our free course [here](https://academy.langchain.com/courses/deep-research-with-langgraph) (and course repo [here](https://github.com/langchain-ai/deep_research_from_scratch)) on building open deep research.

**August 7, 2025**: Added GPT-5 and updated the Deep Research Bench evaluation w/ GPT-5 results.

**August 2, 2025**: Achieved #6 ranking on the [Deep Research Bench Leaderboard](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard) with an overall score of 0.4344. 

**July 30, 2025**: Read about the evolution from our original implementations to the current version in our [blog post](https://rlancemartin.github.io/2025/07/30/bitter_lesson/).

**July 16, 2025**: Read more in our [blog](https://blog.langchain.com/open-deep-research/) and watch our [video](https://www.youtube.com/watch?v=agGiWUpxkhg) for a quick overview.

### 🚀 Quickstart

1. Clone the repository and activate a virtual environment:
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
uv sync
# or
uv pip install -r pyproject.toml
```

3. Set up your `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):
```bash
cp .env.example .env
```

4. Launch the local Streamlit UI (no LangSmith/Studio involved):

```bash
uv run streamlit run app.py
```

Type your question in the chat box. Clarifying questions from the agent appear in the chat; answer them in the same thread.

To start the free local path with SearXNG and Ollama, install Docker, Ollama, and `uv`, then run:

```bash
chmod +x run.sh
./run.sh
```

The script loads `.env`, starts Ollama if needed, runs SearXNG in Docker, downloads the model specified by `OLLAMA_MODEL` if it is missing, and launches the Streamlit UI at http://localhost:8501, configured to use both.

### ⚙️ Configurations

#### LLM :brain:

Open Deep Research runs local models through Ollama via the [init_chat_model() API](https://python.langchain.com/docs/how_to/chat_models_universal_init/). It uses LLMs for a few different tasks. See the below model fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) file for more details. This can be accessed via the LangGraph Studio UI. 

- **Summarization** (default: `ollama:gemma4:latest`): Summarizes search API results
- **Research** (default: `ollama:gemma4:latest`): Power the search agent
- **Compression** (default: `ollama:gemma4:latest`): Compresses research findings
- **Final Report Model** (default: `ollama:gemma4:latest`): Write the final report

> Note: the selected model will need to support [structured outputs](https://python.langchain.com/docs/integrations/chat/) and [tool calling](https://python.langchain.com/docs/how_to/tool_calling/).

> Note: For OpenRouter, follow [this guide](https://github.com/langchain-ai/open_deep_research/issues/75#issuecomment-2811472408).

#### Using Ollama locally

Open Deep Research can use models served by a local [Ollama](https://ollama.com/) instance. The Ollama server must be running on the same machine as the LangGraph app.

1. [Install Ollama](https://ollama.com/download), then start the Ollama app (or start the Ollama service for your operating system).
2. Pull a model that supports both tool calling and structured output. For example:

   ```bash
   ollama pull gpt-oss:20b
   ollama list
   ```

   Choose a model appropriate for your hardware; see Ollama's [model library](https://ollama.com/search?c=tools) for models with tool-calling support.
3. Install or update the project dependencies so the [LangChain Ollama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama) is available:

   ```bash
   uv sync
   ```

4. Start the app with `./run.sh`. It sets the Ollama models through environment variables (`OLLAMA_MODEL` in `.env`); these override the defaults in [configuration.py](./src/open_deep_research/configuration.py).
   Change the model suffix if you pulled a different model.

Ollama's default local endpoint is `http://localhost:11434`, so no Ollama API key is needed for local use. All four model fields default to Ollama and `search_api` defaults to SearXNG, so no external model or search API is used. To disable LangSmith tracing and its outbound requests too, set `LANGSMITH_TRACING=false` in `.env`.

#### Search API :mag:

Open Deep Research searches the web through a self-hosted SearXNG instance and has full MCP compatibility. See the `search_api` and `mcp_config` fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) file for more details. This can be accessed via the LangGraph Studio UI. 

Configure [SearXNG](https://docs.searxng.org/admin/installation.html) and expose its HTTP endpoint. `./run.sh` starts SearXNG with the tracked [settings.yml](./searxng/settings.yml), which inherits SearXNG's default engine catalog, enabled engines, and complete locale list. The configuration preserves SearXNG's API-key and CAPTCHA policy while forwarding selected locale headers to compatible engines. Set `SEARXNG_URL` in `.env` to the instance base URL (for example, `http://localhost:8080`) (`./run.sh` does this for you). SearXNG's JSON format is enabled in the repository configuration. Search results are opened and their article text is extracted with [Trafilatura](https://trafilatura.readthedocs.io/en/latest/); relevant passages are selected with the local Ollama embedding model configured by `EMBEDDING_MODEL`. Fetch or extraction failures are reported rather than replaced with snippets. Pages are fetched with an identifying `OpenDeepResearchLocal` User-Agent; if a site answers 401/402/403/429, the Internet Archive snapshot of that page is used instead and marked as an archived copy in the source text. Optional settings: `SEARXNG_MAX_RESULTS` (pages per query, default 10), `SEARXNG_MAX_PASSAGES` (relevant passages per page, default 3), `SEARXNG_LANGUAGES` (comma-separated language codes, default `all`, each query runs once per language) and `RESEARCH_RUN_DIR` (failed-run artifacts, default `.research-runs`). `all` keeps locale selection broad; add explicit codes such as `all,en,hu,de` when targeted locale variants are needed. No search API key is required.

The launcher pulls `nomic-embed-text` for local passage selection. Failed searches retain fetched source artifacts under `.research-runs/<run-id>/`; successful searches clean up their temporary run directory.

Run the optional live checks separately with `uv run pytest -m live tests/test_live_search.py -k searxng` and `uv run pytest -m live tests/test_live_search.py -k trafilatura`. The SearXNG test requires the configured instance to be running; the Trafilatura test fetches a public webpage.

The end-to-end research-quality test (`uv run pytest -m live tests/test_live_research.py`, about 3 minutes per question, 4 questions defined in `CASES`; add your own question with the facts a good report must contain) runs a full research per question with SearXNG and Ollama running (start them with `./run.sh`) and checks that the report contains the expected facts, that citations are sequential and resolve to a source, and that every cited URL was actually retrieved.

For a fixed benchmark run, generate reports from the first three existing DeepResearch Bench prompts and score them against the reference report dimensions:

```bash
uv run python tests/run_benchmark.py --limit 3
uv run python tests/evaluate_benchmark.py
```

The scorer reports dimension coverage, citation structure and report length. The inputs and rubric dimensions are fixed; the web and model outputs remain variable by design. Benchmark records also retain the research plan, failed-dimension errors, raw-note count, and retrieved source count so retrieval failures can be separated from synthesis failures.
To start Ollama and SearXNG, run the same benchmark, and stop the services automatically when it finishes, use:

```bash
./run.sh benchmark
```

Research progress is logged to the terminal, including planning, dimension attempts,
compression, and report generation. The benchmark also logs the current case ID.

Set `BENCHMARK_LIMIT` to change the number of fixed cases.

#### Other 

See the fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) for various other settings to customize the behavior of Open Deep Research. 

#### Results 

| Name | Commit | Summarization | Research | Compression | Total Cost | Total Tokens | RACE Score | Experiment |
|------|--------|---------------|----------|-------------|------------|--------------|------------|------------|
| GPT-5 | [ca3951d](https://github.com/langchain-ai/open_deep_research/pull/168/commits) | openai:gpt-4.1-mini | openai:gpt-5 | openai:gpt-4.1 |  | 204,640,896 | 0.4943 | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-613c-4bda-8bde-f64f0422bbf3/compare?selectedSessions=4d5941c8-69ce-4f3d-8b3e-e3c99dfbd4cc&baseline=undefined) |
| Defaults | [6532a41](https://github.com/langchain-ai/open_deep_research/commit/6532a4176a93cc9bb2102b3d825dcefa560c85d9) | openai:gpt-4.1-mini | openai:gpt-4.1 | openai:gpt-4.1 | $45.98 | 58,015,332 | 0.4309 | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=cf4355d7-6347-47e2-a774-484f290e79bc&baseline=undefined) |
| Claude Sonnet 4 | [f877ea9](https://github.com/langchain-ai/open_deep_research/pull/163/commits/f877ea93641680879c420ea991e998b47aab9bcc) | openai:gpt-4.1-mini | anthropic:claude-sonnet-4-20250514 | openai:gpt-4.1 | $187.09 | 138,917,050 | 0.4401 | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=04f6002d-6080-4759-bcf5-9a52e57449ea&baseline=undefined) |
| Deep Research Bench Submission | [c0a160b](https://github.com/langchain-ai/open_deep_research/commit/c0a160b57a9b5ecd4b8217c3811a14d8eff97f72) | openai:gpt-4.1-nano | openai:gpt-4.1 | openai:gpt-4.1 | $87.83 | 207,005,549 | 0.4344 | [Link](https://smith.langchain.com/o/ebbaf2eb-769b-4505-aca2-d11de10372a4/datasets/6e4766ca-6[…]ons=e6647f74-ad2f-4cb9-887e-acb38b5f73c0&baseline=undefined) |

### 🚀 Deployments and Usage

#### Hosted deployment
 
You can easily deploy to [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 

#### Open Agent Platform

Open Agent Platform (OAP) is a UI from which non-technical users can build and configure their own agents. OAP is great for allowing users to configure the Deep Researcher with different MCP tools and search APIs that are best suited to their needs and the problems that they want to solve.

We've deployed Open Deep Research to our public demo instance of OAP. All you need to do is add your API Keys, and you can test out the Deep Researcher for yourself! Try it out [here](https://oap.langchain.com)

You can also deploy your own instance of OAP, and make your own custom agents (like Deep Researcher) available on it to your users.
1. [Deploy Open Agent Platform](https://docs.oap.langchain.com/quickstart)
2. [Add Deep Researcher to OAP](https://docs.oap.langchain.com/setup/agents)
