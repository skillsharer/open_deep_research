"""Utility functions and helpers for the Deep Research agent."""

import asyncio
import hashlib
import logging
import os
import re
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
from langchain_core.messages import (
    AIMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from trafilatura import extract, extract_metadata

from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.state import ResearchComplete

@tool(description="Search the web using a SearXNG instance.")
async def searxng_search(
    queries: List[str],
) -> str:
    """Search SearXNG, fetch matching pages, and return source-linked content."""
    base_url = os.getenv("SEARXNG_URL", "").rstrip("/")
    if not base_url:
        raise ValueError("SEARXNG_URL must be set when search_api is 'searxng'.")

    max_results = int(os.getenv("SEARXNG_MAX_RESULTS", "5"))
    languages = [lang.strip() for lang in os.getenv("SEARXNG_LANGUAGES", "all").split(",") if lang.strip()]

    timeout = aiohttp.ClientTimeout(total=30)
    headers = {
        "User-Agent": "OpenDeepResearchLocal/1.0 (+https://github.com/langchain-ai/open_deep_research)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async def search(query: str, language: str) -> dict[str, Any]:
            params = urlencode({"q": query, "format": "json", "language": language})
            async with session.get(f"{base_url}/search?{params}") as response:
                response.raise_for_status()
                return await response.json()

        searches = [(query, language) for query in queries for language in languages]
        results = []
        for index, (query, language) in enumerate(searches):
            if index:
                await asyncio.sleep(2)  # avoid tripping upstream engine rate limits
            results.append(await search(query, language))
        matches = []
        seen_urls = set()
        for (query, _), response in zip(searches, results):
            for result in response.get("results", [])[:max_results]:
                url = result.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                matches.append((query, result))

        async def fetch_html(url: str) -> tuple[bytes, str]:
            """Fetch a page; if the site blocks us, fall back to its Internet Archive snapshot."""
            try:
                async with session.get(url, allow_redirects=True) as response:
                    response.raise_for_status()
                    if "text/html" not in response.headers.get("Content-Type", ""):
                        raise ValueError(f"Search result is not an HTML page: {url}")
                    return await response.read(), ""
            except aiohttp.ClientResponseError as error:
                if error.status not in (401, 402, 403, 429):
                    raise
                async with session.get("https://archive.org/wayback/available", params={"url": url}) as response:
                    snapshot = (await response.json()).get("archived_snapshots", {}).get("closest")
                if not snapshot:
                    raise
                stamp = snapshot["timestamp"]
                async with session.get(f"https://web.archive.org/web/{stamp}id_/{url}") as response:
                    response.raise_for_status()
                    note = f"[Archived copy from {stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}; the live site blocks automated access]\n"
                    return await response.read(), note

        async def fetch_page(query: str, result: dict[str, Any]) -> dict[str, str]:
            url = result["url"]
            html, note = await fetch_html(url)
            page_text = await asyncio.to_thread(extract, html, url=url)
            if not page_text:
                raise ValueError(f"Could not extract article text from search result: {url}")
            metadata = await asyncio.to_thread(extract_metadata, html)
            return {
                "title": result.get("title") or (metadata.title if metadata and metadata.title else url),
                "snippet": " ".join((result.get("content") or "").split()),
                "url": metadata.url if metadata and metadata.url else url,
                "author": metadata.author if metadata and metadata.author else "",
                "date": metadata.date if metadata and metadata.date else "",
                "content": note + page_text,
            }

        pages = await asyncio.gather(
            *(fetch_page(query, result) for query, result in matches),
            return_exceptions=True,
        )

    formatted = []
    for (query, result), page in zip(matches, pages):
        if isinstance(page, BaseException):
            formatted.append(
                f"--- SOURCE UNAVAILABLE: {result.get('title') or result['url']} ---\n"
                f"URL: {result['url']}\n"
                f"Query: {query}\n"
                f"Page could not be read ({page}); no content available."
            )
            continue
        source_id = "S" + hashlib.sha1(page["url"].encode()).hexdigest()[:6]
        formatted.append(
            f"--- SOURCE [{source_id}]: {page['title']} ---\n"
            f"URL: {page['url']}\n"
            f"Author: {page['author'] or 'Not provided'}\n"
            f"Published: {page['date'] or 'Not provided'}\n"
            f"Query: {query}\n"
            f"Search snippet: {page['snippet'] or 'Not provided'}\n"
            f"CONTENT:\n{page['content']}"
        )
    if not formatted:
        unresponsive = sorted({f"{name} ({reason})" for response in results for name, reason in response.get("unresponsive_engines", [])})
        return (
            f"NO SOURCES FOUND for queries: {queries}. "
            f"Unresponsive search engines: {', '.join(unresponsive) or 'none'}."
        )
    return "\n\n".join(formatted)


SOURCE_ID = r"S[0-9a-f]{6}"


def finalize_citations(report: str, raw_notes: List[str]) -> str:
    """Number [S...] citations by first appearance and build Sources from retrieval metadata, not model text."""
    header = rf"--- SOURCE \[({SOURCE_ID})\]: (.*?) ---\nURL: (.*)\nAuthor: (.*)\nPublished: (.*)"
    registry = {m[0]: m[1:] for m in re.findall(header, "\n".join(raw_notes))}
    numbers: dict[str, int] = {}

    def renumber(match: re.Match) -> str:
        ids = [i for i in re.findall(SOURCE_ID, match.group(0)) if i in registry]
        for i in ids:
            numbers.setdefault(i, len(numbers) + 1)
        return "".join(f"[{numbers[i]}]" for i in ids)

    body = re.split(r"\n#+\s*Sources\b", report)[0]
    body = re.sub(rf"\[{SOURCE_ID}(?:\s*[,;]\s*{SOURCE_ID})*\]", renumber, body)
    if not numbers:
        return body
    lines = []
    for source_id, number in numbers.items():
        title, url, author, date = registry[source_id]
        details = ", ".join(x for x in (author, date) if x != "Not provided")
        lines.append(f"[{number}] {title}: {url}" + (f" ({details})" if details else ""))
    return body.rstrip() + "\n\n### Sources\n" + "\n".join(lines)


@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

##########################
# MCP Utils
##########################

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    """Exchange Supabase token for MCP access token using OAuth token exchange.
    
    Args:
        supabase_token: Valid Supabase authentication token
        base_mcp_url: Base URL of the MCP server
        
    Returns:
        Token data dictionary if successful, None if failed
    """
    try:
        # Prepare OAuth token exchange request data
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # Execute token exchange request
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # Successfully obtained token
                    token_data = await response.json()
                    return token_data
                else:
                    # Log error details for debugging
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")
                    
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    
    return None

async def get_tokens(config: RunnableConfig):
    """Retrieve stored authentication tokens with expiration validation.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        
    Returns:
        Token dictionary if valid and not expired, None otherwise
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # Retrieve stored tokens
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # Check token expiration
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # Token expired, clean up and return None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """Store authentication tokens in the configuration store.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        tokens: Token dictionary to store
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # Store the tokens
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """Fetch and refresh MCP tokens, obtaining new ones if needed.
    
    Args:
        config: Runtime configuration with authentication details
        
    Returns:
        Valid token dictionary, or None if unable to obtain tokens
    """
    # Try to get existing valid tokens first
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # Extract Supabase token for new token exchange
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    
    # Extract MCP configuration
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    
    # Exchange Supabase token for MCP tokens
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # Store the new tokens and return them
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap MCP tool with comprehensive authentication and error handling.
    
    Args:
        tool: The MCP structured tool to wrap
        
    Returns:
        Enhanced tool with authentication error handling
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        """Enhanced coroutine with MCP error handling and user-friendly messages."""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """Recursively search for MCP errors in exception chains."""
            if isinstance(exc, McpError):
                return exc
            
            # Handle ExceptionGroup (Python 3.11+) by checking attributes
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # Execute the original tool functionality
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # Search for MCP-specific errors in the exception chain
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # Not an MCP error, re-raise the original exception
                raise original_error
            
            # Handle MCP-specific error cases
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # Check for authentication/interaction required error
            if error_code == -32003:  # Interaction required error code
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                
                # Extract user-friendly message if available
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # Append URL if provided for user reference
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # For other MCP errors, re-raise the original
            raise original_error
    
    # Replace the tool's coroutine with our enhanced version
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """Load and configure MCP (Model Context Protocol) tools with authentication.
    
    Args:
        config: Runtime configuration containing MCP server details
        existing_tool_names: Set of tool names already in use to avoid conflicts
        
    Returns:
        List of configured MCP tools ready for use
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Step 1: Handle authentication if required
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # Step 2: Validate configuration requirements
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    
    if not config_valid:
        return []
    
    # Step 3: Set up MCP server connection
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    
    # Configure authentication headers if tokens are available
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: When Multi-MCP Server support is merged in OAP, update this code
    
    # Step 4: Load tools from MCP server
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # If MCP server connection fails, return empty list
        return []
    
    # Step 5: Filter and configure tools
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # Skip tools with conflicting names
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        
        # Only include tools specified in configuration
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        
        # Wrap tool with authentication handling and add to list
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    
    return configured_tools


##########################
# Tool Utils
##########################

async def get_search_tool(search_api: SearchAPI):
    """Configure and return search tools based on the specified API provider.
    
    Args:
        search_api: The search API provider to use (SearXNG or None)
        
    Returns:
        List of configured search tool objects for the specified provider
    """
    if search_api == SearchAPI.SEARXNG:
        search_tool = searxng_search
        search_tool.metadata = {
            **(search_tool.metadata or {}),
            "type": "search",
            "name": "web_search",
        }
        return [search_tool]

    return []
    
async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including research, search, and MCP tools.
    
    Args:
        config: Runtime configuration specifying search API and MCP settings
        
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]
    
    # Add configured search tools
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    
    # Track existing tool names to prevent conflicts
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # Add MCP tools if configured
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

##########################
# Token Limit Exceeded Utils
##########################

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a context/token limit was exceeded."""
    error_str = str(exception).lower()
    return any(
        phrase in error_str
        for phrase in ("context length", "context window", "too many tokens", "maximum context")
    )

# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
}

def get_model_token_limit(model_string):
    """Look up the token limit for a specific model.
    
    Args:
        model_string: The model identifier string to look up
        
    Returns:
        Token limit as integer if found, None if model not in lookup table
    """
    # Search through known model token limits
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # Model not found in lookup table
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.
    
    This is useful for handling token limit exceeded errors by removing recent context.
    
    Args:
        messages: List of message objects to truncate
        
    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            return messages[:i]
    
    # No AI messages found, return original list
    return messages

##########################
# Misc Utils
##########################

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.
    
    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """Extract value from configuration, handling enums and None values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value
