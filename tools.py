from datetime import datetime
import os

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


# -------------------------
# Save research to history
# -------------------------

@tool
def save_text_to_file(data: str) -> str:
    """
    Save a completed research report.

    The report is saved as:
    - research_output.txt
    - individual TXT file inside research_history
    - individual Markdown file inside research_history
    """

    now = datetime.now()

    timestamp = now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    filename_timestamp = now.strftime(
        "%Y-%m-%d_%H%M%S"
    )

    formatted_text = (
        f"--- Research Output ---\n"
        f"Timestamp: {timestamp}\n\n"
        f"{data}\n\n"
    )

    # -------------------------
    # Save to main TXT log
    # -------------------------

    with open(
        "research_output.txt",
        "a",
        encoding="utf-8"
    ) as f:

        f.write(formatted_text)

    # -------------------------
    # Create history folder
    # -------------------------

    history_folder = "research_history"

    os.makedirs(
        history_folder,
        exist_ok=True
    )

    # -------------------------
    # Save TXT history file
    # -------------------------

    txt_file = os.path.join(
        history_folder,
        f"research_{filename_timestamp}.txt"
    )

    with open(
        txt_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(formatted_text)

    # -------------------------
    # Save Markdown history file
    # -------------------------

    md_file = os.path.join(
        history_folder,
        f"research_{filename_timestamp}.md"
    )

    with open(
        md_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(data)

    return (
        "Research saved successfully.\n"
        f"TXT file: {txt_file}\n"
        f"Markdown file: {md_file}"
    )


# -------------------------
# DuckDuckGo web search
# -------------------------

@tool
def search_web(query: str) -> str:
    """
    Search the web and return structured search results.

    Use this tool for:
    - current events
    - recent developments
    - latest information
    - news
    - recent technology
    - facts that may have changed over time
    """

    try:

        results = DDGS().text(
            query,
            max_results=5
        )

        if not results:
            return "No web search results found."

        formatted_results = [
            "WEB SEARCH RESULTS\n"
        ]

        for i, result in enumerate(
            results,
            start=1
        ):

            formatted_results.append(
                f"SOURCE {i}\n"
                f"Title: {result.get('title', 'Unknown')}\n"
                f"URL: {result.get('href', 'Unknown')}\n"
                f"Snippet: {result.get('body', 'No snippet available')}\n"
            )

        return "\n".join(
            formatted_results
        )

    except Exception as e:

        return (
            f"Web search failed: {str(e)}"
        )


# -------------------------
# Read a webpage
# -------------------------

@tool
def read_webpage(url: str) -> str:
    """
    Read a webpage and extract its main readable content.
    """

    try:

        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/142.0.0.0 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unwanted elements
        for element in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):

            element.decompose()

        # Find main content
        main_content = (
            soup.find("article")
            or soup.find("main")
        )

        if main_content:

            text = main_content.get_text(
                separator="\n",
                strip=True
            )

        else:

            text = soup.get_text(
                separator="\n",
                strip=True
            )

        # Clean blank lines
        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

        text = "\n".join(lines)

        # Limit webpage size
        max_length = 12000

        if len(text) > max_length:

            text = (
                text[:max_length]
                + "\n[Content truncated]"
            )

        if not text:

            return (
                f"WEBPAGE CONTENT\n"
                f"URL: {url}\n\n"
                f"No readable content could be extracted."
            )

        return (
            f"WEBPAGE CONTENT\n"
            f"URL: {url}\n\n"
            f"{text}"
        )

    except requests.exceptions.Timeout:

        return (
            f"Failed to read webpage: "
            f"request timed out for {url}"
        )

    except requests.exceptions.RequestException as e:

        return (
            f"Failed to read webpage: {str(e)}"
        )

    except Exception as e:

        return (
            f"Failed to process webpage: {str(e)}"
        )


# -------------------------
# Wikipedia search
# -------------------------

api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=100,
)

wiki = WikipediaQueryRun(
    api_wrapper=api_wrapper
)


@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for general factual information.
    """

    try:

        return wiki.invoke(query)

    except Exception as e:

        return (
            f"Wikipedia search failed: {str(e)}"
        )


# -------------------------
# Tools available to agent
# -------------------------

search_tool = search_web
read_tool = read_webpage
wiki_tool = search_wikipedia
save_tool = save_text_to_file