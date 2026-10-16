ResearchPilot

A local AI research assistant built with Streamlit, LangChain, Ollama, DuckDuckGo search, Requests, and BeautifulSoup.

Requirements

Python 3.10 or newer

Ollama installed and running

The gemma3:4b model downloaded

Internet access for web search and webpage retrieval

Setup (Windows PowerShell)

python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull gemma3:4b
streamlit run app.py

If Ollama is already running in the background, do not start a second Ollama server.

Features

Quick and Deep research modes

Live web search and webpage text extraction

Local report generation with Gemma 3

Source-number validation and an automated citation-support review

Research history and TXT/Markdown export

Important limitations

The citation audit is an automated aid, not proof that every claim is correct.

A readable webpage is not automatically authoritative.

Local inference speed depends on hardware; a research run can take a minute or more.

This version is intended for a local portfolio/demo workflow, not a public production service.

Review primary sources before relying on important, medical, legal, or financial claims.