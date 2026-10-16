import html
import os
import re
import threading
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st

from main import research_topic
from tools import save_tool


# Process-wide guard: only one local LLM research run at a time.
# This prevents separate Streamlit sessions/tabs from overlapping research jobs.
_RESEARCH_LOCK = threading.Lock()


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ResearchPilot | AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

if "report" not in st.session_state:
    st.session_state["report"] = None

if "urls" not in st.session_state:
    st.session_state["urls"] = []

if "query" not in st.session_state:
    st.session_state["query"] = ""

if "saved" not in st.session_state:
    st.session_state["saved"] = False

if "research_depth" not in st.session_state:
    st.session_state["research_depth"] = "quick"


# =========================================================
# CUSTOM STYLING (DARK DASHBOARD THEME)
# =========================================================

st.markdown(
    """
    <style>
    /* Global layout adjustments */
    .block-container {
        max-width: 1180px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Top Telemetry Bar */
    .telemetry-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-bottom: 1.2rem;
    }

    .telemetry-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #171C26;
        border: 1px solid #30363D;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        color: #9AA4B2;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .telemetry-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
    }

    /* Compact workspace status strip */
    .workspace-status-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 0 0 1.15rem 0;
    }

    .workspace-status-item {
        display: flex;
        align-items: center;
        gap: 8px;
        min-width: 0;
        padding: 10px 12px;
        border: 1px solid #293241;
        border-radius: 10px;
        background: linear-gradient(135deg, #151B25, #11161E);
        color: #AAB4C3;
        font-size: 12px;
        line-height: 1.3;
    }

    .workspace-status-dot {
        flex: 0 0 7px;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #34D399;
        box-shadow: 0 0 8px rgba(52, 211, 153, 0.25);
    }

    .workspace-status-icon {
        flex: 0 0 auto;
        color: #7AA7FF;
        font-size: 16px;
    }

    .workspace-status-label {
        color: #7F8B9D;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.7px;
        white-space: nowrap;
    }

    .workspace-status-value {
        color: #E5EAF2;
        font-weight: 600;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    @media (max-width: 760px) {
        .workspace-status-strip {
            grid-template-columns: 1fr;
            gap: 6px;
        }
    }

    /* Hero Branding */
    .hero-title {
        font-size: 34px;
        font-weight: 750;
        letter-spacing: -0.8px;
        color: #F0F6FC;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 15px;
        color: #8B949E;
        margin-bottom: 1.5rem;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background-color: #171C26;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 14px 16px;
        height: 100%;
    }

    .kpi-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #8B949E;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-size: 19px;
        font-weight: 700;
        color: #F0F6FC;
        margin-bottom: 2px;
    }

    .kpi-caption {
        font-size: 12px;
        color: #8B949E;
    }

    /* Citation Warning Panel */
    .citation-warning-panel {
        background-color: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 1.5rem;
    }

    .citation-warning-title {
        font-size: 15px;
        font-weight: 700;
        color: #F59E0B;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }

    .citation-warning-intro {
        font-size: 13px;
        color: #D1D5DB;
        margin-bottom: 12px;
    }

    .warning-card {
        background-color: rgba(15, 18, 24, 0.75);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 3px solid #F59E0B;
        border-radius: 50%;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
    }

    .warning-src-badge {
        display: inline-block;
        background-color: #F59E0B;
        color: #0F1218;
        font-weight: 700;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 4px;
    }

    .warning-claim {
        font-size: 13px;
        color: #E6E9EF;
        font-style: italic;
        margin-bottom: 4px;
    }

    .warning-reason {
        font-size: 12px;
        color: #9AA4B2;
    }

    /* Source Cards */
    .source-card {
        background-color: #171C26;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }

    .source-domain-tag {
        display: inline-block;
        background-color: #21262D;
        border: 1px solid #30363D;
        color: #58A6FF;
        font-size: 11px;
        font-family: monospace;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 8px;
    }

    /* Feature Explanation Cards */
    .feature-card {
        background-color: #171C26;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        height: 100%;
    }

    .feature-icon {
        font-size: 22px;
        margin-bottom: 8px;
    }

    .feature-title {
        font-size: 14px;
        font-weight: 600;
        color: #F0F6FC;
        margin-bottom: 4px;
    }

    .feature-desc {
        font-size: 12px;
        color: #8B949E;
        line-height: 1.5;
    }

    /* Action bar container */
    .action-panel {
        background-color: #171C26;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        margin-top: 1.5rem;
    }

    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        padding-top: 1rem;
        background-color: #0F1218;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clear_research() -> None:
    """Reset session state for a fresh research query."""
    st.session_state["report"] = None
    st.session_state["urls"] = []
    st.session_state["query"] = ""
    st.session_state["saved"] = False


def get_history_files() -> list[str]:
    """Return sorted list of saved research files."""
    history_folder = "research_history"
    if not os.path.exists(history_folder):
        return []

    files = [
        file
        for file in os.listdir(history_folder)
        if file.endswith(".txt")
    ]
    files.sort(reverse=True)
    return files


def load_history_report(file_name: str) -> None:
    """Load a previously saved report file from history into session state."""
    history_folder = "research_history"
    file_path = os.path.join(history_folder, file_name)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        st.session_state["report"] = content
        st.session_state["urls"] = []
        st.session_state["saved"] = True

    except Exception as e:
        st.error(f"Could not open saved research: {str(e)}")


def clean_display_report(report: str) -> tuple[str, str | None]:
    """
    Cleans leading history headers and duplicate title headings for UI display.
    Returns (cleaned_body, extracted_timestamp).
    """
    if not report:
        return "", None

    text = report.strip()
    extracted_timestamp = None

    # Check for history file header
    history_match = re.match(
        r"^---\s*Research Output\s*---\s*\n+Timestamp:\s*([^\n]+)\s*\n+",
        text,
        flags=re.IGNORECASE
    )
    if history_match:
        extracted_timestamp = history_match.group(1).strip()
        text = text[history_match.end():].strip()

    # Strip duplicate '# Research Report' generated by AI or templates
    text = re.sub(
        r"^#\s+Research Report\s*\n*",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()

    return text, extracted_timestamp


def extract_sources_from_text(text: str) -> tuple[str, list[str]]:
    """
    If '## Sources' exists in text, extracts URLs and returns (text_without_sources, urls).
    Otherwise returns (text, []).
    """
    sources_pattern = r"\n+## Sources\s*\n+([\s\S]*)$"
    match = re.search(sources_pattern, text)
    if not match:
        return text, []

    sources_block = match.group(1)
    urls = []
    for line in sources_block.splitlines():
        line = line.strip()
        if line.startswith("- "):
            candidate = line[2:].strip()
            if candidate.startswith("http://") or candidate.startswith("https://"):
                urls.append(candidate)

    text_without_sources = re.sub(sources_pattern, "", text).strip()
    return text_without_sources, urls


def _parse_warning_item(item_text: str) -> dict:
    """Normalize markdown or HTML-formatted reviewer warnings into plain fields."""
    original = item_text.strip()

    # The reviewer sometimes returns an HTML card instead of the documented
    # `Source N | Claim: ... | Reason: ...` format. Convert that HTML to plain
    # text first, then parse the labels so the UI can render a clean card.
    plain = html.unescape(original)
    plain = re.sub(r"<\s*br\s*/?\s*>", " ", plain, flags=re.IGNORECASE)
    plain = re.sub(r"</\s*(div|span|p|strong|li)\s*>", " ", plain, flags=re.IGNORECASE)
    plain = re.sub(r"<[^>]*>", " ", plain)
    plain = re.sub(r"\s+", " ", plain).strip()

    source_match = re.search(r"\bSOURCE\s+(\d+)\b|\bSources?\s+(\d+)\b", plain, re.IGNORECASE)
    source = next((group for group in source_match.groups() if group), None) if source_match else None

    # Support both the normal markdown format and labels emitted by the model's HTML.
    claim_match = re.search(
        r"\bClaim\s*:\s*[\"'“]?(.+?)(?=[\"'”]?\s*(?:\|\s*Reason\s*:|\s+Evidence\s+Gap\s*:|\s+Reason\s*:)|$)",
        plain,
        re.IGNORECASE,
    )
    reason_match = re.search(r"\b(?:Evidence\s+Gap|Reason)\s*:\s*(.+)$", plain, re.IGNORECASE)

    if claim_match:
        claim = claim_match.group(1).strip().strip('"\'“” ')
    else:
        claim = None

    reason = reason_match.group(1).strip() if reason_match else None

    # Standard pipe-delimited format is more reliable when present.
    standard = re.search(
        r"Sources?\s+(\d+)\s*\|\s*Claim:\s*(.*?)\s*\|\s*Reason:\s*(.*)",
        original,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if standard:
        source = standard.group(1)
        claim = standard.group(2).strip().strip('"\'“” ')
        reason = standard.group(3).strip()

    return {
        "source": source,
        "claim": claim,
        "reason": reason,
        "raw": plain or original,
    }


def extract_citation_warnings(text: str) -> tuple[str, list[dict]]:
    """Extract reviewer warnings, including malformed HTML cards, from report text."""
    warning_header = "## Automated Citation Review Warnings"
    if warning_header not in text:
        return text, []

    pattern = (
        r"\n*## Automated Citation Review Warnings\s*\n+"
        r"(?:These are potential issues identified by an automated reviewer\.[^\n]*\n+)?"
        r"([\s\S]*?)(?=\n## |\Z)"
    )
    match = re.search(pattern, text)

    if not match:
        parts = text.split(warning_header, 1)
        body = parts[0].strip()
        warnings_raw = parts[1].strip()
        parsed = []
        for line in warnings_raw.splitlines():
            line = line.strip()
            if line.startswith("- "):
                parsed.append(_parse_warning_item(line[2:].strip()))
        return body, parsed

    warnings_block = match.group(1).strip()
    parsed_warnings = []

    # Usually each warning is one markdown bullet, but model-generated HTML may
    # span several lines. Split at bullets first; if no bullets exist, parse the
    # entire block as one warning.
    items = re.split(r"(?m)^\s*-\s+", warnings_block)
    items = [item.strip() for item in items if item.strip()]
    if not items and warnings_block:
        items = [warnings_block]

    for item_text in items:
        parsed_warnings.append(_parse_warning_item(item_text))

    body_without_warnings = (text[:match.start()] + text[match.end():]).strip()
    return body_without_warnings, parsed_warnings


def extract_audit_status(report: str) -> tuple[str, str]:
    """Read the explicit backend audit result and remove its internal marker."""
    pattern = r"(?m)^<!--\s*RESEARCHPILOT_AUDIT:(VERIFIED|ISSUES|INCONCLUSIVE)\s*-->\s*$"
    match = re.search(pattern, report or "", flags=re.IGNORECASE)
    status = match.group(1).upper() if match else "INCONCLUSIVE"
    clean_report = re.sub(pattern, "", report or "", flags=re.IGNORECASE).rstrip()
    return status, clean_report


def is_research_failure(report: str, urls: list[str]) -> bool:
    """Detect backend failure messages even when some source URLs were found."""
    if not report or not report.strip():
        return True

    failure_prefixes = (
        "research could not continue",
        "research report generation failed",
        "research failed",
        "the ai model returned an empty report",
    )
    normalized = report.strip().lower()
    return normalized.startswith(failure_prefixes)


def prepare_export_report(raw_report: str, urls: list[str]) -> str:
    """
    Prepare a clean export with citation warnings and one source list.
    Internal audit markers are UI metadata and are not included in exports.
    """
    _, clean_report = extract_audit_status(raw_report or "")
    report_body, embedded_urls = extract_sources_from_text(clean_report)
    final_urls = urls or embedded_urls

    export_text = report_body.rstrip()
    if final_urls:
        export_text += "\n\n## Sources\n\n" + "\n".join(f"- {url}" for url in final_urls) + "\n"
    return export_text


def format_timestamp(raw_timestamp: str) -> str:
    """Format file or report timestamp into a human-readable string."""
    try:
        # Check standard datetime format: YYYY-MM-DD HH:MM:SS
        dt = datetime.strptime(raw_timestamp.strip(), "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%b %d, %Y · %H:%M")
    except Exception:
        pass

    try:
        # Check filename timestamp: YYYY-MM-DD_HHMMSS
        clean_name = raw_timestamp.replace("research_", "").replace(".txt", "")
        dt = datetime.strptime(clean_name, "%Y-%m-%d_%H%M%S")
        return dt.strftime("%b %d, %Y · %H:%M")
    except Exception:
        return raw_timestamp


# =========================================================
# SIDEBAR NAVIGATION & HISTORY
# =========================================================

with st.sidebar:
    st.markdown("### 🔬 ResearchPilot")
    st.caption("Autonomous AI Research Assistant")

    st.markdown(
        """
        <div class="telemetry-bar" style="margin-top: 6px; margin-bottom: 12px;">
            <div class="telemetry-badge" style="font-size: 11px;">
                <span class="telemetry-dot"></span> Ollama (gemma3:4b)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # -----------------------------------------------------
    # Research depth selector (synced with session state)
    # -----------------------------------------------------
    st.markdown("**Research Configuration**")

    depth_options = ["Quick", "Deep"]
    current_depth_index = 0 if st.session_state["research_depth"] == "quick" else 1

    selected_depth = st.segmented_control(
        "Research depth",
        options=depth_options,
        default=depth_options[current_depth_index],
        label_visibility="collapsed"
    )

    if selected_depth is None:
        selected_depth = depth_options[current_depth_index]

    st.session_state["research_depth"] = selected_depth.lower()

    if selected_depth == "Quick":
        st.caption("⚡ Quick · 2 source candidates · Concise briefing")
    else:
        st.caption("🔬 Deep · 4 source candidates · Detailed comparison")

    st.divider()

    # -----------------------------------------------------
    # New research button
    # -----------------------------------------------------
    if st.button("🔄 New Research Session", use_container_width=True):
        clear_research()
        st.rerun()

    st.divider()

    # -----------------------------------------------------
    # Research history archive
    # -----------------------------------------------------
    st.markdown("**📁 Research History**")

    history_files = get_history_files()

    if history_files:
        recent_files = history_files[:8]
        for index, file_name in enumerate(recent_files):
            readable_time = format_timestamp(file_name)
            button_label = f"📄 {readable_time}"

            if st.button(
                button_label,
                key=f"hist_btn_{index}_{file_name}",
                use_container_width=True
            ):
                load_history_report(file_name)
                st.rerun()
    else:
        st.caption("No archived research sessions yet.")

    st.divider()

    # -----------------------------------------------------
    # About expander
    # -----------------------------------------------------
    with st.expander("ℹ️ System Architecture"):
        st.markdown(
            """
            **ResearchPilot Engine**
            - **LLM**: Ollama (`gemma3:4b`)
            - **Framework**: LangChain
            - **Search**: DuckDuckGo API
            - **Parsing**: BeautifulSoup4
            - **Validation**: Citation number verification & LLM claim audit

            **Pipeline Flow**
            1. Query formulation & web retrieval
            2. Webpage content reading & cleaning
            3. Multi-source evidence synthesis
            4. Citation number checking & claim audit
            """
        )


# =========================================================
# SCREEN 1: HOME / SEARCH DASHBOARD (report is None)
# =========================================================

if st.session_state["report"] is None:

    # Hero title
    st.markdown(
        '<div class="hero-title">Research Workspace</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">'
        'Formulate a research inquiry. ResearchPilot retrieves live sources, '
        'synthesizes findings, and audits citations.'
        '</div>',
        unsafe_allow_html=True
    )

    # Top telemetry bar
    st.markdown(
        """
        <div class="workspace-status-strip">
            <div class="workspace-status-item">
                <span class="workspace-status-dot"></span>
                <span class="workspace-status-label">SYSTEM</span>
                <span class="workspace-status-value">Ready</span>
            </div>
            <div class="workspace-status-item">
                <span class="workspace-status-icon">◈</span>
                <span class="workspace-status-label">AI ENGINE</span>
                <span class="workspace-status-value">Gemma 3 · Local</span>
            </div>
            <div class="workspace-status-item">
                <span class="workspace-status-icon">◎</span>
                <span class="workspace-status-label">SOURCES</span>
                <span class="workspace-status-value">Live Web Search</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Main query container
    with st.container(border=True):
        st.markdown("**What topic or question do you want to investigate?**")

        query = st.text_area(
            "Research question",
            value=st.session_state["query"],
            placeholder=(
                "e.g., What are the current clinical applications of artificial "
                "intelligence in healthcare, and what are the primary validation hurdles?"
            ),
            height=120,
            label_visibility="collapsed"
        )
        st.session_state["query"] = query

        col_mode, col_btn = st.columns([1, 1], vertical_alignment="center")

        with col_mode:
            current_mode_label = "⚡ Quick Research" if st.session_state["research_depth"] == "quick" else "🔬 Deep Research"
            st.caption(f"Active Mode: **{current_mode_label}** (configured in sidebar)")

        with col_btn:
            start_btn = st.button(
                "🔎 Launch Research",
                type="primary",
                use_container_width=True
            )


    # -----------------------------------------------------
    # Execute Research
    # -----------------------------------------------------
    if start_btn:
        if not query.strip():
            st.warning("Please enter a research topic or question to begin.")
        elif not _RESEARCH_LOCK.acquire(blocking=False):
            st.warning(
                "Another research task is already running in this app process. "
                "Please wait for it to finish before starting another task."
            )
        else:
            depth = st.session_state["research_depth"]
            mode_title = "Deep Research" if depth == "deep" else "Quick Research"

            # The process-wide lock prevents separate Streamlit sessions/tabs
            # from running expensive local-model research concurrently.
            try:
                with st.status(
                    f"🔎 {mode_title}: Preparing research…",
                    expanded=True
                ) as status:
                    progress_detail = st.empty()
                    progress_detail.info(
                        "Your request has been submitted. The panel will update as each backend stage starts. "
                        "This may take a minute or two; please keep this page open."
                    )
                    st.caption("Live status is connected to the actual research pipeline.")

                    stage_labels = {
                        "search": ("🔎 Searching the web", "running"),
                        "reading": ("📚 Reading and verifying sources", "running"),
                        "generation": ("🧠 Generating the research report", "running"),
                        "review": ("🔍 Reviewing citations", "running"),
                        "complete": ("✅ Research completed", "complete"),
                        "error": ("❌ Research could not continue", "error"),
                    }

                    def update_research_progress(stage: str, message: str) -> None:
                        label, state = stage_labels.get(
                            stage,
                            ("🔄 Research in progress", "running"),
                        )
                        status.update(label=label, state=state, expanded=True)
                        if stage == "error":
                            progress_detail.error(message)
                        elif stage == "complete":
                            progress_detail.success(message)
                        else:
                            progress_detail.info(message)

                    try:
                        report, urls = research_topic(
                            query,
                            depth,
                            progress_callback=update_research_progress,
                        )

                        st.session_state["report"] = report
                        st.session_state["urls"] = urls
                        st.session_state["saved"] = False

                        if is_research_failure(report, urls):
                            status.update(
                                label="❌ Research could not be completed",
                                state="error",
                                expanded=True
                            )
                            progress_detail.error(
                                "Research did not produce a usable report. Review the message above and try again."
                            )
                        else:
                            status.update(
                                label="✅ Research Completed",
                                state="complete",
                                expanded=True
                            )
                        st.rerun()

                    except Exception as e:
                        status.update(
                            label="❌ Research Failed",
                            state="error",
                            expanded=True
                        )
                        progress_detail.error(
                            "An error occurred while executing the research pipeline."
                        )
                        st.caption(str(e))
            finally:
                _RESEARCH_LOCK.release()

    # -----------------------------------------------------
    # Suggested Research Prompts
    # -----------------------------------------------------
    st.markdown("##### 💡 Suggested Inquiries")

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown("🤖 **AI in Healthcare**")
            st.caption("Clinical diagnostics, treatment planning, and hospital robotics.")
            if st.button("Use Prompt", key="prompt_healthcare", use_container_width=True):
                st.session_state["query"] = (
                    "What are the applications of artificial intelligence in healthcare?"
                )
                st.rerun()

    with c2:
        with st.container(border=True):
            st.markdown("💻 **Future of AI**")
            st.caption("AGI trajectories, compute limits, and next-generation architectures.")
            if st.button("Use Prompt", key="prompt_future_ai", use_container_width=True):
                st.session_state["query"] = (
                    "What is the future of artificial intelligence?"
                )
                st.rerun()

    with c3:
        with st.container(border=True):
            st.markdown("🔐 **AI in Cybersecurity**")
            st.caption("Threat detection, defensive automation, and vulnerability assessment.")
            if st.button("Use Prompt", key="prompt_cybersecurity", use_container_width=True):
                st.session_state["query"] = (
                    "How is artificial intelligence being used in cybersecurity?"
                )
                st.rerun()

    # -----------------------------------------------------
    # Architecture Overview Cards
    # -----------------------------------------------------
    st.write("")
    st.markdown("##### 🛡️ Autonomous Pipeline Features")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <div class="feature-title">Live Web Discovery</div>
                <div class="feature-desc">
                    Retrieves authoritative web sources using DuckDuckGo, prioritizing .gov, .edu, and peer-reviewed scientific domains.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📖</div>
                <div class="feature-title">Content Extraction</div>
                <div class="feature-desc">
                    Scrapes and strips non-content DOM elements, parsing clean text evidence for grounded LLM synthesis.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🛡️</div>
                <div class="feature-title">Citation Review</div>
                <div class="feature-desc">
                    Checks citation numbering and uses an automated reviewer to flag possible evidence gaps; results may be inconclusive.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )



# =========================================================
# SCREEN 2: REPORT WORKSPACE (report is not None)
# =========================================================

else:
    stored_report = st.session_state["report"]
    citation_audit_status, raw_report = extract_audit_status(stored_report)
    state_urls = st.session_state.get("urls", [])
    depth = st.session_state.get("research_depth", "quick")

    # Some backend failures are returned as report text instead of raising an exception.
    # Detect these before rendering report metrics, so a failed run is not labelled verified.
    if is_research_failure(raw_report, state_urls):
        st.markdown("## Research could not be completed")
        st.error(raw_report.strip())
        st.warning(
            "No usable sources were returned, so no report or citation audit was completed. "
            "This does not mean the citations were verified."
        )
        retry_col, new_col = st.columns(2)
        with retry_col:
            if st.button("🔁 Try Again", type="primary", use_container_width=True):
                st.session_state["report"] = None
                st.session_state["urls"] = []
                st.session_state["saved"] = False
                st.rerun()
        with new_col:
            if st.button("🔄 New Research", use_container_width=True, key="failed_new_research"):
                clear_research()
                st.rerun()
        st.stop()

    # Clean display body and extract history timestamp if present
    cleaned_body, history_timestamp = clean_display_report(raw_report)

    # Extract sources from text if loaded from history with empty urls list
    body_no_sources, text_urls = extract_sources_from_text(cleaned_body)
    active_urls = state_urls if state_urls else text_urls

    # Extract automated citation review warnings for dedicated UI panel
    display_body, citation_warnings = extract_citation_warnings(body_no_sources)

    # Word count and reading time calculation
    word_count = len(display_body.split())
    reading_time_min = max(1, round(word_count / 200))

    # Construct complete export version (guarantees warnings & sources are preserved)
    full_report = prepare_export_report(raw_report, active_urls)

    # -----------------------------------------------------
    # Header & Metric Bar
    # -----------------------------------------------------
    mode_label = "⚡ Quick Research" if depth == "quick" else "🔬 Deep Research"
    date_display = format_timestamp(history_timestamp) if history_timestamp else "Just now"

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">
            <div class="hero-title" style="font-size: 28px;">Research Dossier</div>
            <div style="color: #8B949E; font-size: 13px;">{date_display}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 KPI Metric Cards
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Research Mode</div>
                <div class="kpi-value">{mode_label}</div>
                <div class="kpi-caption">{"2 source candidates" if depth == "quick" else "4 source candidates"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Sources Read</div>
                <div class="kpi-value">{len(active_urls)} Available</div>
                <div class="kpi-caption">Retrieved webpages</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Report Length</div>
                <div class="kpi-value">~{word_count} words</div>
                <div class="kpi-caption">~{reading_time_min} min read time</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with m4:
        if citation_warnings or citation_audit_status == "ISSUES":
            issue_count = len(citation_warnings)
            audit_val = f"⚠️ {issue_count} Issue(s)" if issue_count else "⚠️ Review needed"
            audit_cap = "Evidence gaps flagged" if issue_count else "Reviewer flagged an issue; details unavailable"
        elif citation_audit_status == "VERIFIED":
            audit_val = "✅ Reviewed"
            audit_cap = "No issues identified by automated review"
        else:
            audit_val = "🟡 Inconclusive"
            audit_cap = "Evidence review not confirmed"

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Citation Audit</div>
                <div class="kpi-value">{audit_val}</div>
                <div class="kpi-caption">{audit_cap}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # -----------------------------------------------------
    # Citation-Warning Panel: use Streamlit native components for warning details.
    # This avoids displaying model-generated HTML as literal text.
    if citation_warnings:
        st.markdown(
            """
            <div class="citation-warning-panel">
                <div class="citation-warning-title">
                    ⚠️ Automated Citation Review Warnings
                </div>
                <div class="citation-warning-intro">
                    The automated reviewer flagged potential discrepancies between claims
                    in this report and the available source evidence. Verify important claims.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for warning in citation_warnings:
            source = str(warning.get("source") or "Unknown")
            claim = str(warning.get("claim") or "Claim unavailable")
            reason = str(warning.get("reason") or "Evidence gap detected.")
            st.warning(
                f"**Source {source}**\n\n"
                f'**Claim:** "{claim}"\n\n'
                f"**Evidence Gap:** {reason}"
            )

    if citation_audit_status == "ISSUES" and not citation_warnings:
        st.warning(
            "The citation reviewer flagged a potential issue, but the warning details "
            "could not be parsed for display. Treat affected claims as unverified."
        )
    elif citation_audit_status == "INCONCLUSIVE":
        st.info(
            "Automated citation review was inconclusive or was not completed. "
            "The report's source links are available, but claims should be checked manually."
        )

    # -----------------------------------------------------
    # Report Dossier Content
    # -----------------------------------------------------
    with st.container(border=True):
        st.markdown(display_body)

    # -----------------------------------------------------
    # Organized Source Cards
    # -----------------------------------------------------
    if active_urls:
        st.write("")
        st.markdown(f"#### 🔗 Retrieved Sources ({len(active_urls)})")
        st.caption("Webpages retrieved and checked for readable content; verify important claims at the source.")

        source_cols = st.columns(min(len(active_urls), 2))
        for index, url in enumerate(active_urls, start=1):
            col_target = source_cols[(index - 1) % len(source_cols)]
            domain = urlparse(url).netloc.replace("www.", "") or "webpage"

            with col_target:
                with st.container(border=True):
                    st.markdown(f"**Source {index}** <span class='source-domain-tag'>{domain}</span>", unsafe_allow_html=True)
                    st.caption(f"[{url}]({url})")

    # -----------------------------------------------------
    # Action Toolbar & Export Suite
    # -----------------------------------------------------
    st.write("")
    with st.container(border=True):
        st.markdown("##### 📥 Export & Session Actions")

        act_col1, act_col2, act_col3, act_col4 = st.columns(4)

        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

        with act_col1:
            if st.session_state["saved"]:
                st.button("✓ Saved in History", disabled=True, use_container_width=True)
            else:
                if st.button("💾 Save to History", type="primary", use_container_width=True):
                    try:
                        save_tool.invoke({"data": full_report})
                        st.session_state["saved"] = True
                        st.success("Archived to research history.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Save failed: {str(e)}")

        with act_col2:
            st.download_button(
                "📄 Download TXT",
                data=full_report,
                file_name=f"research_report_{timestamp_slug}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with act_col3:
            st.download_button(
                "📝 Download Markdown",
                data=full_report,
                file_name=f"research_report_{timestamp_slug}.md",
                mime="text/markdown",
                use_container_width=True
            )

        with act_col4:
            if st.button("🔄 New Research", use_container_width=True):
                clear_research()
                st.rerun()