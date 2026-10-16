import json
import os
import re
import time
from typing import Callable, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from tools import search_tool, read_tool, save_tool


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

MODEL_NAME = "gemma3:4b"

# Use separate generation limits for Quick and Deep Research.
# A lower token budget makes Quick Research faster and more concise.
quick_llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
    num_predict=400,
)

deep_llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.2,
    num_predict=1000,
)

# Citation reviewer: limit its response to a short audit.
citation_llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    num_predict=200,
    format="json",
)

PREFERRED_DOMAINS = [
    "who.int",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "nature.com",
    "sciencedirect.com",
    "springer.com",
    "arxiv.org",
    "ieee.org",
    "gov",
    "edu",
]


# ============================================================
# URL AND SOURCE UTILITIES
# ============================================================

def extract_urls(text: str) -> list[str]:
    """Extract unique URLs from search results."""

    if not text:
        return []

    matches = re.findall(r"https?://[^\s<>\"']+", text)

    urls = []

    for url in matches:
        url = url.rstrip(".,;:!?)\u005d}")

        if url not in urls:
            urls.append(url)

    return urls


def select_urls(urls: list[str], limit: int) -> list[str]:
    """
    Prioritize URLs from selected domains.

    This is a selection preference, not proof that a source
    is authoritative or that its claims are correct.
    """

    preferred = []
    other = []

    for url in urls:
        try:
            hostname = urlparse(url).hostname or ""
            hostname = hostname.lower()

            is_preferred = any(
                hostname == domain
                or hostname.endswith("." + domain)
                for domain in PREFERRED_DOMAINS
            )

            if is_preferred:
                preferred.append(url)
            else:
                other.append(url)

        except Exception:
            other.append(url)

    return (preferred + other)[:limit]


def is_valid_webpage(content: str) -> bool:
    """
    Basic readability check.

    A page passing this check is not necessarily trustworthy;
    it only means that some usable text was extracted.
    """

    if not content:
        return False

    content_lower = content.lower()

    failed_phrases = [
        "failed to read webpage:",
        "failed to process webpage:",
        "no readable content could be extracted.",
        "request timed out",
        "access denied",
        "403 forbidden",
        "404 not found",
    ]

    if any(phrase in content_lower for phrase in failed_phrases):
        return False

    return len(content.strip()) >= 200


def normalize_depth(depth) -> str:
    """Normalize Quick/Deep Research labels."""

    value = str(depth).strip().lower()

    if "deep" in value:
        return "deep"

    return "quick"


def select_relevant_evidence(
    source_text: str,
    report_text: str,
    max_chars: int = 2400,
) -> str:
    """Select relevant source sentences without reviewing only the page opening."""
    if not source_text or not source_text.strip():
        return ""

    sentences = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", source_text)
        if part.strip()
    ]
    if not sentences:
        return source_text[:max_chars]

    stop_words = {
        "about", "after", "also", "among", "because", "been", "being",
        "between", "could", "does", "each", "from", "have", "into",
        "more", "most", "other", "over", "same", "some", "such", "than",
        "that", "their", "them", "there", "these", "they", "this", "those",
        "through", "under", "using", "very", "were", "what", "when",
        "where", "which", "while", "with", "would", "your", "source",
    }

    report_terms = {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", report_text)
        if word.lower() not in stop_words
    }

    scored = []
    for index, sentence in enumerate(sentences):
        sentence_terms = {
            word.lower()
            for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", sentence)
            if word.lower() not in stop_words
        }
        overlap = len(report_terms.intersection(sentence_terms))
        has_number = bool(
            re.search(
                r"\d|%|\bpercent\b|\btrillion\b|\bbillion\b|\bmillion\b",
                sentence.lower(),
            )
        )
        score = overlap * 2 + (4 if has_number else 0)
        scored.append((score, index, sentence))

    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    chosen = []
    used_chars = 0
    for score, index, sentence in ranked:
        addition_size = len(sentence) + 1
        if used_chars + addition_size > max_chars:
            continue
        chosen.append((index, sentence))
        used_chars += addition_size
        if used_chars >= max_chars * 0.85:
            break

    if not chosen:
        return source_text[:max_chars]

    chosen.sort(key=lambda item: item[0])
    return "\n".join(sentence for _, sentence in chosen)


# ============================================================
# CITATION NUMBER VALIDATION
# ============================================================

def validate_citations(
    report: str,
    verified_source_count: int,
) -> list[int]:
    """
    Find references to source numbers outside the available
    verified source range.

    This checks citation numbering only. It does not prove
    that a citation supports the associated claim.
    """

    if not report or verified_source_count <= 0:
        return []

    citation_pattern = re.compile(
        r"\bSources?\s+(\d+)\b",
        re.IGNORECASE,
    )

    invalid_numbers = []

    for match in citation_pattern.finditer(report):
        number = int(match.group(1))

        if number < 1 or number > verified_source_count:
            if number not in invalid_numbers:
                invalid_numbers.append(number)

    if invalid_numbers:
        print(
            "\n⚠️ Citation number warning: "
            f"the report refers to unavailable source number(s): "
            f"{invalid_numbers}"
        )
    else:
        print("\n✅ Citation number validation passed.")

    return invalid_numbers


# ============================================================
# SHORT CITATION SUPPORT REVIEW
# ============================================================

AUDIT_MARKER_PREFIX = "<!-- RESEARCHPILOT_AUDIT:"


def with_audit_status(report: str, status: str) -> str:
    """Append an internal, UI-readable citation audit status marker."""
    clean = re.sub(
        r"(?m)^<!--\s*RESEARCHPILOT_AUDIT:(?:VERIFIED|ISSUES|INCONCLUSIVE)\s*-->\s*$",
        "",
        report or "",
        flags=re.IGNORECASE,
    ).rstrip()
    return f"{clean}\n\n<!-- RESEARCHPILOT_AUDIT:{status.upper()} -->"


def check_claim_support(
    report: str,
    verified_sources: list[dict],
) -> str:
    """Review cited report paragraphs and preserve the report on audit uncertainty."""
    if not report or not verified_sources:
        print("⚠️ Citation review skipped: report or verified sources are missing.")
        return with_audit_status(report, "INCONCLUSIVE")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", report) if p.strip()]
    cited_paragraphs = [
        paragraph
        for paragraph in paragraphs
        if re.search(r"\(\s*Sources?\s+\d+\s*\)", paragraph, re.IGNORECASE)
        or re.search(r"\[\s*Sources?\s+\d+\s*\]", paragraph, re.IGNORECASE)
    ]

    if not cited_paragraphs:
        print("ℹ️ Citation review skipped: no source-cited paragraphs found.")
        return with_audit_status(report, "INCONCLUSIVE")

    review_report = "\n\n".join(cited_paragraphs)[:3000]
    evidence_blocks = []
    valid_source_ids = set()

    for source in verified_sources:
        try:
            source_number = int(source.get("id"))
        except (TypeError, ValueError):
            continue

        source_text = source.get("content", "")
        if source_number < 1 or not isinstance(source_text, str) or not source_text.strip():
            continue

        valid_source_ids.add(source_number)
        relevant_excerpt = select_relevant_evidence(
            source_text=source_text,
            report_text=review_report,
            max_chars=1600,
        )
        evidence_blocks.append(
            f"SOURCE {source_number}\nURL: {source.get('url', '')}\n"
            f"RELEVANT EVIDENCE EXCERPT:\n{relevant_excerpt}"
        )

    if not evidence_blocks:
        print("⚠️ No usable source evidence for citation review.")
        return with_audit_status(report, "INCONCLUSIVE")

    review_prompt = f"""
You are auditing whether cited claims in a report are supported by the supplied evidence.

Review the report paragraphs against the source evidence. Pay special attention
to numerical claims, dates, percentages, counts, rankings, and forecasts.
Do not use outside knowledge.

Return JSON only, using exactly one of these forms:
{{"status":"no_issues","warnings":[]}}
{{"status":"review_needed","warnings":[{{"source":1,"claim":"short exact quote from the report","reason":"specific evidence gap"}}]}}

Rules:
- Use status no_issues only if no clear material issue is identified.
- For a warning, quote a short exact phrase from the report's cited paragraph.
- The source number must be one of the source IDs cited in that paragraph.
- Explain what the supplied evidence does not establish; do not claim a figure is false.
- Do not flag headings or reasonable paraphrases.
- Return at most 2 warnings. Do not add commentary.

REPORT PARAGRAPHS:
{review_report}

SOURCE EVIDENCE:
{chr(10).join(evidence_blocks)}
""".strip()

    print("\n🔍 Reviewing cited claims against source evidence...")
    review_start = time.time()

    try:
        response = citation_llm.invoke(review_prompt)
        review_result = str(response.content or "").strip()
        elapsed = time.time() - review_start
        print(f"⏱️ Citation review time: {elapsed:.2f} seconds")
        print(f"Reviewer response preview: {review_result[:800]!r}")

        if not review_result:
            print("⚠️ Citation reviewer returned an empty response; audit is inconclusive.")
            return with_audit_status(report, "INCONCLUSIVE")

        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            review_result,
            flags=re.IGNORECASE,
        ).strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                print("⚠️ Citation reviewer response contained no JSON object.")
                return with_audit_status(report, "INCONCLUSIVE")
            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError:
                print("⚠️ Citation reviewer returned malformed JSON.")
                return with_audit_status(report, "INCONCLUSIVE")

        if not isinstance(result, dict):
            print("⚠️ Citation reviewer returned unexpected JSON; audit is inconclusive.")
            return with_audit_status(report, "INCONCLUSIVE")

        status = str(result.get("status", "")).strip().lower()
        if status == "no_issues":
            reviewer_warnings = result.get("warnings")
            if not isinstance(reviewer_warnings, list) or reviewer_warnings:
                print("⚠️ Reviewer returned inconsistent no_issues status; audit is inconclusive.")
                return with_audit_status(report, "INCONCLUSIVE")

            print("✅ Citation reviewer completed; no issues were identified.")
            return with_audit_status(report, "VERIFIED")

        if status != "review_needed":
            print(f"⚠️ Citation reviewer returned unknown status {status!r}; audit is inconclusive.")
            return with_audit_status(report, "INCONCLUSIVE")

        raw_warnings = result.get("warnings")
        if not isinstance(raw_warnings, list) or not raw_warnings:
            print("⚠️ Reviewer requested review but returned no warning list items.")
            return with_audit_status(report, "INCONCLUSIVE")

        paragraph_sources = []
        for paragraph in cited_paragraphs:
            source_ids = set()
            for match in re.finditer(
                r"\(\s*Sources?\s+(\d+)\s*\)|\[\s*Sources?\s+(\d+)\s*\]",
                paragraph,
                flags=re.IGNORECASE,
            ):
                number_text = match.group(1) or match.group(2)
                if number_text:
                    source_ids.add(int(number_text))
            paragraph_sources.append((paragraph, source_ids))

        def normalize_text(value: str) -> str:
            value = re.sub(r"[^a-z0-9]+", " ", value.casefold())
            return re.sub(r"\s+", " ", value).strip()

        warnings = []
        rejected_reasons = []

        for item in raw_warnings:
            if not isinstance(item, dict):
                rejected_reasons.append("warning item was not an object")
                continue

            try:
                source_number = int(item.get("source"))
            except (TypeError, ValueError):
                rejected_reasons.append("warning had no valid source number")
                continue

            claim = re.sub(r"\s+", " ", str(item.get("claim", ""))).strip().strip("\"'“”")
            reason = re.sub(r"\s+", " ", str(item.get("reason", ""))).strip()

            if source_number not in valid_source_ids:
                rejected_reasons.append(f"source {source_number} is not a verified source")
                continue
            if not claim or not reason:
                rejected_reasons.append(f"source {source_number} warning lacked claim or reason")
                continue

            matching_paragraphs = [
                paragraph for paragraph, ids in paragraph_sources if source_number in ids
            ]
            normalized_claim = normalize_text(claim)

            def claim_matches_paragraph(paragraph: str) -> bool:
                normalized_paragraph = normalize_text(paragraph)

                # Prefer an exact normalized match. If the small local model
                # adds minor wording/punctuation differences, allow a strict
                # token-coverage fallback while keeping the source-ID check.
                if len(normalized_claim) >= 12 and normalized_claim in normalized_paragraph:
                    return True

                claim_tokens = re.findall(r"[a-z0-9]+", normalized_claim)
                paragraph_tokens = set(re.findall(r"[a-z0-9]+", normalized_paragraph))
                if len(claim_tokens) < 4:
                    return False

                coverage = sum(token in paragraph_tokens for token in claim_tokens) / len(claim_tokens)
                return coverage >= 0.80

            matching_claim = any(
                claim_matches_paragraph(paragraph)
                for paragraph in matching_paragraphs
            )

            if not matching_claim:
                rejected_reasons.append(
                    f"source {source_number} warning claim did not match a paragraph citing that source"
                )
                continue

            warning_text = (
                f"Source {source_number} | Claim: {claim[:180]} | Reason: {reason[:240]}"
            )
            if warning_text not in warnings:
                warnings.append(warning_text)
            if len(warnings) >= 2:
                break

        if not warnings:
            print("⚠️ Reviewer flagged issues, but none passed validation.")
            print("Rejected warning reasons: " + "; ".join(rejected_reasons[:5]))
            return with_audit_status(report, "INCONCLUSIVE")

        warning_section = (
            "\n\n## Automated Citation Review Warnings\n\n"
            "These are potential issues identified by an automated reviewer. "
            "Check the cited source before changing the report.\n\n"
            + "\n".join(f"- {warning}" for warning in warnings)
        )
        print(f"⚠️ Citation review flagged {len(warnings)} potential issue(s).")
        return with_audit_status(report.rstrip() + warning_section, "ISSUES")

    except Exception as error:
        elapsed = time.time() - review_start
        print(f"⚠️ Citation review failed: {error}")
        print(f"⏱️ Time before failure: {elapsed:.2f} seconds")
        print("Citation audit is inconclusive; preserving the original report.")
        return with_audit_status(report, "INCONCLUSIVE")



# ============================================================

def perform_searches(query: str, depth: str) -> list[str]:
    """Run the required web searches."""

    if depth == "deep":
        queries = [
            query,
            f"{query} research studies evidence",
        ]
    else:
        queries = [query]

    search_results = []

    for search_query in queries:
        try:
            result = search_tool.invoke(
                {"query": search_query}
            )

            if isinstance(result, str):
                search_results.append(result)

            else:
                search_results.append(str(result))

        except Exception as error:
            print(
                f"⚠️ Search failed for "
                f"'{search_query}': {error}"
            )

    return search_results


# ============================================================
# MAIN RESEARCH ENGINE
# ============================================================

def research_topic(
    query: str,
    depth: str = "quick",
    progress_callback: Optional[Callable[[str, str], None]] = None,
):
    """
    Run research and return:

        (report_text, verified_urls)

    This return format is required by the existing app.py.
    """

    def notify(stage: str, message: str) -> None:
        """Send best-effort progress updates to the UI without breaking research."""
        if progress_callback is not None:
            try:
                progress_callback(stage, message)
            except Exception as callback_error:
                print(f"⚠️ Progress update failed: {callback_error}")

    query = query.strip()

    if not query:
        notify("error", "Please enter a research topic.")
        return "Please enter a research topic.", []

    depth = normalize_depth(depth)

    total_start = time.time()

    if depth == "deep":
        max_sources = 4
        evidence_limit = 2200
        research_label = "Deep Research"
    else:
        max_sources = 2
        # Reduce prompt size for Quick Research while preserving useful evidence.
        evidence_limit = 2200
        research_label = "Quick Research"

    print("\n" + "=" * 55)
    print(f"🔎 {research_label}: {query}")
    print("=" * 55)

    # --------------------------------------------------------
    # 1. SEARCH
    # --------------------------------------------------------

    print("\n🔎 Searching the web...")
    notify("search", "Searching the web for relevant sources…")

    search_start = time.time()

    search_results = perform_searches(query, depth)

    search_text = "\n\n".join(search_results)

    search_urls = extract_urls(search_text)

    search_urls = select_urls(
        search_urls,
        limit=10 if depth == "deep" else 5,
    )

    search_elapsed = time.time() - search_start

    print("\n✅ Web search completed.")
    print(f"⏱️ Search time: {search_elapsed:.2f} seconds")

    if not search_urls:
        message = (
            "Research could not continue because no usable "
            "web search URLs were found. Please try again."
        )
        notify("error", "No usable source URLs were found. Try another question.")
        return message, []

    print(
        f"\n📚 Selected {len(search_urls)} "
        "source candidate(s) for "
        f"{research_label}."
    )

    # --------------------------------------------------------
    # 2. READ AND VALIDATE SOURCES
    # --------------------------------------------------------

    reading_start = time.time()
    notify("reading", f"Reading and validating up to {max_sources} sources…")

    verified_sources = []
    seen_urls = set()

    for candidate_url in search_urls:

        if len(verified_sources) >= max_sources:
            break

        if candidate_url in seen_urls:
            continue

        seen_urls.add(candidate_url)

        source_number = len(verified_sources) + 1

        print(
            f"\n📖 Reading source {source_number}/"
            f"{max_sources}..."
        )
        notify(
            "reading",
            f"Reading source {source_number} of up to {max_sources}…",
        )

        try:
            page_content = read_tool.invoke(
                {"url": candidate_url}
            )

            if not isinstance(page_content, str):
                page_content = str(page_content)

            if not is_valid_webpage(page_content):
                print(
                    f"   ⚠️ Source rejected: "
                    "insufficient readable content."
                )
                continue

            # Preserve the extracted page for claim-aware citation review.
            # The main report prompt is limited later using evidence_limit.
            verified_sources.append(
                {
                    "id": source_number,
                    "url": candidate_url,
                    "content": page_content,
                }
            )

            print(f"   ✅ Source {source_number} verified.")

        except Exception as error:
            print(f"   ⚠️ Source reading failed: {error}")

    reading_elapsed = time.time() - reading_start

    print("\n✅ Source reading completed.")
    print(f"⏱️ Reading time: {reading_elapsed:.2f} seconds")

    if not verified_sources:
        message = (
            "Research could not continue because none of the "
            "selected webpages provided enough readable content. "
            "Try a different topic or run the research again."
        )
        notify("error", "The selected webpages did not provide enough readable content.")
        return message, []

    print(
        f"\n✅ {len(verified_sources)} verified source(s) "
        "available."
    )

    verified_urls = [
        source["url"] for source in verified_sources
    ]

    # --------------------------------------------------------
    # 3. PREPARE EVIDENCE
    # --------------------------------------------------------

    evidence_blocks = []

    for source in verified_sources:
        evidence_blocks.append(
            f"===== VERIFIED SOURCE {source['id']} =====\n"
            f"SOURCE ID: {source['id']}\n"
            f"URL: {source['url']}\n\n"
            f"{source['content'][:evidence_limit]}\n"
        )

    evidence_text = "\n\n".join(evidence_blocks)

    # --------------------------------------------------------
    # 4. GENERATE REPORT
    # --------------------------------------------------------

    print("\n🧠 Generating research report...")
    notify("generation", "Sources are ready. Generating the research report with the local AI model…")

    generation_start = time.time()

    if depth == "deep":

        report_instructions = """
Write a detailed research report. Prioritize depth and evidence comparison.

Use these headings:
# Executive Summary
# Key Findings
# Detailed Analysis
# Evidence Comparison
# Limitations
# Conclusion

Requirements:
- Explain the topic in depth without repeating the same point.
- Compare the supplied sources and note disagreements when evidence supports this.
- Separate established facts from uncertain findings.
- Cite factual claims using supplied source IDs, for example (Source 1).
- Use only source IDs that actually exist.
- Do not invent statistics, quotations, or findings.
- Do not create a Sources section; the application displays source URLs separately.
"""

    else:

        report_instructions = """
Write a concise research report of approximately 180–250 words.
Do not exceed the available output budget; prioritize the most useful findings.

Use these headings:
# Executive Summary
# Key Findings
# Analysis
# Conclusion

Requirements:
- Use clear, direct language and avoid repetition.
- Include 3–5 useful findings, not a long general introduction.
- Base factual claims on the supplied webpage evidence.
- Cite factual claims using the supplied source IDs, for example (Source 1).
- Use only source IDs that actually exist.
- If evidence is insufficient, say so briefly.
- Do not invent statistics, quotations, or findings.
- Do not create a Sources section; the application displays source URLs separately.
"""

    final_prompt = f"""
You are ResearchPilot, a research assistant.

Research topic:
{query}

Research mode:
{research_label}

Use the supplied webpage evidence as the primary basis
for the report.

IMPORTANT:
- Search snippets are not a substitute for the supplied
  webpage evidence.
- Do not claim that a source supports something unless
  its supplied text provides relevant evidence.
- If evidence is insufficient, state that limitation.
- Never invent a source, URL, quotation, or statistic.
- Do not claim to have performed experiments or surveys.

REPORT INSTRUCTIONS:
{report_instructions}

VERIFIED WEBPAGE EVIDENCE:
{evidence_text}

Now generate the research report.
"""

    try:
        # Quick Research uses a smaller generation budget; Deep Research
        # keeps a larger budget so its report can remain detailed.
        report_model = deep_llm if depth == "deep" else quick_llm
        response = report_model.invoke(final_prompt)
        report_text = response.content.strip()

    except Exception as error:
        print(f"\n❌ Report generation failed: {error}")
        notify("error", "Report generation failed. Check that Ollama is running and the model is available.")

        return (
            "Research report generation failed. "
            "Check that Ollama is running and the configured "
            f"model '{MODEL_NAME}' is available.\n\n"
            f"Error: {error}",
            verified_urls,
        )

    generation_elapsed = time.time() - generation_start

    print(
        "\n⏱️ Initial AI generation time: "
        f"{generation_elapsed:.2f} seconds"
    )

    if not report_text:
        notify("error", "The AI model returned an empty report. Please try again.")
        return (
            "The AI model returned an empty report. "
            "Please try again.",
            verified_urls,
        )

    # --------------------------------------------------------
    # 5. VALIDATE CITATION NUMBERS
    # --------------------------------------------------------

    invalid_citations = validate_citations(
        report_text,
        len(verified_sources),
    )

    # --------------------------------------------------------
    # 6. SHORT CITATION REVIEW
    # --------------------------------------------------------

    notify("review", "Checking citation numbers and reviewing cited claims against source evidence…")

    # Avoid spending time on a full evidence audit when
    # the report already references nonexistent source IDs.
    # The original report is preserved in either case.
    if invalid_citations:
        print(
            "⚠️ Skipping automated claim review because "
            "invalid citation numbers were found."
        )
        report_text = with_audit_status(report_text, "INCONCLUSIVE")

    else:
        report_text = check_claim_support(
            report_text,
            verified_sources,
        )

        # Check numbering again in case the review appended
        # any text containing source references.
        validate_citations(
            report_text,
            len(verified_sources),
        )

    # --------------------------------------------------------
    # 7. FINISH
    # --------------------------------------------------------

    total_elapsed = time.time() - total_start

    print(
        f"\n⏱️ Total research time: "
        f"{total_elapsed:.2f} seconds"
    )

    print("\n✅ Research completed.")
    notify("complete", "Research is complete. Your report and verified source links are ready.")

    # IMPORTANT:
    # Keep this two-value return for compatibility with app.py.
    return report_text, verified_urls


# ============================================================
# TERMINAL RESEARCH AND HISTORY
# ============================================================

def save_research(report_text: str) -> None:
    """Save a completed report using the existing save tool."""

    try:
        result = save_tool.invoke(
            {"data": report_text}
        )

        print("\n" + str(result))

    except Exception as error:
        print(f"\n❌ Could not save the report: {error}")


def show_history() -> None:
    """Display saved research reports."""

    history_folder = "research_history"

    if not os.path.isdir(history_folder):
        print("\nNo research history found.")
        return

    files = [
        filename
        for filename in os.listdir(history_folder)
        if filename.lower().endswith(".txt")
    ]

    files.sort(reverse=True)

    if not files:
        print("\nNo saved research reports found.")
        return

    print("\n========== RESEARCH HISTORY ==========")

    for index, filename in enumerate(files, start=1):
        print(f"{index}. {filename}")

    choice = input(
        "\nEnter a report number to open it, "
        "or press Enter to return: "
    ).strip()

    if not choice:
        return

    try:
        selected_index = int(choice) - 1

        if selected_index < 0 or selected_index >= len(files):
            print("Invalid selection.")
            return

        filepath = os.path.join(
            history_folder,
            files[selected_index],
        )

        with open(filepath, "r", encoding="utf-8") as file:
            print("\n" + "=" * 55)
            print(file.read())
            print("=" * 55)

    except ValueError:
        print("Please enter a valid number.")

    except OSError as error:
        print(f"Could not open the report: {error}")


def start_new_research() -> None:
    """Collect a topic and run terminal-based research."""

    query = input(
        "\nEnter your research topic: "
    ).strip()

    if not query:
        print("Research topic cannot be empty.")
        return

    print("\nChoose research mode:")
    print("1. Quick Research")
    print("2. Deep Research")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        depth = "deep"
    else:
        depth = "quick"

    report_text, verified_urls = research_topic(
        query,
        depth,
    )

    print("\n" + "=" * 55)
    print("RESEARCH REPORT")
    print("=" * 55)
    print(report_text)

    if verified_urls:
        print("\n## Sources")

        for index, url in enumerate(
            verified_urls,
            start=1,
        ):
            print(f"- Source {index}: {url}")

    save_choice = input(
        "\nSave this report? (y/n): "
    ).strip().lower()

    if save_choice == "y":
        save_research(report_text)


def main() -> None:
    """Terminal menu."""

    while True:
        print("\n" + "=" * 45)
        print("          RESEARCHPILOT")
        print("=" * 45)
        print("1. Start new research")
        print("2. View research history")
        print("3. Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            start_new_research()

        elif choice == "2":
            show_history()

        elif choice == "3":
            print("\nExiting ResearchPilot. Goodbye!")
            break

        else:
            print("Invalid choice. Enter 1, 2, or 3.")


if __name__ == "__main__":
    main()