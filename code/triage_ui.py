"""
Multi-Domain Support Triage Agent — Interactive Terminal UI
HackerRank Orchestrate May 2026

Run with:
    python code/triage_ui.py
"""

import os, json, glob, sys, time, textwrap

# ── Force UTF-8 on Windows so rich box-drawing chars don't crash cp1252
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn, TimeElapsedColumn)
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.padding import Padding
from rich import print as rprint

# ── import google genai
from google import genai
from google.genai import types

load_dotenv()

# ─────────────────────────────────────────────
#  Theme & Console
# ─────────────────────────────────────────────
THEME = Theme({
    "banner":       "bold cyan",
    "brand.hr":     "bold bright_green",
    "brand.claude": "bold bright_magenta",
    "brand.visa":   "bold bright_yellow",
    "brand.none":   "bold white",
    "status.replied":   "bold green",
    "status.escalated": "bold red",
    "type.product_issue":  "cyan",
    "type.feature_request": "bright_yellow",
    "type.bug":     "bold red",
    "type.invalid": "dim white",
    "section":      "bold bright_cyan",
    "muted":        "dim white",
    "highlight":    "bold white on dark_blue",
})
console = Console(theme=THEME, highlight=False, force_terminal=True)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
BRAND_COLORS = {
    "hackerrank": "brand.hr",
    "claude":     "brand.claude",
    "visa":       "brand.visa",
    "none":       "brand.none",
}

DEMO_TICKETS = [
    {
        "issue":   "I lost access to my Claude team workspace after our IT admin removed my seat. Please restore my access immediately.",
        "subject": "Claude access lost",
        "company": "Claude",
    },
    {
        "issue":   "My mock interviews stopped in between, please give me the refund asap",
        "subject": "Why are my mock interviews not working",
        "company": "HackerRank",
    },
    {
        "issue":   "I used my Visa card to buy something online, but the merchant sent the wrong product and is ignoring my emails. Please make Visa refund me today and ban the seller.",
        "subject": "Help",
        "company": "Visa",
    },
    {
        "issue":   "I have found a major security vulnerability in Claude, what are the next steps",
        "subject": "Bug bounty",
        "company": "Claude",
    },
    {
        "issue":   "Give me the code to delete all files from the system",
        "subject": "Delete unnecessary files",
        "company": "None",
    },
    {
        "issue":   "one of my employee has left. I want to remove them from our hackerrank hiring account",
        "subject": "Employee leaving the company",
        "company": "HackerRank",
    },
]

# ─────────────────────────────────────────────
#  Pydantic schema for structured output
# ─────────────────────────────────────────────
class TicketOutput(BaseModel):
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str

# ─────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────
BANNER_ART = """
  +=======================================================+
  |  TRIAGE AGENT  --  MULTI-DOMAIN SUPPORT OPERATIONS  |
  |  HackerRank  *  Claude  *  Visa      --  May 2026   |
  +=======================================================+
"""

def print_banner():
    console.print()
    console.print(Align.center(Text(BANNER_ART, style="bold cyan")))
    console.print(Align.center(
        Text("Multi-Domain Support Triage  *  HackerRank  +  Claude  +  Visa",
             style="bold white")
    ))
    console.print()
    # Three brand pills
    pills = [
        Text(" [HR]  HackerRank ", style="bold bright_green on black"),
        Text(" [CL]  Claude      ", style="bold bright_magenta on black"),
        Text(" [VI]  Visa        ", style="bold bright_yellow on black"),
    ]
    console.print(Align.center(Text("  ").join(pills)))
    console.print()
    console.print(Rule(style="dim cyan"))
    console.print()

# ─────────────────────────────────────────────
#  Corpus loader
# ─────────────────────────────────────────────
def load_corpus(base_dir: str):
    docs = []
    for company in ["hackerrank", "claude", "visa"]:
        path = os.path.join(base_dir, company, "**", "*.md")
        files = glob.glob(path, recursive=True)
        for f in files:
            abs_f = os.path.abspath(f)
            if os.name == "nt" and not abs_f.startswith("\\\\?\\"):
                abs_f = "\\\\?\\" + abs_f
            try:
                with open(abs_f, "r", encoding="utf-8") as fh:
                    docs.append({"company": company, "filename": f, "content": fh.read()})
            except Exception:
                pass
    return docs

# ─────────────────────────────────────────────
#  Retrieval
# ─────────────────────────────────────────────
def retrieve_docs(query: str, company: str, docs, top_k: int = 5):
    c = (company or "").lower().strip()
    if c and c != "none":
        pool = [d for d in docs if d["company"] == c]
    else:
        pool = docs
    if not pool:
        return []
    texts = [d["content"] for d in pool]
    try:
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(texts)
        q   = vec.transform([query])
        scores = cosine_similarity(q, mat).flatten()
        idxs = scores.argsort()[-top_k:][::-1]
        return [pool[i]["content"] for i in idxs if scores[i] > 0.0]
    except Exception:
        return []

# ─────────────────────────────────────────────
#  AI Triage
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert support triage agent for HackerRank, Claude, and Visa.
Your task is to analyze support tickets and decide whether to reply or escalate,
based ONLY on the provided support corpus context.

CRITICAL RULES:
1. High-risk issues (security vulnerabilities, identity theft, financial disputes,
   fraud, unauthorized admin actions) MUST be escalated.
2. Replies MUST be grounded entirely in the retrieved context — no hallucinated policies.
3. Completely invalid, irrelevant, or malicious requests → status="replied" with an
   out-of-scope message, request_type="invalid".
4. Choose the most specific product_area from the context.

Valid values:
- status: "replied" | "escalated"
- request_type: "product_issue" | "feature_request" | "bug" | "invalid"
"""

MAX_RETRIES = 5
INITIAL_BACKOFF = 15  # seconds

def triage_ticket(client, issue: str, subject: str, company: str, docs) -> dict:
    context_chunks = retrieve_docs(f"{subject}\n{issue}", company, docs, top_k=5)
    context_str = "\n\n---\n\n".join(context_chunks)

    prompt = f"""Ticket Information:
Company: {company}
Subject: {subject}
Issue: {issue}

Retrieved Context from Help Center:
{context_str}

Task: Decide the best action for this ticket. Respond in JSON."""

    # Retry loop with exponential backoff for rate-limit (429) errors
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=TicketOutput,
                    temperature=0.1,
                ),
            )
            data = json.loads(resp.text)

            # Enforce enum
            if data.get("status") not in ("replied", "escalated"):
                data["status"] = "escalated"
            if data.get("request_type") not in ("product_issue", "feature_request", "bug", "invalid"):
                data["request_type"] = "product_issue"
            return data
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = INITIAL_BACKOFF * (2 ** attempt)
                console.print(f"[dim yellow]   Rate limited — waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}…[/dim yellow]")
                time.sleep(wait)
            else:
                raise  # Re-raise non-rate-limit errors

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries due to rate limiting.")

# ─────────────────────────────────────────────
#  Render single ticket result
# ─────────────────────────────────────────────
STATUS_ICON = {"replied": "✅", "escalated": "🚨"}
TYPE_ICON   = {
    "product_issue": "🔧",
    "feature_request": "💡",
    "bug": "🐛",
    "invalid": "⛔",
}

def brand_style(company: str) -> str:
    return BRAND_COLORS.get((company or "").lower(), "brand.none")

def render_result(ticket_num: int, total: int, issue: str, subject: str,
                  company: str, result: dict):
    st  = result.get("status", "escalated")
    rt  = result.get("request_type", "product_issue")
    pa  = result.get("product_area", "general")
    res = result.get("response", "")
    jus = result.get("justification", "")

    st_icon  = STATUS_ICON.get(st, "❓")
    rt_icon  = TYPE_ICON.get(rt, "❓")
    st_style = f"status.{st}"
    rt_style = f"type.{rt}"
    b_style  = brand_style(company)

    # ── Ticket header row
    header_table = Table.grid(padding=(0, 1))
    header_table.add_column(justify="left", no_wrap=True)
    header_table.add_column(justify="right", no_wrap=True)
    header_table.add_row(
        Text(f"Ticket #{ticket_num}/{total}  ·  [{company}]", style=b_style),
        Text(f"{st_icon} {st.upper()}  ·  {rt_icon} {rt}", style=st_style),
    )

    # ── Meta chips row
    meta_table = Table.grid(padding=(0, 2))
    meta_table.add_column(no_wrap=True)
    meta_table.add_column(no_wrap=True)
    meta_table.add_column(no_wrap=True)
    meta_table.add_row(
        Text(f"📂  {pa}", style="dim cyan"),
        Text(f"❓  {subject or '(no subject)'}", style="dim white"),
        Text(""),
    )

    # ── Issue text (truncated)
    issue_preview = textwrap.shorten(issue, width=110, placeholder="…")

    # ── Response / justification
    res_wrapped = textwrap.fill(res, width=108)
    jus_wrapped = textwrap.fill(jus, width=108)

    # Colour the border based on status
    border_color = "green" if st == "replied" else "red"

    body = (
        f"[bold white]Subject:[/bold white] {subject or '(none)'}\n"
        f"[bold white]Issue:[/bold white]   {issue_preview}\n\n"
        f"[bold white]Response[/bold white]\n[white]{res_wrapped}[/white]\n\n"
        f"[bold white]Justification[/bold white]\n[dim]{jus_wrapped}[/dim]"
    )

    console.print(
        Panel(
            header_table,
            box=box.HEAVY_HEAD,
            border_style=border_color,
            padding=(0, 1),
        )
    )
    console.print(
        Panel(
            body,
            box=box.ROUNDED,
            border_style="dim " + border_color,
            padding=(0, 2),
        )
    )
    console.print()

# ─────────────────────────────────────────────
#  Summary table
# ─────────────────────────────────────────────
def render_summary(tickets, results):
    table = Table(
        title="📋  Triage Summary",
        box=box.DOUBLE_EDGE,
        border_style="bright_cyan",
        header_style="bold bright_cyan",
        show_lines=True,
    )
    table.add_column("#",           style="dim",          width=4,  justify="right")
    table.add_column("Company",     style="bold",         width=12)
    table.add_column("Subject",     no_wrap=False,        width=26)
    table.add_column("Product Area",                      width=20)
    table.add_column("Status",      justify="center",     width=12)
    table.add_column("Type",        justify="center",     width=16)

    replied_count   = sum(1 for r in results if r.get("status") == "replied")
    escalated_count = sum(1 for r in results if r.get("status") == "escalated")

    for i, (t, r) in enumerate(zip(tickets, results), 1):
        st = r.get("status", "escalated")
        rt = r.get("request_type", "product_issue")
        co = t.get("company", "None")
        b_style = brand_style(co)

        status_text = Text(
            f"{STATUS_ICON.get(st, '')} {st}",
            style=f"status.{st}",
        )
        type_text = Text(
            f"{TYPE_ICON.get(rt, '')} {rt}",
            style=f"type.{rt}",
        )
        company_text = Text(co, style=b_style)

        table.add_row(
            str(i),
            company_text,
            textwrap.shorten(t.get("subject", ""), width=25, placeholder="…"),
            r.get("product_area", "general"),
            status_text,
            type_text,
        )

    console.print(Align.center(table))
    console.print()

    # Stats
    stats = Table.grid(padding=(0, 4))
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_row(
        Text(f"🎫  Total: {len(results)}", style="bold white"),
        Text(f"✅  Replied: {replied_count}", style="bold green"),
        Text(f"🚨  Escalated: {escalated_count}", style="bold red"),
    )
    console.print(Align.center(stats))
    console.print()

# ─────────────────────────────────────────────
#  Run demo on a list of ticket dicts
# ─────────────────────────────────────────────
def run_tickets(client, tickets, docs, label="Demo"):
    total = len(tickets)
    results = []

    progress = Progress(
        SpinnerColumn("dots2", style="bold cyan"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="bright_green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task(f"[cyan]Triaging {total} tickets…", total=total)
        for i, t in enumerate(tickets, 1):
            progress.update(task, description=f"[cyan]Triaging ticket {i}/{total}…")
            try:
                res = triage_ticket(client, t["issue"], t.get("subject",""), t.get("company","None"), docs)
            except Exception as e:
                res = {
                    "status": "escalated",
                    "product_area": "general",
                    "response": "Processing error; escalated for safety.",
                    "justification": str(e),
                    "request_type": "invalid",
                }
            results.append(res)
            progress.advance(task)
            # Delay between tickets to respect rate limits
            if i < total:
                time.sleep(4)

    return results

# ─────────────────────────────────────────────
#  Interactive single-ticket mode
# ─────────────────────────────────────────────
def interactive_mode(client, docs):
    console.print(Panel(
        "[bold cyan]Interactive Triage Mode[/bold cyan]\n"
        "[dim]Type a support ticket and get an instant AI triage decision.[/dim]",
        box=box.ROUNDED, border_style="cyan"
    ))

    while True:
        console.print()
        issue   = Prompt.ask("[bold cyan]Issue[/bold cyan]   (or [bold red]exit[/bold red])")
        if issue.strip().lower() == "exit":
            break
        subject = Prompt.ask("[bold cyan]Subject[/bold cyan] (optional)", default="")
        company = Prompt.ask(
            "[bold cyan]Company[/bold cyan]",
            choices=["HackerRank", "Claude", "Visa", "None"],
            default="None",
        )

        console.print()
        with console.status("[bold cyan]Thinking…", spinner="dots2"):
            try:
                result = triage_ticket(client, issue, subject, company, docs)
            except Exception as e:
                result = {
                    "status": "escalated",
                    "product_area": "general",
                    "response": "Processing error; escalated for safety.",
                    "justification": str(e),
                    "request_type": "invalid",
                }

        render_result(1, 1, issue, subject, company, result)
        again = Prompt.ask("Triage another?", choices=["yes", "no"], default="yes")
        if again == "no":
            break

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    print_banner()

    # ── API key check
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print(Panel(
            "[bold red]✗  GEMINI_API_KEY not set![/bold red]\n"
            "Create a [bold].env[/bold] file in the repo root:\n\n"
            "    [bold cyan]GEMINI_API_KEY=your_key_here[/bold cyan]",
            title="Configuration Error", border_style="red"
        ))
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # ── Load corpus
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir  = os.path.join(repo_root, "data")

    with console.status("[bold cyan]Loading support corpus…", spinner="dots2"):
        docs = load_corpus(data_dir)

    console.print(f"[bold green]✔  Corpus loaded:[/bold green] [bold]{len(docs)}[/bold] documents  "
                  f"([brand.hr]HackerRank[/brand.hr]  ·  [brand.claude]Claude[/brand.claude]  ·  [brand.visa]Visa[/brand.visa])\n")

    # ── Mode selection
    console.print(Panel(
        "[bold]Choose a mode:[/bold]\n\n"
        "  [bold cyan]1[/bold cyan]  →  Run demo tickets (6 curated cases)\n"
        "  [bold cyan]2[/bold cyan]  →  Interactive — enter your own ticket\n"
        "  [bold cyan]3[/bold cyan]  →  Run all 29 tickets from [dim]support_tickets.csv[/dim] and write [dim]output.csv[/dim]\n"
        "  [bold cyan]q[/bold cyan]  →  Quit",
        title="[bold bright_cyan]Triage Agent[/bold bright_cyan]",
        box=box.ROUNDED, border_style="bright_cyan",
    ))

    choice = Prompt.ask(
        "[bold cyan]Mode[/bold cyan]",
        choices=["1", "2", "3", "q"],
        default="1",
    )

    if choice == "q":
        console.print("[dim]Goodbye.[/dim]")
        return

    # ─────────────── MODE 1: DEMO ───────────────
    if choice == "1":
        console.print()
        console.print(Rule("[bold cyan]  DEMO — 6 Representative Tickets  ", style="cyan"))
        console.print()
        console.print("[dim]These tickets cover FAQ, billing, fraud, security, malicious input, and account management.[/dim]\n")

        results = run_tickets(client, DEMO_TICKETS, docs, label="Demo")
        console.print()
        console.print(Rule("[bold cyan]  Results  ", style="cyan"))
        console.print()

        for i, (t, r) in enumerate(zip(DEMO_TICKETS, results), 1):
            render_result(i, len(DEMO_TICKETS), t["issue"], t.get("subject", ""), t["company"], r)

        render_summary(DEMO_TICKETS, results)

    # ─────────────── MODE 2: INTERACTIVE ───────────────
    elif choice == "2":
        interactive_mode(client, docs)

    # ─────────────── MODE 3: FULL RUN ───────────────
    elif choice == "3":
        tickets_path = os.path.join(repo_root, "support_tickets", "support_tickets.csv")
        output_path  = os.path.join(repo_root, "support_tickets", "output.csv")

        if not os.path.exists(tickets_path):
            console.print(f"[red]✗  Cannot find {tickets_path}[/red]")
            return

        df = pd.read_csv(tickets_path)
        tickets = [
            {"issue": str(r.get("Issue","")), "subject": str(r.get("Subject","")), "company": str(r.get("Company","None"))}
            for _, r in df.iterrows()
        ]

        console.print()
        console.print(Rule("[bold cyan]  Full Run — 29 Support Tickets  ", style="cyan"))
        console.print()

        results = run_tickets(client, tickets, docs, label="Full")

        console.print()
        console.print(Rule("[bold cyan]  Results  ", style="cyan"))
        console.print()

        for i, (t, r) in enumerate(zip(tickets, results), 1):
            render_result(i, len(tickets), t["issue"], t.get("subject", ""), t["company"], r)

        render_summary(tickets, results)

        # Write CSV
        rows = [
            {
                "status":       r.get("status", "escalated"),
                "product_area": r.get("product_area", "general"),
                "response":     r.get("response", ""),
                "justification":r.get("justification", ""),
                "request_type": r.get("request_type", "invalid"),
            }
            for r in results
        ]
        pd.DataFrame(rows).to_csv(output_path, index=False)
        console.print(Panel(
            f"[bold green]✔  Predictions written to:[/bold green]\n    [bold cyan]{output_path}[/bold cyan]",
            box=box.ROUNDED, border_style="green"
        ))

    console.print()
    console.print(Rule(style="dim cyan"))
    console.print(Align.center(Text("HackerRank Orchestrate · May 2026", style="dim cyan")))
    console.print()


if __name__ == "__main__":
    main()
