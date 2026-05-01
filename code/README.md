# Triage Agent

A terminal-based RAG support triage agent for the HackerRank Orchestrate Hackathon (May 2026).

## Architecture

- **Retrieval**: TF-IDF + Cosine Similarity over the local `data/` Markdown corpus (774 docs).
- **LLM**: Google Gemini via `google-genai` SDK with structured JSON output.
- **Safety**: Explicit escalation rules for fraud, security, financial disputes, and malicious input.

## Installation

1. Python 3.9+ required.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` → `.env` in the repo root and add your key:
   ```
   GEMINI_API_KEY=your_key_here
   ```

## Running the Agent

### 🎨 Interactive Terminal UI (recommended)
```bash
python code/triage_ui.py
```
Choose a mode when prompted:
- **Mode 1** — Demo: runs 6 curated tickets covering FAQ, billing, fraud, security, malicious input, and account management.
- **Mode 2** — Interactive: enter your own ticket in real time.
- **Mode 3** — Full run: processes all `support_tickets/support_tickets.csv` and writes `support_tickets/output.csv`.

### ⚡ Batch-only runner
```bash
python code/main.py
```
Processes all 29 tickets silently and writes `output.csv`.
