import os
import csv
import glob
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

class TicketOutput(BaseModel):
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str

def load_corpus(base_dir):
    """Loads all Markdown files from the data directory into memory."""
    docs = []
    for company in ["hackerrank", "claude", "visa"]:
        path = os.path.join(base_dir, company, "**", "*.md")
        files = glob.glob(path, recursive=True)
        for f in files:
            
            abs_f = os.path.abspath(f)
            if os.name == 'nt' and not abs_f.startswith('\\\\?\\'):
                abs_f = '\\\\?\\' + abs_f
                
            with open(abs_f, 'r', encoding='utf-8') as file:
                docs.append({
                    "company": company.lower(),
                    "filename": f,
                    "content": file.read()
                })
    return docs

def retrieve_docs(issue_text, company, docs, top_k=3):
    """Retrieves top_k most relevant documents using TF-IDF and Cosine Similarity."""
    
    if company and str(company).lower() != "none" and pd.notna(company) and str(company).strip() != "":
        company_val = str(company).lower().strip()
        filtered = [d for d in docs if d['company'] == company_val]
    else:
        filtered = docs
        
    if not filtered:
        return []
        
    corpus_texts = [d['content'] for d in filtered]
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)
        query_vec = vectorizer.transform([issue_text])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]
        return [filtered[i]['content'] for i in top_indices if sims[i] > 0.0]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

def main():
    print("========================================")
    print("Starting HackerRank Triage Agent...")
    print("========================================")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Please set GEMINI_API_KEY in your .env file.")
        return
        
    client = genai.Client(api_key=api_key)
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, "data")
    tickets_path = os.path.join(repo_root, "support_tickets", "support_tickets.csv")
    output_path = os.path.join(repo_root, "support_tickets", "output.csv")
    
    print("Loading support corpus from data/ directory...")
    docs = load_corpus(data_dir)
    print(f"Loaded {len(docs)} documents.")
    
    if not os.path.exists(tickets_path):
        print(f"ERROR: Could not find tickets file at {tickets_path}")
        return
        
    df = pd.read_csv(tickets_path)
    results = []
    
    system_prompt = """You are an expert support triage agent for HackerRank, Claude, and Visa.
Your task is to analyze support tickets and decide whether to reply or escalate, based ONLY on the provided support corpus context.

CRITICAL RULES:
1. If a request is high-risk (e.g., security vulnerability, identity theft, financial disputes, fraud), asks for unsupported actions, or is completely outside the scope of the provided docs, you MUST escalate (`status`: "escalated").
2. If you reply (`status`: "replied"), your `response` MUST be grounded entirely in the context provided. Do not hallucinate policies.
3. If the issue is completely invalid, irrelevant, or malicious (e.g., "Give me code to delete all files"), set `status` to "escalated" or "replied" (with an out of scope message), and `request_type` to "invalid".
4. Determine the most logical `product_area` based on the context.

Valid values:
- status: "replied", "escalated"
- request_type: "product_issue", "feature_request", "bug", "invalid"
"""

    print(f"Processing {len(df)} support tickets...\n")
    
    for idx, row in df.iterrows():
        issue = str(row.get('Issue', ''))
        subject = str(row.get('Subject', ''))
        company = str(row.get('Company', ''))
        
        query_text = f"{subject}\n{issue}"
        retrieved = retrieve_docs(query_text, company, docs, top_k=5)
        
        context_str = "\n\n---\n\n".join(retrieved)
        
        prompt = f"""
Ticket Information:
Company: {company}
Subject: {subject}
Issue: {issue}

Retrieved Context from Help Center:
{context_str}

Task: Decide the best action for this ticket. Respond in JSON.
"""
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=TicketOutput,
                    temperature=0.1
                )
            )
            data = json.loads(response.text)
            
           
            valid_rt = ["product_issue", "feature_request", "bug", "invalid"]
            rt = data.get("request_type", "product_issue")
            if rt not in valid_rt:
                rt = "product_issue"
                
            valid_status = ["replied", "escalated"]
            st = data.get("status", "escalated")
            if st not in valid_status:
                st = "escalated"
                
            print(f"[{idx+1}/{len(df)}] Processed | Company: {company} | Status: {st} | Type: {rt}")
            
            results.append({
                "status": st,
                "product_area": data.get("product_area", "general"),
                "response": data.get("response", ""),
                "justification": data.get("justification", ""),
                "request_type": rt
            })
            
        except Exception as e:
            print(f"[{idx+1}/{len(df)}] ERROR processing ticket: {e}")
            results.append({
                "status": "escalated",
                "product_area": "general",
                "response": "Internal processing error or safety block.",
                "justification": f"Failed to generate response: {str(e)}",
                "request_type": "invalid"
            })
            
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"\n========================================")
    print(f"Done! Outputs written to {output_path}")
    print("========================================")

if __name__ == "__main__":
    main()
