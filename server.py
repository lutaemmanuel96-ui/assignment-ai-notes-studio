import base64
import io
import json
import math
import os
import re
import sqlite3
from collections import Counter
from typing import Optional

import docx
import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pypdf import PdfReader

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
    HAS_SYMPY = True
except Exception:
    HAS_SYMPY = False

app = FastAPI()
DB_PATH = "notes.db"
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"]

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "what", "when", "where", "which", "how", "why", "who", "whom", "into", "onto",
    "using", "use", "find", "calculate", "compute", "solve", "explain", "given",
    "let", "show", "prove", "state", "list", "give", "answer", "question"
}

IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def resolve_api_key(api_key: Optional[str]) -> str:
    return (api_key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            course_unit TEXT DEFAULT 'General',
            page INTEGER,
            content TEXT
        )
        """
    )
    cols = [r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()]
    if "course_unit" not in cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN course_unit TEXT DEFAULT 'General'")
    conn.commit()
    conn.close()


init_db()


def normalize_unit(unit: Optional[str]) -> str:
    cleaned = re.sub(r"\s+", " ", (unit or "")).strip()
    return cleaned if cleaned else "General"


def chunk_text(text: str, size: int = 1200, overlap: int = 200):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    step = max(size - overlap, 1)
    for i in range(0, len(text), step):
        piece = text[i:i + size]
        if len(piece) > 40:
            chunks.append(piece)
    return chunks


def ocr_pdf_with_gemini(pdf_bytes: bytes, api_key: str) -> str:
    effective_key = resolve_api_key(api_key)
    if not effective_key:
        return ""
    b64_pdf = base64.b64encode(pdf_bytes).decode("ascii")
    prompt = (
        "Extract all readable text, headings, formulas, and tables from this scanned PDF "
        "in page order. Format clearly with [Page N] markers. Do not invent missing text."
    )
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": "application/pdf", "data": b64_pdf}},
    ]
    for m in GEMINI_MODELS:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
            f"?key={effective_key}"
        )
        try:
            res = requests.post(
                url,
                json={"contents": [{"parts": parts}]},
                timeout=60,
            )
            data = res.json()
            if "candidates" in data and data["candidates"]:
                c_parts = data["candidates"][0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in c_parts if "text" in p).strip()
                if text:
                    return text
            if res.status_code != 404:
                break
        except Exception:
            break
    return ""


def extract_pages_from_bytes(name: str, data: bytes, api_key: str = ""):
    lower = name.lower()
    pages = []
    effective_key = resolve_api_key(api_key)
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        non_empty = 0
        for idx, page in enumerate(reader.pages, 1):
            txt = page.extract_text() or ""
            if txt.strip():
                non_empty += 1
            pages.append((idx, txt))

        if non_empty == 0 and effective_key:
            ocr_text = ocr_pdf_with_gemini(data, effective_key)
            if ocr_text:
                split_pages = re.split(r"\[Page\s+(\d+)\]", ocr_text)
                parsed = []
                for i in range(1, len(split_pages) - 1, 2):
                    p_num = int(split_pages[i])
                    p_txt = split_pages[i + 1].strip()
                    if p_txt:
                        parsed.append((p_num, p_txt))
                pages = parsed if parsed else [(1, ocr_text)]
    elif lower.endswith(".docx"):
        doc = docx.Document(io.BytesIO(data))
        full = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        pages.append((1, full))
    else:
        pages.append((1, data.decode("utf-8", errors="ignore")))
    return pages


def tokenize(text: str):
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def score_chunks(query: str, rows, top_k: int = 8):
    q_tokens = Counter(tokenize(query))
    if not q_tokens:
        q_tokens = Counter(re.findall(r"[a-z0-9_]+", query.lower()))
    if not q_tokens:
        return []

    doc_freq = Counter()
    tokenized_rows = []
    for _, fname, unit, page, content in rows:
        c_tokens = Counter(tokenize(content))
        tokenized_rows.append((fname, unit, page, content, c_tokens))
        for t in c_tokens:
            doc_freq[t] += 1

    n_docs = max(len(tokenized_rows), 1)
    q_weights = {
        t: cnt * math.log(1 + n_docs / (1 + doc_freq[t]))
        for t, cnt in q_tokens.items()
    }
    norm_q = math.sqrt(sum(w * w for w in q_weights.values())) + 1e-9

    scored = []
    for fname, unit, page, content, c_tokens in tokenized_rows:
        dot = sum(q_weights[t] * c_tokens[t] for t in q_weights if c_tokens[t] > 0)
        if dot > 0:
            norm_c = math.sqrt(sum(v * v for v in c_tokens.values())) + 1e-9
            sim = dot / (norm_q * norm_c)
            scored.append((sim, fname, unit, page, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def call_gemini(parts, api_key: str, timeout: int = 45):
    effective_key = resolve_api_key(api_key)
    if not effective_key:
        return None, "API Error: No Gemini API key provided or set in GEMINI_API_KEY."
    answer = None
    last_err = ""
    for m in GEMINI_MODELS:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
            f"?key={effective_key}"
        )
        try:
            res = requests.post(
                url,
                json={"contents": [{"parts": parts}]},
                timeout=timeout,
            )
            data = res.json()
            if "candidates" in data and data["candidates"]:
                c_parts = data["candidates"][0].get("content", {}).get("parts", [])
                answer = "".join(p.get("text", "") for p in c_parts if "text" in p).strip()
                if answer:
                    return answer, None
            last_err = str(data.get("error", {}).get("message", data))
            if res.status_code != 404:
                break
        except Exception as e:
            last_err = f"Request error: {e}"
            break
    return None, f"API Error: {last_err}"


def fetch_rows_by_unit(course_unit: Optional[str]):
    unit_clean = (course_unit or "").strip()
    conn = sqlite3.connect(DB_PATH)
    if unit_clean and unit_clean.lower() != "all":
        rows = conn.execute(
            "SELECT id, filename, course_unit, page, content FROM chunks WHERE LOWER(course_unit) = LOWER(?)",
            (unit_clean,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, filename, course_unit, page, content FROM chunks"
        ).fetchall()
    conn.close()
    return rows


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/files")
def list_files(course_unit: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    unit_clean = (course_unit or "").strip()
    if unit_clean and unit_clean.lower() != "all":
        file_rows = conn.execute(
            "SELECT filename, course_unit, COUNT(*) FROM chunks WHERE LOWER(course_unit) = LOWER(?) GROUP BY filename, course_unit ORDER BY filename",
            (unit_clean,),
        ).fetchall()
    else:
        file_rows = conn.execute(
            "SELECT filename, course_unit, COUNT(*) FROM chunks GROUP BY filename, course_unit ORDER BY filename"
        ).fetchall()

    units = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT course_unit FROM chunks ORDER BY course_unit"
        ).fetchall()
    ]
    conn.close()

    return {
        "files": [
            {"filename": r[0], "course_unit": r[1] or "General", "chunks": r[2]}
            for r in file_rows
        ],
        "units": units,
    }


@app.delete("/files/{filename}")
def delete_file(filename: str, course_unit: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    if course_unit and course_unit.strip() and course_unit.strip().lower() != "all":
        conn.execute(
            "DELETE FROM chunks WHERE filename = ? AND LOWER(course_unit) = LOWER(?)",
            (filename, course_unit.strip()),
        )
    else:
        conn.execute("DELETE FROM chunks WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    course_unit: str = Form("General"),
    api_key: str = Form(""),
):
    effective_key = resolve_api_key(api_key)
    data = await file.read()
    name = file.filename or "uploaded_file"
    unit_name = normalize_unit(course_unit)
    pages = extract_pages_from_bytes(name, data, api_key=effective_key)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "DELETE FROM chunks WHERE filename = ? AND LOWER(course_unit) = LOWER(?)",
        (name, unit_name),
    )
    count = 0
    for page_num, text in pages:
        for ch in chunk_text(text):
            conn.execute(
                "INSERT INTO chunks (filename, course_unit, page, content) VALUES (?, ?, ?, ?)",
                (name, unit_name, page_num, ch),
            )
            count += 1
    conn.commit()
    conn.close()
    note = ""
    if count == 0 and name.lower().endswith(".pdf") and not effective_key:
        note = " (Scanned PDF detected: add a Gemini API key or set GEMINI_API_KEY to enable OCR)"
    return {"filename": name, "course_unit": unit_name, "chunks": count, "note": note}


@app.post("/ask")
async def ask_question(
    question: str = Form(""),
    api_key: str = Form(""),
    course_unit: str = Form("All"),
    history: str = Form("[]"),
    style: str = Form("standard"),
    attachment: Optional[UploadFile] = File(None),
):
    effective_key = resolve_api_key(api_key)
    rows = fetch_rows_by_unit(course_unit)

    att_name = ""
    att_is_image = False
    att_mime = ""
    att_b64 = ""
    att_text = ""

    if attachment and attachment.filename:
        att_name = attachment.filename
        att_data = await attachment.read()
        lower = att_name.lower()
        ext = next((e for e in IMAGE_MIME_TYPES if lower.endswith(e)), "")
        if ext:
            att_is_image = True
            att_mime = attachment.content_type or IMAGE_MIME_TYPES[ext]
            att_b64 = base64.b64encode(att_data).decode("ascii")
        else:
            pages = extract_pages_from_bytes(att_name, att_data, api_key=effective_key)
            att_text = "\n\n".join(
                f"[{att_name} - Page {p}]\n{re.sub(r'\\s+', ' ', t).strip()}"
                for p, t in pages
                if (t or "").strip()
            )[:6000]

    effective_query = (question or "").strip()
    if not effective_query and att_text:
        effective_query = att_text[:800]
    elif not effective_query and att_is_image:
        effective_query = "Solve the assignment question shown in the attached photo step by step."

    if not effective_query and not att_is_image:
        return JSONResponse({
            "answer": "Type a question or attach a photo / document first.",
            "sources": [],
        })

    top = score_chunks(effective_query, rows, top_k=8) if rows else []
    sources = [
        {"filename": f, "course_unit": u, "page": p, "score": round(s, 3), "snippet": c}
        for s, f, u, p, c in top
    ]

    context_block = (
        "\n\n".join([f"[{f} | {u} - Page {p}]\n{c}" for _, f, u, p, c in top])
        if top
        else "(No matching indexed notes found for this unit filter)"
    )

    if effective_key:
        recent_turns = []
        try:
            parsed_hist = json.loads(history or "[]")
            if isinstance(parsed_hist, list):
                for turn in parsed_hist[-6:]:
                    role = "Student" if turn.get("role") == "user" else "Tutor"
                    text = re.sub(r"\s+", " ", str(turn.get("text", ""))).strip()
                    if text:
                        recent_turns.append(f"{role}: {text[:800]}")
        except Exception:
            recent_turns = []

        history_block = (
            "\n".join(recent_turns) if recent_turns else "(No prior turns in this session)"
        )

        style_instruction = (
            "Write in clear, natural, everyday academic English—vary sentence lengths, avoid "
            "formulaic AI transitions or stiff filler phrases, and sound like a thoughtful student "
            "or clear lecturer explaining the points plainly while keeping accuracy and citations. "
            if style == "normal"
            else ""
        )

        prompt_parts = [
            "You are a thorough university assignment tutor. Use the recent conversation "
            "history for context on follow-up questions, prioritize the provided course notes "
            "below, cite the filename, course unit, and page number whenever using them, "
            "format math using $...$ for inline and $$...$$ for display equations, and show "
            f"clear step-by-step reasoning. {style_instruction}\n\n"
            f"RECENT CONVERSATION:\n{history_block}\n\n"
            f"COURSE NOTES:\n{context_block}\n\n"
        ]
        if att_text:
            prompt_parts.append(f"ATTACHED QUESTION DOCUMENT ({att_name}):\n{att_text}\n\n")
        prompt_parts.append(
            f"CURRENT QUESTION / INSTRUCTIONS:\n{question.strip() or 'Answer the attached question clearly and step by step.'}"
        )
        prompt = "".join(prompt_parts)

        req_parts = [{"text": prompt}]
        if att_is_image and att_b64:
            req_parts.append({"inline_data": {"mime_type": att_mime, "data": att_b64}})

        answer, err = call_gemini(req_parts, api_key=effective_key, timeout=45)
        if not answer:
            answer = err
    else:
        prefix = (
            "Photos require a Gemini API key in the top bar (or GEMINI_API_KEY on the server) so the vision model can read the image. "
            if att_is_image
            else "Offline Retrieval Mode (Add a Gemini API key in the top bar or set GEMINI_API_KEY on the server for full step-by-step AI explanations): "
        )
        answer = (
            prefix
            + "Here are the closest matching passages from your indexed notes:\n\n"
            + (
                "\n\n---\n\n".join([f"**{f} [{u}] (Page {p})**: {c}" for _, f, u, p, c in top[:3]])
                if top
                else "No matching notes found yet."
            )
        )

    return JSONResponse({"answer": answer, "sources": sources})


@app.post("/humanize")
async def humanize_text(text: str = Form(...), api_key: str = Form("")):
    effective_key = resolve_api_key(api_key)
    raw = (text or "").strip()
    if not raw:
        return JSONResponse({"answer": "Provide some text to rewrite into normal English."})
    if not effective_key:
        return JSONResponse({
            "answer": "Add a Gemini API key in the top bar (or set GEMINI_API_KEY on the server) first to rewrite text into normal English."
        })

    prompt = (
        "Rewrite the following text into natural, plain, everyday English that reads smoothly "
        "and humanly. Keep all facts, numbers, formulas, and file/page citations accurate, "
        "vary sentence structure naturally, and avoid robotic or overly formal AI clichés. "
        "Return only the rewritten text.\n\n"
        f"TEXT:\n{raw}"
    )
    answer, err = call_gemini([{"text": prompt}], api_key=effective_key, timeout=45)
    return JSONResponse({"answer": answer or err})


@app.post("/practice")
async def generate_practice(
    topic: str = Form(""),
    api_key: str = Form(""),
    course_unit: str = Form("All"),
    num_questions: int = Form(4),
):
    effective_key = resolve_api_key(api_key)
    rows = fetch_rows_by_unit(course_unit)
    if not rows:
        return JSONResponse({
            "answer": "Upload study notes first so I can generate practice questions from your course material.",
            "sources": [],
        })

    query = topic.strip() or f"key concepts formulas definitions exercises {course_unit}"
    top = score_chunks(query, rows, top_k=8)
    if not top:
        top = [(0.5, r[1], r[2], r[3], r[4]) for r in rows[:6]]

    sources = [
        {"filename": f, "course_unit": u, "page": p, "score": round(s, 3), "snippet": c}
        for s, f, u, p, c in top
    ]
    context_block = "\n\n".join([f"[{f} | {u} - Page {p}]\n{c}" for _, f, u, p, c in top])
    count = max(1, min(int(num_questions or 4), 10))

    if not effective_key:
        excerpts = "\n\n---\n\n".join(
            [f"**{f} [{u}] (Page {p})**: {c}" for _, f, u, p, c in top[:3]]
        )
        return JSONResponse({
            "answer": (
                "Add a Gemini API key in the top bar (or set GEMINI_API_KEY on the server) to generate full practice questions & worked solutions. "
                "Here are key passages from your notes to drill from right now:\n\n" + excerpts
            ),
            "sources": sources,
        })

    prompt = (
        f"Create {count} university-level practice/CAT questions based on the course notes below"
        f"{' focusing on: ' + topic.strip() if topic.strip() else ''}. "
        "Include a mix of conceptual and calculation/application questions where appropriate, "
        "format math with $...$ / $$...$$, cite the note filename and page, and put "
        "step-by-step worked solutions under a clear '## Worked Solutions' heading at the end.\n\n"
        f"COURSE NOTES:\n{context_block}"
    )
    answer, err = call_gemini([{"text": prompt}], api_key=effective_key, timeout=45)
    return JSONResponse({"answer": answer or err, "sources": sources})


@app.post("/verify-math")
async def verify_math(expression: str = Form(...)):
    expr_str = (expression or "").strip()
    if not expr_str:
        return JSONResponse({
            "ok": False,
            "result": "Enter a math expression or equation (e.g., (1+0.05)**-3 or 2*x + 4 = 10).",
        })

    cleaned = expr_str.replace("^", "**")
    if HAS_SYMPY:
        try:
            transforms = standard_transformations + (implicit_multiplication_application,)
            if "=" in cleaned and "==" not in cleaned:
                left_s, right_s = cleaned.split("=", 1)
                lhs = parse_expr(left_s, transformations=transforms)
                rhs = parse_expr(right_s, transformations=transforms)
                diff = sp.simplify(lhs - rhs)
                free_syms = sorted(diff.free_symbols, key=lambda s: s.name)
                if not free_syms:
                    is_eq = bool(sp.simplify(lhs - rhs) == 0)
                    return JSONResponse({
                        "ok": True,
                        "result": f"LHS = {sp.simplify(lhs)}, RHS = {sp.simplify(rhs)} -> {'Equal' if is_eq else 'Not equal'}",
                    })
                sols = sp.solve(sp.Eq(lhs, rhs), free_syms)
                return JSONResponse({
                    "ok": True,
                    "result": f"Simplified difference: {diff} = 0 | Solution(s): {sols}",
                })
            parsed = parse_expr(cleaned, transformations=transforms)
            simplified = sp.simplify(parsed)
            numeric_val = ""
            if not simplified.free_symbols:
                try:
                    numeric_val = f" (approx. {float(simplified.evalf()):.6g})"
                except Exception:
                    pass
            return JSONResponse({
                "ok": True,
                "result": f"Simplified: {simplified}{numeric_val}",
            })
        except Exception as e:
            return JSONResponse({"ok": False, "result": f"Could not parse with SymPy: {e}"})

    allowed_names = {
        k: getattr(math, k)
        for k in dir(math)
        if not k.startswith("_") and (isinstance(getattr(math, k), (int, float)) or callable(getattr(math, k)))
    }
    if not re.fullmatch(r"[0-9a-zA-Z_+\-*/().^%,\s=]+", expr_str):
        return JSONResponse({"ok": False, "result": "Unsupported characters in expression."})
    try:
        val = eval(cleaned, {"__builtins__": {}}, allowed_names)
        return JSONResponse({"ok": True, "result": f"Result: {val}"})
    except Exception as e:
        return JSONResponse({"ok": False, "result": f"Evaluation error: {e}"})


@app.post("/export/docx")
async def export_docx(title: str = Form("Assignment Solutions"), content: str = Form(...)):
    doc = docx.Document()
    doc.add_heading(title.strip() or "Assignment Solutions", level=1)
    for block in (content or "").split("\n\n"):
        b = block.strip()
        if not b:
            continue
        if b.startswith("### "):
            doc.add_heading(b[4:].strip(), level=3)
        elif b.startswith("## "):
            doc.add_heading(b[3:].strip(), level=2)
        elif b.startswith("# "):
            doc.add_heading(b[2:].strip(), level=1)
        else:
            doc.add_paragraph(b)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", (title or "assignment_solutions").strip()) or "assignment_solutions"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.docx"'},
    )