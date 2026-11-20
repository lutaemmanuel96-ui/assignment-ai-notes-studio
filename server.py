import io
import os
import sqlite3
from typing import Optional

import requests
import sympy as sp
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pptx import Presentation
from pydantic import BaseModel
from pypdf import PdfReader

DB_PATH = os.getenv("DB_PATH", "notes.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

app = FastAPI(title="Assignment AI & Notes Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            unit TEXT DEFAULT 'General',
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


class AskRequest(BaseModel):
    question: str
    unit: Optional[str] = None
    style: str = "Normal English"
    api_key: Optional[str] = None


class MathRequest(BaseModel):
    expression: str


class ExportDocxRequest(BaseModel):
    title: str = "Assignment AI Export"
    content: str


def extract_text_from_upload(filename: str, raw_bytes: bytes) -> str:
    lower = (filename or "").lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    if lower.endswith(".docx"):
        doc = Document(io.BytesIO(raw_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    if lower.endswith(".pptx"):
        prs = Presentation(io.BytesIO(raw_bytes))
        slides_text = []
        for idx, slide in enumerate(prs.slides, start=1):
            parts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    txt = shape.text_frame.text.strip()
                    if txt:
                        parts.append(txt)
                if shape.has_table:
                    for row in shape.table.rows:
                        row_vals = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_vals:
                            parts.append(" | ".join(row_vals))
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes_txt = slide.notes_slide.notes_text_frame.text.strip()
                if notes_txt:
                    parts.append(f"[Speaker Notes: {notes_txt}]")
            if parts:
                slides_text.append(f"--- Slide {idx} ---\n" + "\n".join(parts))
        return "\n\n".join(slides_text).strip()
    return raw_bytes.decode("utf-8", errors="ignore").strip()


def call_gemini(prompt: str, api_key: Optional[str] = None) -> str:
    key = (api_key or GEMINI_API_KEY or "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail="GEMINI_API_KEY is not configured on the server or provided in the request.",
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    resp = requests.post(
        url,
        params={"key": key},
        json=payload,
        timeout=60,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Gemini API error: {resp.text}")

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Unexpected response format from Gemini API.")


@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("index.html", media_type="text/html")


@app.get("/manifest.json")
def get_manifest():
    return FileResponse("manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
def get_service_worker():
    return FileResponse("sw.js", media_type="application/javascript")


@app.get("/icon.svg")
def get_app_icon():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect width="512" height="512" rx="112" fill="#0f172a"/>
  <rect x="44" y="44" width="424" height="424" rx="88" fill="#1e293b" stroke="#38bdf8" stroke-width="12"/>
  <text x="256" y="295" font-family="Arial, sans-serif" font-size="180" font-weight="bold" fill="#38bdf8" text-anchor="middle">AI</text>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/notes")
def list_notes(unit: Optional[str] = None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if unit and unit != "All":
        cur.execute(
            "SELECT id, title, unit, substr(content, 1, 240) AS preview, created_at FROM notes WHERE unit = ? ORDER BY id DESC",
            (unit,),
        )
    else:
        cur.execute(
            "SELECT id, title, unit, substr(content, 1, 240) AS preview, created_at FROM notes ORDER BY id DESC"
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"notes": rows}


@app.post("/api/notes/upload")
async def upload_note(
    title: str = Form(...),
    unit: str = Form("General"),
    file: Optional[UploadFile] = File(None),
    text_content: Optional[str] = Form(None),
):
    content = (text_content or "").strip()
    if file is not None:
        raw = await file.read()
        extracted = extract_text_from_upload(file.filename or "", raw)
        content = f"{content}\n{extracted}".strip() if content else extracted

    if not content:
        raise HTTPException(status_code=400, detail="Provide a file or text content to save a note.")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notes (title, unit, content) VALUES (?, ?, ?)",
        (title.strip(), (unit or "General").strip(), content),
    )
    conn.commit()
    note_id = cur.lastrowid
    conn.close()
    return JSONResponse({"id": note_id, "title": title.strip(), "unit": unit or "General"})


@app.post("/api/ask")
def ask_ai(req: AskRequest):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if req.unit and req.unit != "All":
        cur.execute("SELECT title, unit, content FROM notes WHERE unit = ? ORDER BY id DESC LIMIT 5", (req.unit,))
    else:
        cur.execute("SELECT title, unit, content FROM notes ORDER BY id DESC LIMIT 5")
    notes = [dict(r) for r in cur.fetchall()]
    conn.close()

    context_blocks = []
    for n in notes:
        snippet = n["content"][:2500]
        context_blocks.append(f"[{n['title']} | Unit: {n['unit']}]\n{snippet}")
    context_text = "\n\n".join(context_blocks) if context_blocks else "No saved notes available."

    prompt = (
        f"You are a helpful study tutor. Write your response in '{req.style}' style.\n"
        f"Use the student's saved notes below when relevant and cite their titles.\n\n"
        f"SAVED NOTES:\n{context_text}\n\n"
        f"QUESTION:\n{req.question}"
    )
    answer = call_gemini(prompt, api_key=req.api_key)
    return {"answer": answer}


@app.post("/api/math")
def verify_math(req: MathRequest):
    expr_str = req.expression.strip()
    if not expr_str:
        raise HTTPException(status_code=400, detail="Expression cannot be empty.")

    try:
        if "=" in expr_str:
            lhs_str, rhs_str = expr_str.split("=", 1)
            lhs = sp.sympify(lhs_str)
            rhs = sp.sympify(rhs_str)
            sol = sp.solve(sp.Eq(lhs, rhs))
            return {"input": expr_str, "type": "equation", "solution": str(sol), "simplified": str(sp.simplify(lhs - rhs))}
        parsed = sp.sympify(expr_str)
        return {"input": expr_str, "type": "expression", "simplified": str(sp.simplify(parsed)), "evaluated": str(parsed.evalf())}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse math expression: {exc}")


@app.post("/api/export/docx")
def export_docx(req: ExportDocxRequest):
    doc = Document()
    doc.add_heading(req.title or "Assignment AI Export", level=1)
    for paragraph in (req.content or "").split("\n"):
        if paragraph.strip():
            doc.add_paragraph(paragraph)

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="study_output.docx"'},
    )