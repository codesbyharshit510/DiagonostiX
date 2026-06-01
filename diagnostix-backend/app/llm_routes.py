from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

import io
import json
import os
from urllib import error as urlerror
from urllib import request as urlrequest

from pypdf import PdfReader

# Optional OCR dependencies for scanned/image PDFs.
try:
    import pytesseract
    from pdf2image import convert_from_bytes
except Exception:  # pragma: no cover
    pytesseract = None
    convert_from_bytes = None

# OpenAI remains optional. Local Ollama is the primary path.
try:
    from openai import (
        OpenAI,
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
    )
except Exception:  # pragma: no cover
    OpenAI = None
    APIConnectionError = Exception
    APITimeoutError = Exception
    AuthenticationError = Exception
    BadRequestError = Exception
    RateLimitError = Exception


router = APIRouter(prefix="/llm", tags=["LLM"])


SYSTEM_PROMPT = (
    "You are DiagnostiX, a medical assistant. You provide clear, context-aware "
    "explanations of medical predictions and reports. You can answer general medical "
    "questions, but you do not provide medical advice or diagnoses. Always recommend "
    "consulting a licensed clinician for medical decisions.\n\n"
    "Write in clean GitHub-flavored Markdown. Use short sections and compact bullets. "
    "Prefer this structure when relevant:\n"
    "## Summary\n"
    "## Key Findings\n"
    "## What This Means\n"
    "## Next Steps\n"
    "## Medical Caution\n\n"
    "Use bold labels such as **Prediction**:, **Confidence**:, or **Interpretation**: "
    "when helpful. Keep the tone calm, clear, and readable."
)

MAX_REPORT_CHARS = 8000
MAX_REPORT_CONTEXT_CHARS = 5000
MAX_OCR_PAGES = 5
SUMMARY_KEYWORDS = (
    "impression",
    "findings",
    "conclusion",
    "summary",
    "diagnosis",
    "predicted",
    "prediction",
    "confidence",
    "recommend",
    "risk",
    "class",
    "result",
)


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _ndjson(payload: dict) -> str:
    return json.dumps(payload) + "\n"


def _chunk_text(text: str, chunk_size: int = 48):
    for idx in range(0, len(text), chunk_size):
        yield text[idx: idx + chunk_size]


def _extract_pdf_text_native(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def _extract_pdf_text_ocr(file_bytes: bytes) -> str:
    if convert_from_bytes is None or pytesseract is None:
        return ""

    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    poppler_path = os.getenv("POPPLER_PATH")
    kwargs = {}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    images = convert_from_bytes(file_bytes, dpi=220, fmt="png", **kwargs)
    chunks = []
    total_chars = 0

    for img in images[:MAX_OCR_PAGES]:
        ocr_text = (pytesseract.image_to_string(img) or "").strip()
        if not ocr_text:
            continue
        chunks.append(ocr_text)
        total_chars += len(ocr_text)
        if total_chars >= MAX_REPORT_CHARS:
            break

    return "\n".join(chunks).strip()


def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        native_text = _extract_pdf_text_native(file_bytes)
    except Exception as e:
        raise HTTPException(400, f"Failed to read PDF: {e}")

    if native_text and len(native_text.strip()) >= 120:
        return native_text[:MAX_REPORT_CHARS]

    try:
        ocr_text = _extract_pdf_text_ocr(file_bytes)
    except Exception:
        ocr_text = ""

    combined = "\n".join([part for part in [native_text, ocr_text] if part]).strip()
    return combined[:MAX_REPORT_CHARS]


def _prepare_report_context(report_text: str) -> str:
    if not report_text:
        return ""

    normalized_lines = []
    for raw_line in report_text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if len(line) >= 4:
            normalized_lines.append(line)

    if not normalized_lines:
        return report_text[:MAX_REPORT_CONTEXT_CHARS]

    prioritized = []
    seen = set()
    for line in normalized_lines:
        lower = line.lower()
        if any(keyword in lower for keyword in SUMMARY_KEYWORDS):
            if line not in seen:
                prioritized.append(line)
                seen.add(line)

    for line in normalized_lines:
        if len(prioritized) >= 24:
            break
        if line not in seen:
            prioritized.append(line)
            seen.add(line)

    context = "\n".join(prioritized)
    return context[:MAX_REPORT_CONTEXT_CHARS]


def _build_system_content(report_text: str) -> str:
    system_content = SYSTEM_PROMPT
    report_context = _prepare_report_context(report_text)
    if report_context:
        system_content += (
            "\n\nThe user provided a medical report. Focus on this extracted summary context:\n"
            f"{report_context}"
        )
    return system_content


def _rule_based_reply(message: str, has_report: bool) -> str:
    lower = (message or "").lower()

    if has_report:
        return (
            "## Summary\n"
            "I could not use the LLM service for full report interpretation.\n\n"
            "## What I Need\n"
            "- **Diagnosis** line\n"
            "- **Confidence** value\n"
            "- **Impression** or **Findings** section\n\n"
            "## What I Can Do\n"
            "Once you share those parts, I can explain each item in plain language.\n\n"
            "## Medical Caution\n"
            "This is informational only and not a medical diagnosis."
        )

    if any(k in lower for k in ["hello", "hi", "hey"]):
        return (
            "## Summary\n"
            "Hello. I can help explain disease basics, model outputs, Grad-CAM, and SHAP results.\n\n"
            "## You Can Ask Me\n"
            "- Explain a prediction\n"
            "- Summarize a report\n"
            "- Clarify Grad-CAM or SHAP\n"
            "- Describe a disease in simple language"
        )

    if "grad-cam" in lower or "gradcam" in lower:
        return (
            "Grad-CAM highlights image regions the model focused on for prediction. "
            "Brighter regions usually had stronger influence on the predicted class."
        )

    if "shap" in lower:
        return (
            "SHAP explains feature impact for tabular predictions. Positive SHAP values "
            "push toward the predicted class, while negative values push away."
        )

    if "pneumonia" in lower:
        return "Pneumonia is a lung infection. Imaging and clinical evaluation are both needed for confirmation."

    if "diabetes" in lower:
        return "Diabetes risk is often assessed using glucose, BMI, age, and related markers with clinical follow-up."

    if "heart" in lower:
        return "Heart disease risk models use factors like blood pressure, cholesterol, age, and symptoms."

    if "alzheimer" in lower:
        return "Alzheimer's is a progressive neurodegenerative condition; early clinical assessment improves management planning."

    if "brain tumor" in lower or "brain tumour" in lower:
        return "Brain tumor assessment typically uses MRI findings, symptom history, and specialist interpretation."

    return (
        "I am currently in fallback mode. Ask about disease explanations, Grad-CAM, SHAP, "
        "or share report findings and I will explain them in plain language."
    )


def _call_ollama(system_content: str, prior_messages: list[dict], message: str) -> str:
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            *prior_messages,
            {"role": "user", "content": message},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)

    content = (
        (data.get("message") or {}).get("content")
        if isinstance(data, dict)
        else None
    )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama returned an empty response")

    return content.strip()


def _stream_ollama(system_content: str, prior_messages: list[dict], message: str):
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            *prior_messages,
            {"role": "user", "content": message},
        ],
        "stream": True,
        "options": {"temperature": 0.2},
    }

    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlrequest.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue

            data = json.loads(line)
            delta = ((data.get("message") or {}).get("content") or "")
            if delta:
                yield _ndjson({"delta": delta, "provider": "ollama"})

            if data.get("done"):
                yield _ndjson({"done": True, "provider": "ollama"})
                break


def _call_openai(system_content: str, prior_messages: list[dict], message: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        raise RuntimeError("OpenAI fallback unavailable")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            *prior_messages,
            {"role": "user", "content": message},
        ],
        temperature=0.2,
    )

    reply = response.choices[0].message.content if response.choices else ""
    if not isinstance(reply, str) or not reply.strip():
        raise RuntimeError("OpenAI returned empty response")
    return reply.strip()


def _stream_text_reply(reply: str, provider: str, note: str | None = None):
    for chunk in _chunk_text(reply):
        yield _ndjson({"delta": chunk, "provider": provider})
    payload = {"done": True, "provider": provider}
    if note:
        payload["note"] = note
    yield _ndjson(payload)


def _stream_chat_response(system_content: str, prior_messages: list[dict], message: str, has_report: bool):
    try:
        yield from _stream_ollama(system_content, prior_messages, message)
        return
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, RuntimeError) as ollama_error:
        ollama_error_text = str(ollama_error)

    try:
        reply = _call_openai(system_content, prior_messages, message)
        yield from _stream_text_reply(reply, "openai_fallback")
        return
    except Exception:
        pass

    fallback_reply = _rule_based_reply(message, has_report=has_report)
    note = (
        "LLM providers unavailable. "
        f"ollama_error={ollama_error_text[:160]}"
    )
    yield from _stream_text_reply(fallback_reply, "fallback_rules", note=note)


@router.post("/chat")
async def chat_llm(
    message: str = Form(...),
    history: str | None = Form(None),
    pdf: UploadFile | None = File(None),
    stream: str | None = Form(None),
):
    prior_messages = []
    if history:
        try:
            parsed = json.loads(history)
            if isinstance(parsed, list):
                for item in parsed:
                    role = item.get("role")
                    content = item.get("content")
                    if role in ("user", "assistant") and isinstance(content, str):
                        prior_messages.append({"role": role, "content": content})
        except Exception:
            pass

    report_text = ""
    if pdf:
        file_bytes = await pdf.read()
        report_text = _extract_pdf_text(file_bytes)

    system_content = _build_system_content(report_text)

    if _is_true(stream):
        return StreamingResponse(
            _stream_chat_response(
                system_content=system_content,
                prior_messages=prior_messages,
                message=message,
                has_report=bool(report_text),
            ),
            media_type="application/x-ndjson",
        )

    try:
        reply = _call_ollama(system_content, prior_messages, message)
        return JSONResponse({"reply": reply, "provider": "ollama"})
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, RuntimeError) as ollama_error:
        ollama_error_text = str(ollama_error)

    try:
        reply = _call_openai(system_content, prior_messages, message)
        return JSONResponse({"reply": reply, "provider": "openai_fallback"})
    except RateLimitError:
        pass
    except AuthenticationError:
        pass
    except BadRequestError:
        pass
    except (APIConnectionError, APITimeoutError):
        pass
    except Exception:
        pass

    fallback_reply = _rule_based_reply(message, has_report=bool(report_text))
    return JSONResponse(
        {
            "reply": fallback_reply,
            "provider": "fallback_rules",
            "note": "LLM providers unavailable. Configure Ollama for full local responses.",
            "debug": f"ollama_error={ollama_error_text[:200]}",
        }
    )
