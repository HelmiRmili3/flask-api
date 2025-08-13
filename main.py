import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
import pandas as pd
import tempfile
import shutil
import uvicorn
import cv2
import numpy as np
import os
import string
import re
import Levenshtein  # pip install python-Levenshtein
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()          # reads .env automatically

app = FastAPI(title="Receipt OCR API")

reader = easyocr.Reader(["en", "fr"], gpu=False)

TOTAL_KEYWORDS = [
    "total", "total à payer", "total due", "amount due", "amount payable",
    "total to pay", "payable amount", "grand total", "balance due", "total bill",
    "total price", "amount owed",
    "montant total", "montant à payer", "somme à payer", "total dû",
    "total général", "solde à payer", "prix total", "facture totale", "montant dû", "à régler"
]

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN")
)
SYSTEM_PROMPT = (
    "You receive a raw list of strings. "
    "Return a JSON array that contains **exactly the same number of strings**, "
    "in the same order, with only obvious spelling or OCR errors corrected. "
    "Do **not** merge lines, delete lines, or change their meaning. "
    "Output must be valid JSON and nothing else."
)

def correct_typos(lines: list[str]) -> list[str]:
    """Call the LLM and return the corrected list."""
    if not lines:
        return lines
    user_prompt = "\n".join(lines)
    response = client.chat.completions.create(
        model="gpt-4o-mini",         # or "gpt-3.5-turbo", "claude-3-haiku", etc.
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0,
        max_tokens=500
    )
    # print(response)

    raw = response.choices[0].message.content.strip()
    
    raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(raw)
###############################################################################
# Helpers
###############################################################################
def _sanitize(text: str) -> str:
    """Lower-case, strip punctuation and extra spaces."""
    return re.sub(r"\s+", " ", text.lower().translate(str.maketrans("", "", string.punctuation))).strip()

def _is_price(text: str) -> bool:
    """Return True if the text looks like a price (e.g. 12.34, 12,34, 12$)."""
    text = text.strip()
    # remove leading currency symbols once
    text = re.sub(r"^[€$£]", "", text)
    # accept both dot and comma as decimal separator
    return bool(re.fullmatch(r"\d+(?:[.,]\d{1,2})?", text))

def _to_float(text: str) -> float | None:
    """Convert price string to float (comma or dot decimal)."""
    try:
        text = re.sub(r"^[€$£]", "", text.strip())
        text = text.replace(",", ".")
        return float(text)
    except Exception:
        return None

def fuzzy_match_keyword(text: str, keywords: list[str], threshold: float = 0.75) -> bool:
    """True if *text* is within Levenshtein distance (ratio) of any keyword."""
    text = _sanitize(text)
    for kw in keywords:
        if Levenshtein.ratio(text, _sanitize(kw)) >= threshold:
            return True
    return False

###############################################################################
# Grouping & Total extraction
###############################################################################
def group_boxes_by_line(predictions, y_threshold: int = 20, image_width: int = 1000):
    """Return list of dicts: {y_min, y_max, members, box}."""
    groups = []
    for bbox, text, conf in sorted(predictions, key=lambda p: sum(pt[1] for pt in p[0]) / 4):
        y_center = sum(pt[1] for pt in bbox) / 4
        for g in groups:
            if abs(y_center - (g["y_min"] + g["y_max"]) / 2) <= y_threshold:
                g["members"].append((bbox, text, conf))
                g["y_min"] = min(g["y_min"], min(pt[1] for pt in bbox))
                g["y_max"] = max(g["y_max"], max(pt[1] for pt in bbox))
                break
        else:
            y_min = min(pt[1] for pt in bbox)
            y_max = max(pt[1] for pt in bbox)
            groups.append({"y_min": y_min, "y_max": y_max, "members": [(bbox, text, conf)]})
    # build full-width box for drawing
    for g in groups:
        g["box"] = [[0, g["y_min"]], [image_width - 1, g["y_min"]],
                    [image_width - 1, g["y_max"]], [0, g["y_max"]]]
    return groups
def extract_total(groups, image_height: int) -> str | None:
    """
    First attempt: keyword-based.
    Fallback: largest numeric value in bottom 30 % of the receipt.
    Returns raw text (string) or None.
    """

    def _clean(text: str) -> str:
        """Keep digits, comma, dot, $€£ and strip spaces."""
        return re.sub(r"[^\d.,$€£]", "", text).strip()

    # ---------- Pass 1 : keyword ----------
    for g in groups:
        members = sorted(g["members"], key=lambda m: min(pt[0] for pt in m[0]))
        for idx, (_, text, _) in enumerate(members):
            if fuzzy_match_keyword(text, TOTAL_KEYWORDS):
                # look right
                for j in range(idx + 1, len(members)):
                    cand = members[j][1]
                    # print(">>> keyword candidate:", repr(cand))
                    return cand

    return None

###############################################################################
# FastAPI route
###############################################################################
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/extract-receipt")
async def extract_receipt(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        image = cv2.imread(tmp_path)
        if image is None:
            return JSONResponse(content={"detail": "Cannot decode image"}, status_code=400)
        h, w = image.shape[:2]

        predictions = reader.readtext(tmp_path)
        groups = group_boxes_by_line(predictions, y_threshold=20, image_width=w)

        total_value = extract_total(groups, h)

        # Recognised lines (for CSV)
        lines = []
        for g in groups:
            line = " ".join(txt for _, txt, _ in sorted(g["members"], key=lambda m: min(pt[0] for pt in m[0])))
            lines.append(line)

        # Annotate
        vis = image.copy()
        for g in groups:
            cv2.polylines(vis, [np.array(g["box"], dtype=np.int32)], True, (255, 255, 0), 2)
        for bbox, text, _ in predictions:
            cv2.polylines(vis, [np.array(bbox, dtype=np.int32)], True, (0, 255, 0), 1)
            cv2.putText(vis, text, (int(bbox[0][0]), int(bbox[0][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out_path = os.path.join(os.getcwd(), "annotated_receipt.jpg")
        cv2.imwrite(out_path, vis)

        # CSV
        df = pd.DataFrame({"line_text": lines})
        csv_str = df.to_csv(index=False)

        return {
            "status": "success",
            "total": total_value,
            "recognized_lines":correct_typos(lines),
            "full_predictions": [
                {"text": t, "box": [[float(c) for c in pt] for pt in b], "confidence": float(c)}
                for b, t, c in predictions
            ],
            "csv": csv_str,
            "annotated_image_path": out_path
        }
    finally:
        os.unlink(tmp_path)

###############################################################################
# Entrypoint
###############################################################################
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)