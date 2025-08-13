

# from fastapi import FastAPI, File, UploadFile
# import easyocr
# import pandas as pd
# import tempfile
# import shutil
# import uvicorn
# import cv2
# import numpy as np
# import os

# app = FastAPI()

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en','fr'], gpu=True)


# def center_y(pred):
#     bbox = pred[0]
#     return sum(p[1] for p in bbox) / len(bbox)


# def center_x(pred):
#     bbox = pred[0]
#     return sum(p[0] for p in bbox) / len(bbox)


# def group_boxes_by_y(predictions, y_tolerance_ratio=0.5):
#     """
#     Group OCR predictions into horizontal lines based on the height of the first box in each line.
#     """
#     predictions = sorted(predictions, key=lambda p: center_y(p))  # top to bottom
#     grouped_lines = []
#     used = set()

#     for i, pred in enumerate(predictions):
#         if i in used:
#             continue

#         # First box defines the Y band
#         bbox = pred[0]
#         y_min = min(p[1] for p in bbox)
#         y_max = max(p[1] for p in bbox)
#         box_height = y_max - y_min
#         y_tol = box_height * y_tolerance_ratio

#         same_line = [pred]
#         used.add(i)

#         for j, other in enumerate(predictions):
#             if j in used:
#                 continue
#             obox = other[0]
#             oy_min = min(p[1] for p in obox)
#             oy_max = max(p[1] for p in obox)

#             # Check if vertically aligned within tolerance
#             if not (oy_max < y_min - y_tol or oy_min > y_max + y_tol):
#                 same_line.append(other)
#                 used.add(j)

#         # Sort horizontally
#         same_line.sort(key=lambda p: center_x(p))
#         grouped_lines.append(same_line)

#     return grouped_lines


# @app.post("/extract-receipt")
# async def extract_receipt(file: UploadFile = File(...)):
#     # Save uploaded file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#         shutil.copyfileobj(file.file, tmp)
#         tmp_path = tmp.name

#     # Load image
#     image = cv2.imread(tmp_path)

#     # OCR
#     predictions = reader.readtext(tmp_path)

#     # Annotate
#     annotated_image = image.copy()
#     for bbox, text, _ in predictions:
#         pts = np.array(bbox, dtype=np.int32)
#         cv2.polylines(annotated_image, [pts], True, (0, 255, 0), 2)
#         label_pos = (int(bbox[0][0]), int(bbox[0][1] - 10))
#         cv2.putText(annotated_image, text, label_pos,
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     output_path = os.path.join(os.getcwd(), 'annotated_receipt.jpg')
#     cv2.imwrite(output_path, annotated_image)

#     # Group by Y (lines)
#     grouped_lines = group_boxes_by_y(predictions)

#     recognized_text = []
#     full_predictions = []

#     for line_preds in grouped_lines:
#         line_text = ' '.join(text for _, text, _ in line_preds)
#         recognized_text.append(line_text)

#         for bbox, text, conf in line_preds:
#             full_predictions.append({
#                 "text": text,
#                 "box": [[float(x), float(y)] for x, y in bbox],
#                 "confidence": float(conf)
#             })

#     # CSV
#     df = pd.DataFrame({"line_text": recognized_text})
#     csv_data = df.to_csv(index=False)

#     os.unlink(tmp_path)  # cleanup

#     return {
#         "status": "success",
#         "recognized_text": recognized_text,
#         "full_predictions": full_predictions,
#         "csv": csv_data,
#         "annotated_image_path": output_path
#     }


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# from fastapi import FastAPI, File, UploadFile
# import easyocr
# import pandas as pd
# import tempfile
# import shutil
# import uvicorn
# import cv2
# import numpy as np
# import os

# app = FastAPI()

# reader = easyocr.Reader(['en'], gpu=False)

# # Assumed DPI of the image for mm calculation (adjust if known)
# DPI = 300
# PIXELS_PER_MM = DPI / 25.4  # ~11.81 pixels per mm

# def center_point(box):
#     xs = [p[0] for p in box]
#     ys = [p[1] for p in box]
#     return (sum(xs) / len(xs), sum(ys) / len(ys))

# def is_point_inside_box(point, box):
#     polygon = np.array(box, dtype=np.int32)
#     result = cv2.pointPolygonTest(polygon, point, False)
#     return result >= 0

# def create_full_width_box(box_height, y_min, image_width):
#     return [
#         [0, y_min],
#         [image_width - 1, y_min],
#         [image_width - 1, y_min + box_height],
#         [0, y_min + box_height]
#     ]

# def horizontal_gaps_between_boxes(boxes):
#     gaps = []
#     for i in range(len(boxes) - 1):
#         box_a = boxes[i]
#         box_b = boxes[i + 1]
#         right_a = max(p[0] for p in box_a)
#         left_b = min(p[0] for p in box_b)
#         gap = left_b - right_a
#         gaps.append(gap)
#     return gaps

# def pixels_to_spaces(pixels):
#     mm = pixels / PIXELS_PER_MM
#     spaces = round(mm / 3)  # 3 mm = 1 space character
#     return max(spaces, 0)

# def left_margin_spaces(box):
#     left_x = min(p[0] for p in box)
#     return pixels_to_spaces(left_x)

# @app.post("/extract-receipt")
# async def extract_receipt(file: UploadFile = File(...)):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#         shutil.copyfileobj(file.file, tmp)
#         tmp_path = tmp.name

#     image = cv2.imread(tmp_path)
#     image_height, image_width = image.shape[:2]

#     predictions = reader.readtext(tmp_path)

#     grouping_boxes = []

#     for pred in predictions:
#         bbox, text, conf = pred
#         c_point = center_point(bbox)
#         y_min = min(p[1] for p in bbox)
#         y_max = max(p[1] for p in bbox)
#         box_height = y_max - y_min

#         assigned = False
#         for group in grouping_boxes:
#             if is_point_inside_box(c_point, group["box"]):
#                 group["members"].append(pred)
#                 assigned = True
#                 break

#         if not assigned:
#             full_width_box = create_full_width_box(box_height, y_min, image_width)
#             grouping_boxes.append({
#                 "box": full_width_box,
#                 "members": [pred],
#                 "height": box_height,
#                 "y_min": y_min
#             })

#     annotated_image = image.copy()

#     group_color = (0, 255, 255)
#     box_color = (0, 255, 0)

#     for group in grouping_boxes:
#         pts = np.array(group["box"], dtype=np.int32)
#         cv2.polylines(annotated_image, [pts], True, group_color, 3)

#     for pred in predictions:
#         bbox, text, conf = pred
#         pts = np.array(bbox, dtype=np.int32)
#         cv2.polylines(annotated_image, [pts], True, box_color, 2)
#         label_pos = (int(bbox[0][0]), int(bbox[0][1]) - 10)
#         cv2.putText(annotated_image, text, label_pos,
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     output_path = os.path.join(os.getcwd(), 'annotated_receipt.jpg')
#     cv2.imwrite(output_path, annotated_image)

#     recognized_lines = []
#     full_predictions = []
#     spaces_per_group = []

#     for group in grouping_boxes:
#         members = sorted(group["members"], key=lambda p: center_point(p[0])[0])
#         line_text = ' '.join(text for _, text, _ in members)
#         recognized_lines.append(line_text)

#         for bbox, text, conf in members:
#             full_predictions.append({
#                 "text": text,
#                 "box": [[float(x), float(y)] for x, y in bbox],
#                 "confidence": float(conf)
#             })

#         boxes = [m[0] for m in members]
#         gaps_px = horizontal_gaps_between_boxes(boxes)
#         gaps_spaces = [pixels_to_spaces(gap) for gap in gaps_px]

#         # For the first box, calculate left margin spaces (distance from image left edge)
#         first_box_margin = left_margin_spaces(boxes[0]) if boxes else 0

#         # Insert the margin spaces at the start to reflect leading space in the line
#         spaces_line = [first_box_margin] + gaps_spaces

#         spaces_per_group.append(spaces_line)

#     df = pd.DataFrame({"line_text": recognized_lines})
#     csv_data = df.to_csv(index=False)

#     os.unlink(tmp_path)

#     return {
#         "status": "success",
#         "recognized_lines": recognized_lines,
#         "full_predictions": full_predictions,
#         "spaces_between_boxes_as_spaces": spaces_per_group,
#         "csv": csv_data,
#         "annotated_image_path": output_path
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


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