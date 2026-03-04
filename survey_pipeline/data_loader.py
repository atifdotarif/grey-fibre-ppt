"""
Data loading helpers for the survey slide pipeline.

Provides a single entry point `load_ai_long` that returns a DataFrame
in the ai_long format, whether the input Excel is already cleaned
(`ai_long` sheet) or is the raw 250870FN.xlsx-style `ExcelData` sheet.
"""

from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd


def load_ai_long(data_path: str) -> pd.DataFrame:
    """
    Load survey data in the unified ai_long format.

    Supports two input styles:
      1) Cleaned file with an 'ai_long' sheet
      2) Raw crosstab file like 250870FN.xlsx with an 'ExcelData' sheet
         (single wide table with Question / Response rows)
    """
    xl = pd.ExcelFile(data_path)
    if "ai_long" in xl.sheet_names:
        return pd.read_excel(xl, sheet_name="ai_long")
    if "ExcelData" in xl.sheet_names:
        raw = pd.read_excel(xl, sheet_name="ExcelData")
        return _build_ai_long_from_exceldata(raw)
    raise ValueError(
        f"Unsupported Excel format for {data_path!r} — expected a sheet named "
        "'ai_long' or 'ExcelData'."
    )


def _build_ai_long_from_exceldata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw 'ExcelData' sheet (250870FN.xlsx) into ai_long format.

    The sheet is a single wide crosstab with blocks like:
      Table N
      Question X:  (or Q18 / Q18:)
      <one or more lines of question text>
      (blank row)
      BASE=TOTAL SAMPLE  (or BASE: DON'T KNOW / REF, BASE: ..., i.e. BASE= or BASE:)
      <answer option rows with TOTAL column values>

    We:
      - Derive question numbers from 'Question X:' lines in the Response column
      - Build question_text from the following non-empty lines (Question/Response)
      - Use the first data column after 'Response' as TOTAL (overall) values
      - Treat each subsequent non-empty Response row as an answer option.

    This is tailored to the 250870FN.xlsx layout, not a generic crosstab parser.
    """
    records: List[Dict] = []

    if df.shape[1] < 5:
        return _empty_ai_long()

    total_col = df.columns[4]  # first data column after Response
    n_rows = len(df)

    i = 0
    while i < n_rows:
        resp = df.at[i, "Response"]
        resp_str = str(resp).strip() if not pd.isna(resp) else ""

        # Match "Question 18:" or "Q18" / "Q18:" / "Q 18" (some exports use Q-prefix)
        m = re.search(r"Question\s+(\d+)\s*:", resp_str)
        if m:
            qnum = int(m.group(1))
        else:
            mq = re.search(r"^Q\s*(\d+)\s*:?\s*$", resp_str, re.IGNORECASE)
            if not mq:
                i += 1
                continue
            qnum = int(mq.group(1))

        # Collect question text from following lines until blank/next section
        qtext_parts: List[str] = []
        j = i + 1
        while j < n_rows:
            q_col = df.at[j, "Question"] if "Question" in df.columns else None
            r_col = df.at[j, "Response"]
            q_text = str(q_col).strip() if not pd.isna(q_col) else ""
            r_text = str(r_col).strip() if not pd.isna(r_col) else ""

            if not q_text and not r_text:
                break
            if r_text.startswith("Table ") or r_text.startswith("Question ") or re.match(r"^Q\s*\d+\s*:?\s*$", r_text, re.IGNORECASE):
                break
            if re.match(r"^BASE[=:]", r_text):
                break

            line = q_text or r_text
            if line:
                qtext_parts.append(line)

            j += 1

        question_text = " ".join(qtext_parts).strip()

        # Find the BASE row after the question text (e.g. BASE=TOTAL SAMPLE or BASE: DON'T KNOW / REF)
        base_row = None
        k = j + 1
        while k < n_rows:
            r_col = df.at[k, "Response"]
            r_text = str(r_col).strip() if not pd.isna(r_col) else ""
            if r_text.startswith("Question ") or re.match(r"^Q\s*\d+\s*:?\s*$", r_text, re.IGNORECASE):
                break
            if re.match(r"^BASE[=:]", r_text):
                base_row = k
                break
            k += 1

        if base_row is None:
            i = j + 1
            continue

        base_n = df.at[base_row, total_col]
        table_number = df.at[base_row, "Table ID"] if "Table ID" in df.columns else None
        qid = f"Q{qnum}"

        # Answer option rows: from base_row+1 until blank / next table / next question
        a = base_row + 1
        while a < n_rows:
            r_col = df.at[a, "Response"]
            r_text = str(r_col).strip() if not pd.isna(r_col) else ""
            if not r_text:
                break
            if r_text.startswith("Table ") or r_text.startswith("Question ") or re.match(r"^Q\s*\d+\s*:?\s*$", r_text, re.IGNORECASE):
                break

            val = df.at[a, total_col]
            if pd.isna(val):
                a += 1
                continue

            try:
                num = float(val)
            except Exception:
                a += 1
                continue

            pct = num * 100.0 if num <= 1.0 else num

            records.append(
                {
                    "table_number": table_number,
                    "question_id": qid,
                    "question_text": question_text or qid,
                    "base_n": base_n,
                    "answer_option": r_text,
                    "raw_value": num,
                    "pct": pct,
                }
            )

            a += 1

        i = a

    if not records:
        return _empty_ai_long()

    ai_long = pd.DataFrame.from_records(records)

    ai_long["rank_pct_desc"] = (
        ai_long.groupby("question_id")["pct"].rank(method="dense", ascending=False).astype(int)
    )
    ai_long["is_top3"] = ai_long["rank_pct_desc"] <= 3
    ai_long["is_net"] = ai_long["answer_option"].str.contains("NET", case=False, na=False)

    return ai_long


def _empty_ai_long() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "table_number",
            "question_id",
            "question_text",
            "base_n",
            "answer_option",
            "raw_value",
            "pct",
            "rank_pct_desc",
            "is_top3",
            "is_net",
        ]
    )
