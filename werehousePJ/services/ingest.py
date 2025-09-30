from __future__ import annotations
from typing import List, Dict, Any
import re
import pandas as pd

# ---- Required/Optional columns per table (no dim_date) ----
REQUIRED = {
    "edu.dim_student": ["student_id", "full_name", "class"],
    "edu.dim_course": ["course_id", "name"],
    "edu.fact_assessment": ["student_id", "course_id", "date_key", "assessment_type", "score", "max_score"],
    "edu.fact_attendance": ["student_id", "course_id", "date_key", "present"],
    "edu.fact_assignment": ["student_id", "course_id", "date_key", "submitted"],
}
ALLOWED_EXTRA = {
    "edu.dim_student": ["program", "birthdate"],
    "edu.dim_course": ["credit"],
    "edu.fact_assignment": ["score"],
}

# ---- Header normalization & synonyms ----
def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    def norm(col: str) -> str:
        c = col.strip()
        c = c.replace("\ufeff", "")
        c = re.sub(r"\(.*?\)", "", c)             # "present(0/1)" -> "present"
        c = re.sub(r"[^A-Za-z0-9_]+", "_", c)     # non-alnum -> underscore
        c = re.sub(r"_+", "_", c).strip("_")
        return c.lower()
    df = df.copy()
    df.columns = [norm(c) for c in df.columns]
    return df

SYNONYMS = {
    "full_name": ["name", "fullname"],
    "present": ["is_present", "attended", "attendance", "status_present"],
    "submitted": ["is_submitted", "turned_in"],
    "date_key": ["dateid", "datekey"],
}

def _apply_synonyms(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for target, alts in SYNONYMS.items():
        if target not in d.columns:
            for a in alts:
                if a in d.columns:
                    d[target] = d[a]
                    break
    # Special: status → present
    if "present" not in d.columns and "status" in d.columns:
        s = d["status"].astype(str).str.lower()
        d["present"] = s.eq("present").astype(int)
    return d

def _ensure_date_key_local(df: pd.DataFrame) -> pd.DataFrame:
    """
    ถ้ามี 'dt' แต่ไม่มี 'date_key' ให้คำนวณ date_key = YYYYMMDD จาก dt โดยตรง (ไม่อ้าง dim_date)
    """
    d = df.copy()
    if "date_key" not in d.columns and "dt" in d.columns:
        dt = pd.to_datetime(d["dt"], errors="coerce")
        d["date_key"] = (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).astype("Int64")
    return d

def _normalize_dtypes(table: str, df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if table.endswith("fact_attendance") and "present" in d.columns:
        d["present"] = d["present"].astype(int, errors="ignore")
    if table.endswith("fact_assignment") and "submitted" in d.columns:
        d["submitted"] = d["submitted"].astype(int, errors="ignore")
    if "date_key" in d.columns:
        d["date_key"] = d["date_key"].astype(int, errors="ignore")
    return d

def _detect_table(df: pd.DataFrame, hint: str) -> str:
    name = hint.lower()
    for t in REQUIRED.keys():
        if t.split(".")[-1] in name:
            return t
    cols = set(df.columns)
    for t, req in REQUIRED.items():
        if set(req).issubset(cols):
            return t
    raise ValueError(f"Cannot detect target table for '{hint}'. Please rename file/sheet or include required columns.")

# ---- Public: preview (no DB writes) ----
def peek_files(files: List[Any]) -> List[Dict[str, Any]]:
    """Return per-file preview info (after header normalization & synonyms)."""
    result = []
    for f in files:
        fname = getattr(f, "name", "uploaded")
        if fname.lower().endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(f)
            for sheet in xls.sheet_names:
                raw = xls.parse(sheet)
                df = _ensure_date_key_local(_apply_synonyms(_normalize_headers(raw)))
                try:
                    table = _detect_table(df, hint=f"{fname}:{sheet}")
                except Exception as e:
                    table = f"UNKNOWN ({e})"
                result.append({
                    "source": f"{fname}:{sheet}",
                    "rows": int(len(df)),
                    "columns": list(df.columns),
                    "table_guess": table,
                    "sample": df.head(5).to_dict(orient="records")
                })
        else:
            raw = pd.read_csv(f)
            df = _ensure_date_key_local(_apply_synonyms(_normalize_headers(raw)))
            try:
                table = _detect_table(df, hint=fname)
            except Exception as e:
                table = f"UNKNOWN ({e})"
            result.append({
                "source": fname,
                "rows": int(len(df)),
                "columns": list(df.columns),
                "table_guess": table,
                "sample": df.head(5).to_dict(orient="records")
            })
    return result

# ---- Public: ingest (write to DB) ----
def ingest_files(ch, files: List[Any], forced_table: str | None = None) -> Dict[str, int]:
    """
    Load CSV/XLSX into ClickHouse edu.* tables.
    - Normalizes headers/synonyms automatically
    - Compute date_key from dt when needed (no dim_date)
    """
    report: Dict[str, int] = {}
    for f in files:
        fname = getattr(f, "name", "uploaded")
        if fname.lower().endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(f)
            for sheet in xls.sheet_names:
                raw = xls.parse(sheet)
                df = _ensure_date_key_local(_apply_synonyms(_normalize_headers(raw)))
                table = forced_table or _detect_table(df, hint=f"{fname}:{sheet}")
                df = _normalize_dtypes(table, df)

                cols = REQUIRED[table] + [c for c in ALLOWED_EXTRA.get(table, []) if c in df.columns]
                cols = [c for c in cols if c in df.columns]
                if df.empty or len(df.index) == 0:
                    report[table] = report.get(table, 0) + 0
                    continue
                if not set(REQUIRED[table]).issubset(set(df.columns)):
                    missing = list(set(REQUIRED[table]) - set(df.columns))
                    raise ValueError(f"{fname}:{sheet} missing required columns: {missing}")
                ch.insert_df(table, df, cols)
                report[table] = report.get(table, 0) + int(len(df))
        else:
            raw = pd.read_csv(f)
            df = _ensure_date_key_local(_apply_synonyms(_normalize_headers(raw)))
            table = forced_table or _detect_table(df, hint=fname)
            df = _normalize_dtypes(table, df)

            cols = REQUIRED[table] + [c for c in ALLOWED_EXTRA.get(table, []) if c in df.columns]
            cols = [c for c in cols if c in df.columns]
            if df.empty or len(df.index) == 0:
                report[table] = report.get(table, 0) + 0
                continue
            if not set(REQUIRED[table]).issubset(set(df.columns)):
                missing = list(set(REQUIRED[table]) - set(df.columns))
                raise ValueError(f"{fname} missing required columns: {missing}")
            ch.insert_df(table, df, cols)
            report[table] = report.get(table, 0) + int(len(df))
    return report
