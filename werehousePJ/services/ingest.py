from __future__ import annotations
from typing import List, Dict, Any
import re
import pandas as pd

# =========================
# Config: required columns
# =========================
REQUIRED: Dict[str, List[str]] = {
    "edu.dim_student": ["student_id", "full_name", "class"],
    "edu.dim_course": ["course_id", "name"],
    "edu.fact_assessment": ["student_id", "course_id", "date_key", "assessment_type", "score", "max_score"],
    "edu.fact_attendance": ["student_id", "course_id", "date_key", "present"],
    "edu.fact_assignment": ["student_id", "course_id", "date_key", "submitted"],
    # unified events (ไฟล์เดียวจบ)
    "edu.events_raw": ["student_id", "course_id", "date_key", "kind"],
}

ALLOWED_EXTRA: Dict[str, List[str]] = {
    "edu.dim_student": ["program", "birthdate"],
    "edu.dim_course": ["credit"],
    "edu.fact_assignment": ["score"],
    "edu.events_raw": ["dt", "present", "submitted", "assessment_type", "score", "max_score", "note"],
}

# header synonyms
SYNONYMS: Dict[str, List[str]] = {
    "full_name": ["name", "fullname"],
    "present": ["is_present", "attended", "attendance", "status_present", "status"],
    "submitted": ["is_submitted", "turned_in"],
    "date_key": ["dateid", "datekey", "date_id"],
    "kind": ["type", "event", "category"],
}

# =========================
# Helpers
# =========================
def _drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # แทนช่องว่างล้วนเป็น NA แล้วทิ้งแถวที่ว่างทั้งแถว
    d = d.replace(r"^\s*$", pd.NA, regex=True)
    d = d.dropna(how="all")
    return d

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """ทำความสะอาดชื่อคอลัมน์ให้เป็น snake_case ตัวเล็ก ไม่มี BOM/วงเล็บ"""
    def norm(col: str) -> str:
        c = str(col).strip().replace("\ufeff", "")
        c = re.sub(r"\(.*?\)", "", c)             # "present(0/1)" -> "present"
        c = re.sub(r"[^A-Za-z0-9_]+", "_", c)     # non-alnum -> underscore
        c = re.sub(r"_+", "_", c).strip("_")
        return c.lower()
    d = df.copy()
    d.columns = [norm(c) for c in d.columns]
    return d

def _apply_synonyms(df: pd.DataFrame) -> pd.DataFrame:
    """map คอลัมน์ตาม SYNONYMS; สร้าง 'present' จาก 'status' ที่เป็นคำ"""
    d = df.copy()
    cols = set(d.columns)
    # map generic synonyms
    for target, alts in SYNONYMS.items():
        if target not in cols:
            for a in alts:
                if a in d.columns:
                    d[target] = d[a]
                    break
    # special: status (string) -> present (0/1)
    if "present" not in d.columns and "status" in d.columns:
        s = d["status"].astype(str).str.lower()
        d["present"] = s.isin(["present", "attend", "attended", "true", "1", "yes", "y"]).astype(int)
    return d

def _ensure_date_key_local(df: pd.DataFrame) -> pd.DataFrame:
    """
    ถ้ามี 'dt' แต่ไม่มี/มี date_key ที่ว่าง ให้คำนวณ date_key = YYYYMMDD จาก dt
    รองรับ dt เป็น string / pandas.Timestamp / datetime.date
    """
    d = _drop_empty_rows(df).copy()

    # ถ้ามี date_key ให้แปลงเป็นตัวเลขก่อน
    if "date_key" in d.columns:
        d["date_key"] = pd.to_numeric(d["date_key"], errors="coerce")

    # ถ้ามี dt → สร้าง/เติม date_key
    if "dt" in d.columns:
        dt = pd.to_datetime(d["dt"], errors="coerce")
        dk = (dt.dt.year * 10000 + dt.dt.month * 100 + dt.dt.day).astype("Int64")
        if "date_key" in d.columns:
            d["date_key"] = d["date_key"].fillna(dk)
        else:
            d["date_key"] = dk

    # validate
    if "date_key" not in d.columns:
        raise ValueError("Missing 'date_key' (or 'dt'). Please provide dt=YYYY-MM-DD หรือ date_key=YYYYMMDD")

    if d["date_key"].isna().any():
        # อนุญาตให้ทิ้งแถวที่ไม่สมบูรณ์ เพื่อกัน error จาก Excel ที่มีแถวว่าง
        d = d.loc[~d["date_key"].isna()].copy()

    if d.empty:
        raise ValueError("All rows invalid: date_key/dt cannot be parsed")

    d["date_key"] = pd.to_numeric(d["date_key"], errors="coerce").astype("int64")
    return d

def _normalize_dtypes(table: str, df: pd.DataFrame) -> pd.DataFrame:
    """บังคับชนิดข้อมูลให้เหมาะกับ ClickHouse"""
    d = df.copy()

    # text columns
    text_cols = [
        "student_id", "course_id", "assessment_type", "name", "full_name",
        "class", "program", "kind", "note"
    ]
    for c in text_cols:
        if c in d.columns:
            d[c] = d[c].astype(str)

    # integers 0/1
    for c in ["present", "submitted"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype("int64")

    # numerics
    for c in ["score", "max_score", "credit"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # date_key
    if "date_key" in d.columns:
        d["date_key"] = pd.to_numeric(d["date_key"], errors="coerce").astype("int64")

    # birthdate -> python date
    if "birthdate" in d.columns:
        bd = pd.to_datetime(d["birthdate"], errors="coerce")
        d["birthdate"] = bd.dt.date

    return d

def _detect_table(df: pd.DataFrame, hint: str) -> str:
    """
    เดาตารางปลายทางจากเนื้อหา DataFrame + hint (ชื่อไฟล์:ชีต)
      1) events_raw หากพบ kind + student_id + course_id + (date_key|dt)
      2) เดาตามชื่อไฟล์/ชีต
      3) เดาตามชุดคอลัมน์
      4) Fallback หา table ที่ required ครบ/ใกล้สุด
    """
    name = (hint or "").lower()
    cols = set(c.lower() for c in df.columns)

    # 1) unified events
    if (
        "kind" in cols
        and "student_id" in cols
        and "course_id" in cols
        and ("date_key" in cols or "dt" in cols)
    ):
        return "edu.events_raw"

    # 2) hint keywords
    hint_map = [
        (("events_raw", "events", "event"), "edu.events_raw"),
        (("fact_attendance", "attendance", "attend", "present"), "edu.fact_attendance"),
        (("fact_assessment", "assessment", "score", "exam", "quiz", "test"), "edu.fact_assessment"),
        (("fact_assignment", "assignment", "homework", "submit"), "edu.fact_assignment"),
        (("dim_student", "students", "student"), "edu.dim_student"),
        (("dim_course", "courses", "course"), "edu.dim_course"),
    ]
    for keys, table in hint_map:
        if any(k in name for k in keys):
            return table

    # 3) columns pattern
    has_date = ("date_key" in cols) or ("dt" in cols)
    if {"student_id", "course_id", "present"}.issubset(cols) and has_date:
        return "edu.fact_attendance"
    if {"student_id", "course_id", "assessment_type", "score", "max_score"}.issubset(cols) and has_date:
        return "edu.fact_assessment"
    if {"student_id", "course_id", "submitted"}.issubset(cols) and has_date:
        return "edu.fact_assignment"
    if {"student_id", "full_name", "class"}.issubset(cols):
        return "edu.dim_student"
    if {"course_id", "name"}.issubset(cols):
        return "edu.dim_course"

    # 4) fallback by best match
    best_table, best_score, missing_for_best = None, -1, None
    for table, req in REQUIRED.items():
        req_set = set(rc.lower() for rc in req)
        score = len(cols & req_set)
        if score > best_score:
            best_table, best_score = table, score
            missing_for_best = list(req_set - cols)

    if best_table and best_score > 0:
        if best_table.startswith("edu.fact_") and not has_date:
            raise ValueError(
                f"Cannot detect table (need date_key or dt for fact tables). Columns: {sorted(cols)}"
            )
        return best_table

    req_text = {t: REQUIRED[t] for t in REQUIRED}
    raise ValueError(
        f"Cannot detect target table for '{hint}'. "
        f"Detected columns: {sorted(cols)}; Required: {req_text}"
    )

# =========================
# Public: preview & ingest
# =========================
def peek_files(files: List[Any]) -> List[Dict[str, Any]]:
    """คืนข้อมูลพรีวิวต่อไฟล์/ชีต: คอลัมน์, จำนวนแถว, ตัวอย่าง 5 แถว, ตารางที่เดาได้"""
    result = []
    for f in files:
        fname = getattr(f, "name", "uploaded")
        if fname.lower().endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(f)
            for sheet in xls.sheet_names:
                raw = xls.parse(sheet)
                df0 = _apply_synonyms(_normalize_headers(raw))
                try:
                    table_guess = _detect_table(df0, hint=f"{fname}:{sheet}")
                    # ทำ date_key เฉพาะที่จำเป็นเท่านั้น
                    df = _ensure_date_key_local(df0) if _needs_date(table_guess) else df0
                except Exception as e:
                    table_guess = f"UNKNOWN ({e})"
                    df = df0  # อย่างน้อยยังแสดง preview ได้
                result.append({
                    "source": f"{fname}:{sheet}",
                    "rows": int(len(df)),
                    "columns": list(df.columns),
                    "table_guess": table_guess,
                    "sample": df.head(5).to_dict(orient="records"),
                })
        else:
            raw = pd.read_csv(f)
            df0 = _apply_synonyms(_normalize_headers(raw))
            try:
                table_guess = _detect_table(df0, hint=fname)
                df = _ensure_date_key_local(df0) if _needs_date(table_guess) else df0
            except Exception as e:
                table_guess = f"UNKNOWN ({e})"
                df = df0
            result.append({
                "source": fname,
                "rows": int(len(df)),
                "columns": list(df.columns),
                "table_guess": table_guess,
                "sample": df.head(5).to_dict(orient="records"),
            })
    return result


def ingest_files(ch, files: List[Any], forced_table: str | None = None) -> Dict[str, int]:
    """
    Load CSV/XLSX into ClickHouse edu.* tables.
    - Normalize headers/synonyms
    - Compute date_key from dt **เฉพาะ** ตารางที่ต้องการ (facts/events_raw)
    - Normalize dtypes for ClickHouse
    - รายงานจำนวนแถวต่อ table
    """
    report: Dict[str, int] = {}
    for f in files:
        fname = getattr(f, "name", "uploaded")
        try:
            if fname.lower().endswith((".xlsx", ".xls")):
                xls = pd.ExcelFile(f)
                for sheet in xls.sheet_names:
                    try:
                        raw = xls.parse(sheet)
                        df0 = _apply_synonyms(_normalize_headers(raw))
                        table = forced_table or _detect_table(df0, hint=f"{fname}:{sheet}")
                        df1 = _ensure_date_key_local(df0) if _needs_date(table) else df0
                        df = _normalize_dtypes(table, df1)

                        cols = REQUIRED[table] + [c for c in ALLOWED_EXTRA.get(table, []) if c in df.columns]
                        cols = [c for c in cols if c in df.columns]

                        missing = list(set(REQUIRED[table]) - set(cols))
                        if missing:
                            raise ValueError(f"Missing required columns: {missing}")

                        if df.empty:
                            report[table] = report.get(table, 0) + 0
                            continue

                        ch.insert_df(table, df, cols)
                        report[table] = report.get(table, 0) + int(len(df))

                    except Exception as e_sheet:
                        raise RuntimeError(f"{fname}:{sheet} → {e_sheet}") from e_sheet

            else:
                raw = pd.read_csv(f)
                df0 = _apply_synonyms(_normalize_headers(raw))
                table = forced_table or _detect_table(df0, hint=fname)
                df1 = _ensure_date_key_local(df0) if _needs_date(table) else df0
                df = _normalize_dtypes(table, df1)

                cols = REQUIRED[table] + [c for c in ALLOWED_EXTRA.get(table, []) if c in df.columns]
                cols = [c for c in cols if c in df.columns]

                missing = list(set(REQUIRED[table]) - set(cols))
                if missing:
                    raise ValueError(f"{fname} missing required columns: {missing}")

                if df.empty:
                    report[table] = report.get(table, 0) + 0
                    continue

                ch.insert_df(table, df, cols)
                report[table] = report.get(table, 0) + int(len(df))

        except Exception as e_file:
            raise RuntimeError(f"Upload failed for {fname}: {e_file}") from e_file

    return report


def _needs_date(table: str) -> bool:
    if not table:
        return False
    t = table.lower()
    return t.startswith("edu.fact_") or t.endswith("edu.events_raw") or t.endswith("events_raw")
