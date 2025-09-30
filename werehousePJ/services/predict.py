# services/predict.py (no dim_date)
import os, json
import pandas as pd
from textwrap import dedent

# --- helper: derived term expression (ClickHouse SQL snippet) ---
TERM_SQL_EXPR = """
    if(
        toMonth(toDate(parseDateTimeBestEffortOrNull(toString({alias}.date_key)))) BETWEEN 1 AND 6,
        concat('1-', toString(toYear(toDate(parseDateTimeBestEffortOrNull(toString({alias}.date_key)))))),
        concat('2-', toString(toYear(toDate(parseDateTimeBestEffortOrNull(toString({alias}.date_key))))))
    )
"""

# ---------- LLM helper (Groq) ----------
def _llm_summarize(payload: dict, model: str | None = None, temperature: float = 0.2):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, False, "no_api_key"
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        use_model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        system_prompt = (
            "คุณเป็นครูประจำชั้นที่เข้มงวดแต่ให้กำลังใจ สรุปผลการเรียนจาก payload ที่ให้ "
            "อ้างอิงเฉพาะตัวเลขจริง ห้ามสมมุติ ระบุเปอร์เซ็นต์ชัด แจกแจงจุดแข็งและโอกาสพัฒนา 2–3 ข้อ "
            "และปิดท้ายคำแนะนำเฉพาะบุคคล 2 ข้อ"
        )
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content, True, use_model
    except Exception as e:
        return f"_[Groq LLM error → fallback | {e}]_", False, str(e)

# ---------- Rule-based fallback ----------
def _rule_based_summary(student_name: str, term: str, per_course: pd.DataFrame) -> str:
    def pct(v):
        try:
            if pd.isna(v): return "-"
        except Exception:
            pass
        try:
            return f"{round(float(v)*100)}%"
        except Exception:
            return "-"
    lines = [f"**Summary for {student_name} — {term}**", ""]
    if not per_course.empty:
        if "assess_ratio" in per_course and per_course["assess_ratio"].notna().any():
            overall = per_course["assess_ratio"].mean()
            lines.append(f"- ภาพรวมคะแนน ~ **{round(float(overall)*4, 2)}/4.0** (proxy)")
        if "attendance_rate" in per_course and per_course["attendance_rate"].notna().any():
            lines.append(f"- อัตราเข้าเรียนเฉลี่ย **{pct(per_course['attendance_rate'].mean())}**")
        if "ontime_rate" in per_course and per_course["ontime_rate"].notna().any():
            lines.append(f"- ส่งงานตรงเวลาเฉลี่ย **{pct(per_course['ontime_rate'].mean())}**")
        lines.append("")
        for _, r in per_course.fillna(0).iterrows():
            lines.append(
                f"• {r['course_id']}: คะแนน {pct(r.get('assess_ratio'))}, "
                f"เข้าเรียน {pct(r.get('attendance_rate'))}, ส่งงานตรงเวลา {pct(r.get('ontime_rate'))}"
            )
    else:
        lines.append("_ยังไม่มีข้อมูลเพียงพอสำหรับสรุปต่อวิชา_")
    return "\n".join(lines)

# ---------- Main: build features & summarize ----------
def generate_prediction(ch, student_id: str, term: str):
    # META
    df_meta = ch.df(
        f"SELECT student_id, full_name, class FROM edu.dim_student "
        f"WHERE lowerUTF8(trim(student_id))=lowerUTF8(trim('{student_id}')) LIMIT 1"
    )
    student_name = df_meta.iloc[0]["full_name"] if not df_meta.empty else student_id

    # SCORES (term derived from a.date_key)
    df_scores = ch.df(f"""
        SELECT a.course_id, a.assessment_type, sum(a.score) as score, sum(a.max_score) as max_score
        FROM edu.fact_assessment a
        WHERE lowerUTF8(trim(a.student_id))=lowerUTF8(trim('{student_id}'))
          AND {TERM_SQL_EXPR.format(alias='a')} = '{term}'
        GROUP BY a.course_id, a.assessment_type
        ORDER BY a.course_id, a.assessment_type
    """)

    # ATT timeline (date from t.date_key)
    df_att = ch.df(f"""
        SELECT
          toDate(parseDateTimeBestEffortOrNull(toString(t.date_key))) AS date,
          t.present
        FROM edu.fact_attendance t
        WHERE lowerUTF8(trim(t.student_id))=lowerUTF8(trim('{student_id}'))
          AND {TERM_SQL_EXPR.format(alias='t')} = '{term}'
        ORDER BY date
    """)

    # ATT per course
    df_att_course = ch.df(f"""
        SELECT t.course_id, avg(if(t.present=1, 1.0, 0.0)) AS attendance_rate
        FROM edu.fact_attendance t
        WHERE lowerUTF8(trim(t.student_id))=lowerUTF8(trim('{student_id}'))
          AND {TERM_SQL_EXPR.format(alias='t')} = '{term}'
        GROUP BY t.course_id
        ORDER BY t.course_id
    """)

    # ASSIGN per course
    df_assign = ch.df(f"""
        SELECT s.course_id, avg(s.submitted) as ontime_rate, avg(s.score) as avg_assignment_score
        FROM edu.fact_assignment s
        WHERE lowerUTF8(trim(s.student_id))=lowerUTF8(trim('{student_id}'))
          AND {TERM_SQL_EXPR.format(alias='s')} = '{term}'
        GROUP BY s.course_id
        ORDER BY s.course_id
    """)

    # Assess ratio per course
    if df_scores.empty:
        df_ratio = pd.DataFrame(columns=["course_id", "assess_ratio"])
    else:
        g = df_scores.groupby("course_id", as_index=False).agg(
            score=("score","sum"),
            max_score=("max_score","sum"),
        )
        g["assess_ratio"] = (g["score"]/g["max_score"]).where(g["max_score"]>0)
        df_ratio = g[["course_id","assess_ratio"]]

    # Merge per_course from all sources
    ids = set()
    def _ids(df, col="course_id"):
        if df is None or df.empty or col not in df.columns: return set()
        return set(df[col].dropna().astype(str).unique())
    ids |= _ids(df_ratio)
    ids |= _ids(df_assign)
    ids |= _ids(df_att_course)

    per_course = pd.DataFrame({"course_id": sorted(ids)}) if ids else pd.DataFrame(columns=["course_id"])
    if not df_ratio.empty:      per_course = per_course.merge(df_ratio, on="course_id", how="left")
    if not df_assign.empty:     per_course = per_course.merge(df_assign, on="course_id", how="left")
    if not df_att_course.empty: per_course = per_course.merge(df_att_course, on="course_id", how="left")

    # Overall attendance (timeline-based)
    overall_att = None if df_att.empty else df_att["present"].astype(int).mean()

    # Fallback: if no per-course attendance but we have timeline → fill with overall_att
    if ("attendance_rate" not in per_course.columns or per_course["attendance_rate"].isna().all()) and overall_att is not None:
        if per_course.empty:
            per_course = pd.DataFrame([{"course_id": "(all)", "attendance_rate": overall_att}])
        else:
            per_course["attendance_rate"] = overall_att

    payload = {
        "student": {"student_id": student_id, "full_name": student_name},
        "term": term,
        "per_course": per_course.to_dict("records"),
        "overall": {"attendance_rate": float(overall_att) if overall_att is not None else None},
    }

    text, used_llm, model_used = _llm_summarize(payload)
    if not used_llm:
        text = _rule_based_summary(student_name, term, per_course)
        header = dedent(f"> **Report provenance**\n> mode: **rule-based**\n\n**{student_name}** — {term}\n")
    else:
        header = dedent(f"> **Report provenance**\n> mode: **groq-llm** • model: `{model_used}`\n\n**{student_name}** — {term}\n")

    # stats for debugging
    stats = {
        "scores_rows": int(len(df_scores)),
        "att_timeline_rows": int(len(df_att)),
        "att_course_rows": int(len(df_att_course)),
        "assign_rows": int(len(df_assign)),
        "per_course_rows": int(len(per_course)),
        "overall_att": float(overall_att) if overall_att is not None else None,
    }

    return {
        "header_md": header,
        "summary_md": text,
        "df_scores": df_scores,
        "df_att": df_att,
        "df_assign": df_assign,
        "per_course": per_course,
        "df_att_course": df_att_course,
        "stats": stats,
    }
