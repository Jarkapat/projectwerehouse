import os
import subprocess
from pathlib import Path
import streamlit as st
from services.ch_dw import CH
from services.charts import course_bars, trend_presence, rate_bars
from services.util import cache_df, clear_cache
from services.predict import generate_prediction
from services.ingest import ingest_files, peek_files

st.set_page_config(page_title="Student Reports Web", page_icon="🎓", layout="wide")
st.title("🎓 Student Reports Web (ClickHouse, no dim_date)")

# ===== Default connection (no sidebar) =====
HOST = os.getenv("CH_HOST", "localhost")
PORT = int(os.getenv("CH_PORT", 8123))   # HTTP port for clickhouse-connect
USER = os.getenv("CH_USER", "default")
PASSWORD = os.getenv("CH_PASSWORD", "")
DATABASE = os.getenv("CH_DB", "edu")     # default DB

# Small connection banner
st.caption(f"Connected to ClickHouse at **{HOST}:{PORT}**, DB **{DATABASE}** (user: {USER})")

# Keep single CH in session
if "ch" not in st.session_state:
    st.session_state.ch = CH(host=HOST, port=PORT, user=USER, password=PASSWORD, database=DATABASE)
ch: CH = st.session_state.ch

# ---- Tabs (Data Entry removed) ----
TAB_DASH, TAB_REPORTS, TAB_UPLOAD, TAB_DOCTOR = st.tabs(
    ["📊 Dashboard", "📝 Reports (Generate)", "📤 Upload & Predict", "🩺 Data Doctor"]
)

# ---------- Helper: term filter (derived from date_key) ----------
# rule: Jan-Jun -> '1-YYYY', Jul-Dec -> '2-YYYY'
TERM_SQL_EXPR = """
    if(
        toMonth(toDate(parseDateTimeBestEffortOrNull(toString({alias}.date_key)))) BETWEEN 1 AND 6,
        concat('1-', toString(toYear(toDate(parseDateTimeBestEffortOrNull(toString({alias}.date_key)))))),
        concat('2-', toString(toYear(toDate(parseDateTimeBestEffortOrNull(toString({alias}.date_key))))))
    )
"""

# =======================
# 📊 Tab: Dashboard
# =======================
with TAB_DASH:
    st.subheader("📊 Dashboard (term derived from date_key)")
    view_mode = st.radio("มุมมอง", ["ตามนักเรียน", "ทั้งห้อง"], horizontal=True, key="dash_view_mode")
    dash_term = st.text_input("เทอม (เช่น 1-2025)", value="1-2025", key="dash_term")

    if view_mode == "ตามนักเรียน":
        df_students = cache_df(
            "students",
            lambda: ch.df("SELECT student_id, full_name, class FROM edu.dim_student ORDER BY student_id")
        )
        if df_students.empty:
            st.info("ยังไม่มีรายชื่อนักเรียน")
        else:
            dash_sid = st.selectbox("เลือกนักเรียน", df_students["student_id"].tolist(), key="dash_sid")

            # Scores per course (derived term from a.date_key)
            q_scores = f"""
                SELECT a.course_id, a.assessment_type,
                       sum(a.score) AS score, sum(a.max_score) AS max_score
                FROM edu.fact_assessment a
                WHERE lowerUTF8(trim(a.student_id)) = lowerUTF8(trim('{dash_sid}'))
                  AND {TERM_SQL_EXPR.format(alias='a')} = '{dash_term}'
                GROUP BY a.course_id, a.assessment_type
                ORDER BY a.course_id, a.assessment_type
            """
            df_scores = ch.df(q_scores)

            # Attendance timeline (date from t.date_key)
            q_att = f"""
                SELECT
                  toDate(parseDateTimeBestEffortOrNull(toString(t.date_key))) AS date,
                  t.present
                FROM edu.fact_attendance t
                WHERE lowerUTF8(trim(t.student_id)) = lowerUTF8(trim('{dash_sid}'))
                  AND {TERM_SQL_EXPR.format(alias='t')} = '{dash_term}'
                ORDER BY date
            """
            df_att = ch.df(q_att)

            st.markdown("##### คะแนนต่อวิชา")
            st.altair_chart(course_bars(df_scores), use_container_width=True)

            st.markdown("##### การเข้าเรียนตามวัน")
            st.altair_chart(trend_presence(df_att), use_container_width=True)

            st.markdown("##### ตารางดิบ")
            c1, c2 = st.columns(2)
            with c1: st.dataframe(df_scores, use_container_width=True)
            with c2: st.dataframe(df_att, use_container_width=True)

    else:
        dash_class = st.text_input("ห้อง (class)", value="1A", key="dash_class")

        # ห้องเรียนรวม (match by class, term derived from each table's date_key)
        q_sum = f"""
            WITH att AS (
              SELECT
                t.student_id,
                avg( if(t.present=1, 1.0, 0.0) ) AS avg_att
              FROM edu.fact_attendance t
              WHERE {TERM_SQL_EXPR.format(alias='t')} = '{dash_term}'
              GROUP BY t.student_id
            )
            SELECT
               s.class,
               a.course_id,
               sum(a.score) AS score,
               sum(a.max_score) AS max_score,
               avg(att.avg_att) AS avg_att_rate
            FROM edu.fact_assessment a
            INNER JOIN edu.dim_student s ON s.student_id = a.student_id
            LEFT JOIN att ON att.student_id = a.student_id
            WHERE {TERM_SQL_EXPR.format(alias='a')} = '{dash_term}'
              AND lowerUTF8(trim(s.class)) = lowerUTF8(trim('{dash_class}'))
            GROUP BY s.class, a.course_id
            ORDER BY a.course_id
        """
        df_sum = ch.df(q_sum)

        st.markdown("##### ภาพรวมทั้งห้อง")
        st.altair_chart(
            course_bars(df_sum.rename(columns={"score": "score", "max_score": "max_score"})),
            use_container_width=True
        )
        st.dataframe(df_sum, use_container_width=True)

# =======================
# 📝 Tab: Reports (external script)
# =======================
with TAB_REPORTS:
    st.subheader("📝 Generate Markdown Report (uses existing script)")
    st.caption("สคริปต์ภายนอกจะอ่าน CSV ของคุณเอง (ไม่ได้อ่านจาก DB โดยตรง)")

    rep_sid = st.text_input("student_id", value="S001", key="rep_sid")
    rep_term = st.text_input("term", value="1-2025", key="rep_term")

    colx, coly = st.columns([2, 1])
    with colx:
        rep_data_dir = st.text_input("Data directory (CSV)", value="data", key="rep_data_dir")
        rep_out_path = st.text_input("Output .md", value=f"out_report_{rep_sid}.md", key="rep_out")
    with coly:
        rep_run = st.button("🚀 Run genai_report_local_groq.py", key="rep_run_btn")

    if rep_run:
        script = Path("genai_report_local_groq.py").resolve()
        if not script.exists():
            st.error("ไม่พบ genai_report_local_groq.py ในโฟลเดอร์นี้")
        else:
            cmd = [
                "python", str(script), "--data_dir", rep_data_dir,
                "--student_id", rep_sid, "--term", rep_term, "--out", rep_out_path
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                st.code("$ " + " ".join(cmd))
                st.text_area("stdout", proc.stdout, height=150)
                st.text_area("stderr", proc.stderr, height=120)
                if Path(rep_out_path).exists():
                    st.success(f"✅ Generated: {rep_out_path}")
                    st.download_button(
                        "⬇️ Download report",
                        data=Path(rep_out_path).read_text(encoding="utf-8"),
                        file_name=Path(rep_out_path).name,
                    )
            except Exception as e:
                st.error(str(e))

# =======================
# 📤 Tab: Upload & Predict
# =======================
with TAB_UPLOAD:
    st.subheader("📤 Upload & Predict")
    st.caption("อัปโหลด CSV/XLSX แล้วโหลดเข้าฐาน `edu` โดยอัตโนมัติ จากนั้นเลือกนักเรียน/เทอมเพื่อแสดงสรุปบนหน้านี้")

    up_files = st.file_uploader(
        "Upload files (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="up_files"
    )

    if up_files:
        # Preview & detection
        preview = peek_files(up_files)
        st.markdown("**🔎 Preview & Detect**")
        for item in preview:
            with st.expander(f"{item['source']}  • rows={item['rows']}  • table_guess={item['table_guess']}"):
                st.write("columns:", item["columns"])
                st.write(item["sample"])

    st.markdown("**โหมดการจับตาราง**")
    up_mode = st.radio("", ["Auto-detect", "Force target table"], horizontal=True, key="up_mode")
    forced_table = None
    if up_mode == "Force target table":
        forced_table = st.selectbox(
            "เลือกตารางปลายทาง",
            ["edu.dim_student","edu.dim_course",
             "edu.fact_assessment","edu.fact_attendance","edu.fact_assignment"],
            key="up_force_table"
        )

    if st.button("⬆️ Load into ClickHouse", disabled=not up_files, key="up_load_btn"):
        try:
            report = ingest_files(ch, up_files, forced_table=forced_table)
            clear_cache()
            st.success("Loaded!")
            st.json(report)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Upload failed: {e}")

    st.divider()
    st.markdown("### 🔮 Predict inline")
    colp1, colp2, colp3 = st.columns([2,2,1])
    with colp1:
        pred_sid = st.text_input("student_id", value="S001", key="pred_sid")
    with colp2:
        pred_term = st.text_input("term", value="1-2025", key="pred_term")
    with colp3:
        do_pred = st.button("Run Prediction", key="pred_run_btn")

    if do_pred:
        out = generate_prediction(ch, pred_sid, pred_term)
        st.markdown(out["header_md"])
        st.markdown(out["summary_md"])

        st.markdown("**Stats (debug)**")
        st.json(out["stats"])

        st.markdown("#### Charts")
        st.altair_chart(course_bars(out["df_scores"]), use_container_width=True)
        st.altair_chart(trend_presence(out["df_att"]), use_container_width=True)
        st.altair_chart(rate_bars(out["per_course"], "attendance_rate", "Attendance %"), use_container_width=True)
        st.altair_chart(rate_bars(out["per_course"], "ontime_rate", "On-time Submission %"), use_container_width=True)

        st.markdown("#### Tables")
        st.dataframe(out["per_course"], use_container_width=True)

# =======================
# 🩺 Tab: Data Doctor (no dim_date)
# =======================
with TAB_DOCTOR:
    st.subheader("🩺 Data Doctor (derived terms, no dim_date)")
    st.caption("ตรวจเทอมจาก date_key โดยตรง")

    colx, coly, colz = st.columns([2,2,1])
    with colx:
        dd_sid = st.text_input("student_id", value="S001", key="dd_sid")
    with coly:
        dd_term = st.text_input("term", value="1-2025", key="dd_term")
    with colz:
        if st.button("🔄 Hard refresh (clear cache)", key="dd_refresh_btn"):
            clear_cache()
            st.experimental_rerun()

    st.markdown("#### 1) Term distribution (derived from date_key)")
    # Attendance
    sql_term_att = f"""
      SELECT
        {TERM_SQL_EXPR.format(alias='t')} AS term_gen,
        count() AS rows
      FROM edu.fact_attendance t
      GROUP BY term_gen ORDER BY term_gen
    """
    # Assessment
    sql_term_ass = f"""
      SELECT
        {TERM_SQL_EXPR.format(alias='a')} AS term_gen,
        count() AS rows
      FROM edu.fact_assessment a
      GROUP BY term_gen ORDER BY term_gen
    """
    # Assignment
    sql_term_asg = f"""
      SELECT
        {TERM_SQL_EXPR.format(alias='s')} AS term_gen,
        count() AS rows
      FROM edu.fact_assignment s
      GROUP BY term_gen ORDER BY term_gen
    """
    c1, c2, c3 = st.columns(3)
    with c1: st.write("**fact_attendance**"); st.dataframe(ch.df(sql_term_att), use_container_width=True)
    with c2: st.write("**fact_assessment**"); st.dataframe(ch.df(sql_term_ass), use_container_width=True)
    with c3: st.write("**fact_assignment**"); st.dataframe(ch.df(sql_term_asg), use_container_width=True)

    st.markdown("#### 2) แถวล่าสุดของแต่ละ fact (*สำหรับ student/term ที่เลือก*)")
    sql_att_last = f"""
      SELECT
        t.student_id,
        t.course_id,
        toDate(parseDateTimeBestEffortOrNull(toString(t.date_key))) AS dt,
        t.present,
        t.date_key
      FROM edu.fact_attendance t
      WHERE lowerUTF8(trim(t.student_id)) = lowerUTF8(trim('{dd_sid}'))
        AND {TERM_SQL_EXPR.format(alias='t')} = '{dd_term}'
      ORDER BY t.date_key DESC LIMIT 20
    """
    sql_ass_last = f"""
      SELECT
        a.student_id, a.course_id,
        a.assessment_type,
        sum(a.score) score, sum(a.max_score) max_score,
        anyLast(a.date_key) date_key
      FROM edu.fact_assessment a
      WHERE lowerUTF8(trim(a.student_id)) = lowerUTF8(trim('{dd_sid}'))
        AND {TERM_SQL_EXPR.format(alias='a')} = '{dd_term}'
      GROUP BY a.student_id, a.course_id, a.assessment_type
      ORDER BY date_key DESC LIMIT 20
    """
    sql_asg_last = f"""
      SELECT
        s.student_id, s.course_id,
        avg(s.submitted) ontime_rate, avg(s.score) avg_assignment_score,
        anyLast(s.date_key) date_key
      FROM edu.fact_assignment s
      WHERE lowerUTF8(trim(s.student_id)) = lowerUTF8(trim('{dd_sid}'))
        AND {TERM_SQL_EXPR.format(alias='s')} = '{dd_term}'
      GROUP BY s.student_id, s.course_id
      ORDER BY date_key DESC LIMIT 20
    """
    c4, c5, c6 = st.columns(3)
    with c4: st.write("**fact_attendance**"); st.dataframe(ch.df(sql_att_last), use_container_width=True)
    with c5: st.write("**fact_assessment**"); st.dataframe(ch.df(sql_ass_last), use_container_width=True)
    with c6: st.write("**fact_assignment**"); st.dataframe(ch.df(sql_asg_last), use_container_width=True)
