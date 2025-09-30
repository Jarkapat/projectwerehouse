"""
CSV -> Python -> (Groq LLM optional) -> Charts + Markdown Report

วิธีใช้:
  pip install pandas matplotlib groq python-dotenv
  python genai_report_local_groq.py --data_dir data --student_id S001 --term 1-2025 --out out_report_S001.md

โฟลเดอร์ข้อมูล (อย่างน้อย):
- data/dim_student.csv: student_id,name,class,program,birthdate
- data/dim_course.csv:  course_id,name,credit
- data/dim_date.csv:    date_key,dt,week,term,year
- data/fact_assessment.csv: student_id,course_id,date_key,assessment_type,score,max_score
- data/fact_attendance.csv: student_id,course_id,date_key,present(0/1)
- data/fact_assignment.csv: student_id,course_id,date_key,submitted(0/1),score(optional)

หมายเหตุ:
- ถ้ามีตัวแปรแวดล้อม GROQ_API_KEY จะเรียก LLM (โมเดลเริ่มต้น: llama3-8b-8192)
- ถ้าไม่มี GROQ_API_KEY จะสรุปผลแบบ rule-based อัตโนมัติ (ไม่ใช้ AI)
"""

import os, argparse, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- helpers -----------------
def load_csv(p):
    return pd.read_csv(p) if Path(p).exists() else pd.DataFrame()

def pct(x, digits=0):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "-"
    return f"{round(float(x)*100, digits)}%"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rule_strengths(r):
    s=[]
    if r.get("assess_ratio",0)>=0.8: s.append("ทำคะแนนรวมของวิชาได้ดีต่อเนื่อง")
    if r.get("attendance_rate",0)>=0.9: s.append("มีความสม่ำเสมอในการเข้าเรียน")
    if r.get("ontime_rate",0)>=0.8: s.append("ส่งงานตรงเวลา")
    if not s: s.append("มีพัฒนาการต่อเนื่อง หากรักษาความสม่ำเสมอ")
    return s[:3]

def rule_opps(r):
    o=[]
    if r.get("assess_ratio",1)<0.7: o.append("ควรทบทวนบทเรียนเพิ่มเติมเพื่อยกระดับคะแนนรวม")
    if r.get("attendance_rate",1)<0.85: o.append("ควรรักษาอัตราการเข้าเรียนให้สม่ำเสมอมากขึ้น")
    if r.get("ontime_rate",1)<0.75: o.append("บริหารเวลาเพื่อส่งงานให้ตรงกำหนด")
    if not o: o.append("รักษามาตรฐานปัจจุบันและตั้งเป้าคะแนนที่สูงขึ้น")
    return o[:3]

def provenance(md_list, mode, model=None, temp=None):
    md_list.append("> **Report provenance**")
    md_list.append(f"> mode: **{mode}**" + (f" • model: `{model}` • temperature: {temp}" if model else ""))
    md_list.append(f"> generated_at: {datetime.now().isoformat(timespec='seconds')}")
    md_list.append("")

def try_groq_narrative(payload: dict, model="llama3-8b-8192", temperature=0.2):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, False, "GROQ_API_KEY not set"
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        system_prompt = (
            "คุณเป็นครูประจำชั้นที่เข้มงวดแต่ให้กำลังใจ สรุปผลการเรียนจากข้อมูลที่ให้มา "
            "อ้างอิงเฉพาะข้อมูลจริงจาก payload ห้ามสมมุติข้อมูลใหม่ "
            "ระบุเปอร์เซ็นต์ชัด แจกแจงจุดแข็งและโอกาสพัฒนาอย่างละ 2–3 ข้อ "
            "ปิดท้ายคำแนะนำเฉพาะบุคคล 2 ข้อ"
        )
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", model),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content, True, None
    except Exception as e:
        return f"_[Groq LLM error → fallback to rule-based | {e}]_", False, str(e)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--student_id", required=True)
    ap.add_argument("--term", required=True)
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--force_llm", action="store_true", help="Force LLM even if no key")
    ap.add_argument("--force_rule", action="store_true", help="Force rule-based even if key exists")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path("out"); ensure_dir(out_dir)

    # Load CSVs
    dim_student = load_csv(data_dir/"dim_student.csv")
    dim_course  = load_csv(data_dir/"dim_course.csv")
    dim_date    = load_csv(data_dir/"dim_date.csv")
    fa = load_csv(data_dir/"fact_assessment.csv")
    ft = load_csv(data_dir/"fact_attendance.csv")
    fs = load_csv(data_dir/"fact_assignment.csv")

    if dim_student.empty:
        raise SystemExit("ไม่พบ dim_student.csv")
    if fa.empty and ft.empty and fs.empty:
        raise SystemExit("ไม่พบข้อมูล fact_* (assessment/attendance/assignment)")

    # Student info
    stu = dim_student.loc[dim_student["student_id"]==args.student_id, ["student_id","name","class"]]
    if stu.empty:
        raise SystemExit("ไม่พบนักเรียนตามรหัสที่ระบุ")
    stu = stu.iloc[0].to_dict()

    # Map date_key -> term
    term_map = {}
    if not dim_date.empty and "term" in dim_date.columns and "date_key" in dim_date.columns:
        term_map = dict(zip(dim_date["date_key"].astype(int), dim_date["term"].astype(str)))

    def filter_term(df):
        if df.empty: return df
        if "date_key" not in df.columns: return df
        if term_map:
            df = df.assign(_term=df["date_key"].astype(int).map(term_map))
            return df.loc[df["_term"]==args.term].drop(columns=["_term"])
        else:
            return df

    fa = filter_term(fa)
    ft = filter_term(ft)
    fs = filter_term(fs)

    # Aggregate
    agg_list = []
    if not fa.empty:
        g = fa.groupby(["student_id","course_id"], as_index=False).agg(
            score_sum=("score","sum"),
            max_sum=("max_score","sum")
        )
        g["assess_ratio"] = np.where(g["max_sum"]>0, g["score_sum"]/g["max_sum"], np.nan)
        agg_list.append(g[["student_id","course_id","assess_ratio"]])
    if not ft.empty:
        ft2 = ft.copy()
        ft2["present"] = ft2["present"].astype(float)
        g = ft2.groupby(["student_id","course_id"], as_index=False)["present"].mean()
        g = g.rename(columns={"present":"attendance_rate"})
        agg_list.append(g)
    if not fs.empty:
        fs2 = fs.copy()
        fs2["submitted"] = fs2["submitted"].astype(float)
        g = fs2.groupby(["student_id","course_id"], as_index=False).agg(
            ontime_rate=("submitted","mean"),
            avg_assignment_score=("score","mean")
        )
        agg_list.append(g)

    if not agg_list:
        raise SystemExit("ไม่มีข้อมูลเพียงพอในการสรุป")

    from functools import reduce
    merged = reduce(lambda l,r: pd.merge(l,r,on=["student_id","course_id"],how="outer"), agg_list)
    merged = merged.loc[merged["student_id"]==args.student_id].copy()
    if not merged.empty and not dim_course.empty:
        merged = merged.merge(dim_course[["course_id","name"]], on="course_id", how="left")

    # ---- Charts ----
    def bar_save(x, y, title, fname):
        plt.figure()
        plt.bar(x, y)
        plt.title(title)
        plt.xticks(rotation=0)
        plt.tight_layout()
        outp = out_dir/fname
        plt.savefig(outp, dpi=150)
        plt.close()
        return outp

    x_labels = merged["course_id"].astype(str).tolist()
    charts = []
    if "assess_ratio" in merged:
        y = (merged["assess_ratio"]*100).tolist()
        charts.append(("assess", bar_save(x_labels, y, "Assessment %", "assess.png")))
    if "attendance_rate" in merged:
        y = (merged["attendance_rate"]*100).tolist()
        charts.append(("attendance", bar_save(x_labels, y, "Attendance %", "attendance.png")))
    if "ontime_rate" in merged:
        y = (merged["ontime_rate"]*100).tolist()
        charts.append(("ontime", bar_save(x_labels, y, "On-time Submission %", "ontime.png")))

    class_att = merged["attendance_rate"].mean() if "attendance_rate" in merged and not merged["attendance_rate"].isna().all() else None

    payload = {
        "student": stu,
        "term": args.term,
        "by_course": merged[["course_id","assess_ratio","attendance_rate","ontime_rate","avg_assignment_score"]].to_dict("records"),
        "class_stats": {"attendance_rate_avg": float(class_att) if class_att is not None else None}
    }

    # Decide mode
    use_llm_env = bool(os.getenv("GROQ_API_KEY"))
    if args.force_rule:
        use_llm = False
    elif args.force_llm:
        use_llm = True
    else:
        use_llm = use_llm_env

    model_name = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    temperature = 0.2

    narrative = None
    mode_label = "rule-based"
    if use_llm:
        llm_text, used_llm, llm_err = try_groq_narrative(payload, model=model_name, temperature=temperature)
        print(f"[LLM] attempting Groq (model={model_name}, temp={temperature}) … used_llm={used_llm}, err={llm_err}")
        if used_llm:
            narrative = llm_text + f"\n\n_{'Generated by Groq • model=' + model_name + f' • temp={temperature}'}_"
            mode_label = "groq-llm"

    if narrative is None:
        paras = []
        paras.append(f"สรุปผลการเรียนของ {stu['name']} ({stu['student_id']}) ห้อง {stu['class']} ในเทอม {args.term}")
        for _, row in merged.iterrows():
            r = {
                "course_id": row.get("course_id"),
                "assess_ratio": row.get("assess_ratio"),
                "attendance_rate": row.get("attendance_rate"),
                "ontime_rate": row.get("ontime_rate"),
                "avg_assignment_score": row.get("avg_assignment_score"),
            }
            strengths = rule_strengths(r)
            opps = rule_opps(r)
            paras.append(
                f"วิชา {r['course_id']}: คะแนนรวม {pct(r['assess_ratio'])}, "
                f"เข้าเรียน {pct(r['attendance_rate'])}, ส่งงานตรงเวลา {pct(r['ontime_rate'])}.\n"
                f"- จุดแข็ง: " + "; ".join(strengths) + "\n"
                f"- โอกาสพัฒนา: " + "; ".join(opps)
            )
        narrative = "\n\n".join(paras)

    # Markdown output
    md = []
    provenance(md, mode=mode_label, model=(model_name if mode_label=="groq-llm" else None), temp=(temperature if mode_label=="groq-llm" else None))
    md.append(f"# รายงานผลการเรียน ({args.term})")
    md.append(f"**นักเรียน:** {stu['name']} ({stu['student_id']})  |  **ห้อง:** {stu['class']}")
    if class_att is not None:
        md.append(f"**อัตราเข้าเรียนเฉลี่ยโดยประมาณ:** {pct(class_att)}")
    md.append("")
    for tag, p in charts:
        md.append(f"![{tag}]({(p).as_posix()})")
    md.append("")
    md.append(narrative)
    if mode_label=="rule-based":
        md.append("\n_หมายเหตุ: สร้างสรุปแบบ rule-based (ไม่ได้ใช้โมเดลภาษา)_")

    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print(f"✅ สร้างรายงานแล้ว: {args.out}")
    print(f"🖼️ รูปกราฟอยู่ในโฟลเดอร์: {out_dir}/")

if __name__ == "__main__":
    main()
