import os, json, argparse
from pathlib import Path
from clickhouse_connect import get_client
from openai import OpenAI

SYSTEM_PROMPT = """คุณเป็นครูประจำชั้นที่เข้มงวดแต่ให้กำลังใจ...
- อิงเฉพาะข้อมูลที่มี
- แจกแจงจุดแข็ง/โอกาสพัฒนา และคำแนะนำ 2 ข้อ
"""

def ch():
    url  = os.getenv("CLICKHOUSE_URL", "http://localhost:8123")
    user = os.getenv("CLICKHOUSE_USER", "default")
    pwd  = os.getenv("CLICKHOUSE_PASSWORD", "")
    host = url.split("://")[1].split(":")[0]; port = int(url.split(":")[-1])
    return get_client(host=host, port=port, username=user, password=pwd)

def get_student(cli, sid):
    rows = cli.query("SELECT student_id,name,class FROM edu.dim_student WHERE student_id=%(sid)s",
                     parameters={"sid": sid}).named_results()
    return dict(rows[0]) if rows else None

def get_by_course(cli, sid, term):
    q = """SELECT course_id,assess_ratio,attendance_rate,ontime_rate,avg_assignment_score
           FROM edu.vw_student_term_summary WHERE student_id=%(sid)s AND term=%(t)s ORDER BY course_id"""
    return [dict(r) for r in cli.query(q, parameters={"sid": sid, "t": term}).named_results()]

def get_class_stats(cli, term):
    v = cli.query("SELECT avg(attendance_rate) FROM edu.vw_student_term_summary WHERE term=%(t)s",
                  parameters={"t": term}).result_rows[0][0]
    return {"attendance_rate_avg": float(v) if v is not None else None}

def run_llm(payload, model=os.getenv("LLM_MODEL","gpt-4o-mini")):
    client = OpenAI()  # ต้องตั้ง OPENAI_API_KEY ใน env
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_id", required=True)
    ap.add_argument("--term", required=True)
    ap.add_argument("--out", default="report.md")
    args = ap.parse_args()

    cli = ch()
    stu = get_student(cli, args.student_id)
    if not stu: raise SystemExit("student not found")
    by_course = get_by_course(cli, args.student_id, args.term)
    cls = get_class_stats(cli, args.term)
    payload = {"student": stu, "term": args.term, "by_course": by_course, "class_stats": cls}

    text = run_llm(payload)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"saved {args.out}")

if __name__ == "__main__":
    main()
