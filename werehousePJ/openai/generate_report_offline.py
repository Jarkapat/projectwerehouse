"""
Offline report generator (no API key, no LLM).
- ดึงข้อมูลสรุปจาก ClickHouse view `edu.vw_student_term_summary`
- สร้างรายงานภาษาไทยแบบ rule-based แล้วเขียนออกไฟล์ Markdown

วิธีใช้:
  export CLICKHOUSE_URL=http://localhost:8123
  export CLICKHOUSE_USER=default
  export CLICKHOUSE_PASSWORD=
  python generate_report_offline.py --student_id S001 --term 1-2025 --out report_S001.md
"""

import os, argparse
from clickhouse_connect import get_client

def ch_client():
    url  = os.getenv("CLICKHOUSE_URL", "http://localhost:8123")
    user = os.getenv("CLICKHOUSE_USER", "default")
    pwd  = os.getenv("CLICKHOUSE_PASSWORD", "")
    host = url.split("://")[1].split(":")[0]
    port = int(url.split(":")[-1])
    return get_client(host=host, port=port, username=user, password=pwd)

def get_student(cli, sid):
    res = cli.query(
        "SELECT student_id,name,class FROM edu.dim_student WHERE student_id=%(sid)s",
        parameters={"sid": sid}
    )
    if not res.result_rows:
        return None
    cols = res.column_names
    row  = res.result_rows[0]
    return dict(zip(cols, row))

def get_by_course(cli, sid, term):
    q = """SELECT course_id,assess_ratio,attendance_rate,ontime_rate,avg_assignment_score
           FROM edu.vw_student_term_summary
           WHERE student_id=%(sid)s AND term=%(t)s
           ORDER BY course_id"""
    res = cli.query(q, parameters={"sid": sid, "t": term})
    cols = res.column_names
    return [dict(zip(cols, r)) for r in res.result_rows]

def get_class_stats(cli, term):
    res = cli.query(
        "SELECT avg(attendance_rate) AS attendance_rate_avg "
        "FROM edu.vw_student_term_summary WHERE term=%(t)s",
        parameters={"t": term}
    )
    return float(res.result_rows[0][0]) if res.result_rows and res.result_rows[0][0] is not None else None

def pct(x):
    if x is None: return "-"
    return f"{round(x*100)}%"

def bullet_strengths(row):
    s = []
    if row.get("assess_ratio",0) >= 0.8: s.append("ทำคะแนนการประเมินรวมได้ดีต่อเนื่อง")
    if row.get("attendance_rate",0) >= 0.9: s.append("มีความสม่ำเสมอในการเข้าเรียน")
    if row.get("ontime_rate",0) >= 0.8: s.append("ส่งงานตรงเวลา")
    if not s: s.append("มีพัฒนาการที่สามารถต่อยอดได้ หากรักษาความสม่ำเสมอ")
    return s[:3]

def bullet_opps(row):
    o = []
    if row.get("assess_ratio",1) < 0.7: o.append("ควรทบทวนบทเรียนเพิ่มเติมเพื่อยกระดับคะแนนรวม")
    if row.get("attendance_rate",1) < 0.85: o.append("ควรรักษาอัตราการเข้าเรียนให้สม่ำเสมอมากขึ้น")
    if row.get("ontime_rate",1) < 0.75: o.append("บริหารเวลาเพื่อส่งงานให้ตรงกำหนด")
    if not o: o.append("รักษามาตรฐานปัจจุบันและตั้งเป้าคะแนนที่สูงขึ้น")
    return o[:3]

def render_report(stu, term, by_course, class_avg_att):
    lines = []
    lines.append(f"# รายงานผลการเรียน ({term})")
    lines.append(f"**นักเรียน:** {stu['name']} ({stu['student_id']})  |  **ห้อง:** {stu['class']}")
    if class_avg_att is not None:
        lines.append(f"**อัตราเข้าเรียนเฉลี่ยของชั้นเรียน:** {pct(class_avg_att)}")
    lines.append("")
    for r in by_course:
        lines.append(f"## วิชา {r['course_id']}")
        lines.append(f"- คะแนนรวม (Assess): **{pct(r.get('assess_ratio'))}**")
        lines.append(f"- เข้าเรียน: **{pct(r.get('attendance_rate'))}**")
        lines.append(f"- ส่งงานตรงเวลา: **{pct(r.get('ontime_rate'))}**")
        if r.get("avg_assignment_score") is not None:
            lines.append(f"- คะแนนการบ้านเฉลี่ย: {round(r['avg_assignment_score'],1)}")
        lines.append("")
        lines.append("**จุดแข็ง**")
        for b in bullet_strengths(r): lines.append(f"- {b}")
        lines.append("**โอกาสพัฒนา**")
        for b in bullet_opps(r): lines.append(f"- {b}")
        lines.append("")
    lines.append("_หมายเหตุ: รายงานฉบับนี้สร้างแบบ rule-based โดยไม่ใช้โมเดลภาษา_")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_id", required=True)
    ap.add_argument("--term", required=True)
    ap.add_argument("--out", default="report.md")
    args = ap.parse_args()

    cli = ch_client()
    stu = get_student(cli, args.student_id)
    if not stu:
        raise SystemExit("ไม่พบนักเรียนตามรหัสที่ระบุ")
    by_course = get_by_course(cli, args.student_id, args.term)
    class_avg = get_class_stats(cli, args.term)
    report = render_report(stu, args.term, by_course, class_avg)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"บันทึกรายงานที่ {args.out}")

if __name__ == "__main__":
    main()
