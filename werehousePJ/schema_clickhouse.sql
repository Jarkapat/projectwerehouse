CREATE DATABASE IF NOT EXISTS edu;

CREATE TABLE IF NOT EXISTS edu.dim_student
(
  student_id String, name String, class String, program String, birthdate Nullable(Date)
) ENGINE=MergeTree ORDER BY (student_id);

CREATE TABLE IF NOT EXISTS edu.dim_course
(
  course_id String, name String, credit Int32
) ENGINE=MergeTree ORDER BY (course_id);

CREATE TABLE IF NOT EXISTS edu.dim_date
(
  date_key UInt32, dt Date, week UInt16, term String, year UInt16
) ENGINE=MergeTree ORDER BY (date_key);

CREATE TABLE IF NOT EXISTS edu.fact_assessment
(
  student_id String, course_id String, date_key UInt32,
  assessment_type LowCardinality(String), score Float64, max_score Float64
) ENGINE=MergeTree ORDER BY (student_id, course_id, date_key);

CREATE TABLE IF NOT EXISTS edu.fact_attendance
(
  student_id String, course_id String, date_key UInt32, present UInt8
) ENGINE=MergeTree ORDER BY (student_id, course_id, date_key);

CREATE TABLE IF NOT EXISTS edu.fact_assignment
(
  student_id String, course_id String, date_key UInt32, submitted UInt8, score Nullable(Float64)
) ENGINE=MergeTree ORDER BY (student_id, course_id, date_key);

-- มุมมองสรุปตัวเลขต่อ นักเรียน–วิชา–เทอม
CREATE OR REPLACE VIEW edu.vw_student_term_summary AS
WITH assess AS (
  SELECT a.student_id, a.course_id, d.term,
         round(sum(a.score)/nullIf(sum(a.max_score),0),4) AS assess_ratio
  FROM edu.fact_assessment a INNER JOIN edu.dim_date d ON d.date_key=a.date_key
  GROUP BY a.student_id, a.course_id, d.term
),
att AS (
  SELECT t.student_id, t.course_id, d.term,
         avg(if(t.present=1,1.0,0.0)) AS attendance_rate
  FROM edu.fact_attendance t INNER JOIN edu.dim_date d ON d.date_key=t.date_key
  GROUP BY t.student_id, t.course_id, d.term
),
asn AS (
  SELECT s.student_id, s.course_id, d.term,
         avg(if(s.submitted=1,1.0,0.0)) AS ontime_rate,
         avg(coalesce(s.score,0)) AS avg_assignment_score
  FROM edu.fact_assignment s INNER JOIN edu.dim_date d ON d.date_key=s.date_key
  GROUP BY s.student_id, s.course_id, d.term
)
SELECT coalesce(assess.student_id,att.student_id,asn.student_id) AS student_id,
       coalesce(assess.course_id,att.course_id,asn.course_id)     AS course_id,
       coalesce(assess.term,att.term,asn.term)                    AS term,
       assess.assess_ratio, att.attendance_rate, asn.ontime_rate, asn.avg_assignment_score
FROM assess
FULL OUTER JOIN att USING (student_id,course_id,term)
FULL OUTER JOIN asn USING (student_id,course_id,term);
