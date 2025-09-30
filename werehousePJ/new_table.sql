CREATE TABLE IF NOT EXISTS edu.events_raw (
  student_id String,
  course_id  String,
  date_key   UInt32,   -- หรือปล่อยว่าง แล้วส่ง dt มาให้ ingest แปลงเป็น date_key
  kind       Enum8('attendance' = 1, 'assessment' = 2, 'assignment' = 3),
  present            Nullable(UInt8),
  submitted          Nullable(UInt8),
  assessment_type    Nullable(String),
  score              Nullable(Float32),
  max_score          Nullable(Float32),
  note               Nullable(String)
) ENGINE = MergeTree ORDER BY (student_id, course_id, date_key, kind);

CREATE MATERIALIZED VIEW IF NOT EXISTS edu.mv_events_to_attendance
TO edu.fact_attendance AS
SELECT student_id, course_id, date_key, toUInt8(coalesce(present, 0)) AS present
FROM edu.events_raw WHERE kind = 'attendance';

CREATE MATERIALIZED VIEW IF NOT EXISTS edu.mv_events_to_assessment
TO edu.fact_assessment AS
SELECT student_id, course_id, date_key,
       coalesce(assessment_type, 'quiz') AS assessment_type,
       toFloat32(coalesce(score, 0))     AS score,
       toFloat32(coalesce(max_score, 0)) AS max_score
FROM edu.events_raw WHERE kind = 'assessment';

CREATE MATERIALIZED VIEW IF NOT EXISTS edu.mv_events_to_assignment
TO edu.fact_assignment AS
SELECT student_id, course_id, date_key,
       toUInt8(coalesce(submitted, 0)) AS submitted,
       toFloat32(score)                AS score
FROM edu.events_raw WHERE kind = 'assignment';
