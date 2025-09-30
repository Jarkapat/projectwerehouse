import altair as alt
import pandas as pd

def course_bars(df: pd.DataFrame):
    if df is None or df.empty:
        return alt.Chart(pd.DataFrame({"note": ["no data"]})).mark_text().encode(text="note")
    if "max_score" in df.columns and "score" in df.columns:
        agg = df.groupby("course_id", as_index=False).agg(score=("score", "sum"), max_score=("max_score", "sum"))
        agg["pct"] = (agg["score"] / agg["max_score"]).fillna(0) * 100
        return (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X("course_id:N", title="Course"),
                y=alt.Y("pct:Q", title="% of Max"),
                tooltip=["course_id", "pct"],
            )
            .properties(height=300)
        )
    fallback = df.groupby("course_id", as_index=False).size()
    return alt.Chart(fallback).mark_bar().encode(x="course_id:N", y="size:Q")

def trend_presence(df_att: pd.DataFrame):
    if df_att is None or df_att.empty:
        return alt.Chart(pd.DataFrame({"note": ["no data"]})).mark_text().encode(text="note")

    tmp = df_att.copy()
    if "date" not in tmp.columns and "dt" in tmp.columns:
        tmp["date"] = tmp["dt"]
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"])

    if "present" not in tmp.columns:
        if "status" in tmp.columns:
            tmp["present"] = tmp["status"].astype(str).str.lower().eq("present").astype(int)
        else:
            tmp["present"] = 1
    else:
        try:
            tmp["present"] = tmp["present"].astype(int)
        except Exception:
            tmp["present"] = tmp["present"].astype(str).str.lower().isin(["1", "true", "t", "yes"]).astype(int)

    return (
        alt.Chart(tmp)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("present:Q", title="Present(1)/Absent(0)"),
            tooltip=[alt.Tooltip("date:T", title="date"), alt.Tooltip("present:Q", title="present")],
        )
        .properties(height=250)
    )

def rate_bars(df: pd.DataFrame, field: str, title: str):
    if df is None or df.empty or field not in df.columns:
        return alt.Chart(pd.DataFrame({"note": ["no data"]})).mark_text().encode(text="note")
    tmp = df[["course_id", field]].dropna().copy()
    tmp["pct"] = (tmp[field].astype(float) * 100).clip(0, 100)
    return (
        alt.Chart(tmp)
        .mark_bar()
        .encode(
            x=alt.X("course_id:N", title="Course"),
            y=alt.Y("pct:Q", title=title),
            tooltip=["course_id", "pct"],
        )
        .properties(height=280)
    )
