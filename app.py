import io
import re
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Data Analyst Studio", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top:1.25rem; max-width:1500px;}
    .app-title {font-size:2.35rem; font-weight:760; letter-spacing:-0.03em;}
    .app-subtitle {color:#8f96a3; margin-bottom:1.15rem;}
    div[data-testid="stMetric"] {border:1px solid rgba(130,140,155,.22); border-radius:12px; padding:12px 14px;}
    [data-testid="stSidebar"] {border-right:1px solid rgba(125,135,150,.15);}
    </style>
    """,
    unsafe_allow_html=True,
)

NULL_TOKENS = {"", "null", "none", "n/a", "na", "nan", "-", "--", "missing"}
CURRENCY_CODES = r"(?:USD|EUR|GBP|JPY|VND|AUD|CAD)"
CURRENCY_SYMBOLS = r"[$€£¥]"
PERCENT_RE = re.compile(r"^\s*[-+]?\s*[\d.,]+\s*%\s*$")
NUMERIC_RE = re.compile(r"^\s*[-+]?\s*[\d.,]+\s*$")


@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    name = file_name.lower()
    if name.endswith(".csv"):
        last_error = None
        for enc in ("utf-8-sig", "utf-8", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=enc, sep=None, engine="python")
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Không thể đọc CSV: {last_error}")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(file_bytes))
    raise ValueError("Định dạng file chưa được hỗ trợ.")


def snake_case(value: str) -> str:
    s = re.sub(r"[\s\-/]+", "_", str(value).strip())
    s = re.sub(r"[^0-9a-zA-Z_]+", "", s)
    return re.sub(r"_+", "_", s).strip("_").lower() or "column"


def unique_names(columns: List[str]) -> List[str]:
    seen, result = {}, []
    for col in columns:
        n = seen.get(col, 0)
        result.append(col if n == 0 else f"{col}_{n}")
        seen[col] = n + 1
    return result


def normalize_text(series: pd.Series) -> pd.Series:
    out = series.map(lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x)
    return out.map(lambda x: np.nan if isinstance(x, str) and x.lower() in NULL_TOKENS else x)


def parse_number(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    s = re.sub(CURRENCY_CODES, "", s, flags=re.I)
    s = re.sub(CURRENCY_SYMBOLS, "", s)
    s = re.sub(r"\s+", "", s)
    if not s:
        return np.nan
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        last = s.split(",")[-1]
        s = s.replace(".", "").replace(",", ".") if len(last) in (1, 2) else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_mixed_date(series: pd.Series) -> pd.Series:
    try:
        a = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
        b = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=False)
    except TypeError:
        a = pd.to_datetime(series, errors="coerce", dayfirst=True)
        b = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return a if a.notna().sum() >= b.notna().sum() else b


def standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    out, log = df.copy(), []
    old = list(out.columns)
    out.columns = unique_names([snake_case(c) for c in old])
    if old != list(out.columns):
        log.append({"step": "Standardize columns", "column": "(all)", "detail": "Converted names to unique snake_case.", "affected": None})
    for col in out.select_dtypes(include="object").columns:
        before = out[col].copy()
        out[col] = normalize_text(out[col])
        changed = int((before.fillna("__NA__").astype(str) != out[col].fillna("__NA__").astype(str)).sum())
        if changed:
            log.append({"step": "Normalize text", "column": col, "detail": "Trimmed whitespace and standardized null tokens.", "affected": changed})
    return out, log


def detect_semantics(df: pd.DataFrame) -> Dict[str, dict]:
    result = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            result[col] = {"kind": "date", "confidence": 1.0}; continue
        if pd.api.types.is_numeric_dtype(s):
            result[col] = {"kind": "numeric", "confidence": 1.0}; continue
        if s.dtype != "object":
            result[col] = {"kind": "other", "confidence": 1.0}; continue
        sample = s.dropna().astype(str).head(800)
        if len(sample) < 3:
            result[col] = {"kind": "text", "confidence": 1.0}; continue

        name = col.lower()
        id_like = any(k in name for k in ("id", "code", "zip", "postal", "phone"))
        pct_rate = sample.map(lambda x: bool(PERCENT_RE.match(x))).mean()
        if pct_rate >= .85:
            result[col] = {"kind": "percentage_text", "confidence": float(pct_rate)}; continue

        explicit_currency = sample.str.contains(r"[$€£¥]|\b(?:USD|EUR|GBP|JPY|VND|AUD|CAD)\b", case=False, regex=True).mean()
        numeric_like = sample.map(lambda x: pd.notna(parse_number(x))).mean()
        if not id_like and explicit_currency >= .20 and numeric_like >= .90:
            result[col] = {"kind": "currency_text", "confidence": float(numeric_like)}; continue

        numeric_rate = sample.map(lambda x: bool(NUMERIC_RE.match(x))).mean()
        if not id_like and numeric_rate >= .92:
            result[col] = {"kind": "numeric_text", "confidence": float(numeric_rate)}; continue

        date_rate = parse_mixed_date(sample).notna().mean()
        date_name = any(k in name for k in ("date", "time", "created", "updated", "occur", "declar", "birth"))
        if (date_name and date_rate >= .60) or date_rate >= .93:
            result[col] = {"kind": "date_text", "confidence": float(date_rate)}; continue

        result[col] = {"kind": "text", "confidence": 1.0}
    return result


def clean_safe(df: pd.DataFrame, semantics: Dict[str, dict]) -> Tuple[pd.DataFrame, List[dict]]:
    out, log = df.copy(), []
    dup = int(out.duplicated().sum())
    if dup:
        out = out.drop_duplicates().reset_index(drop=True)
        log.append({"step": "Remove duplicates", "column": "(rows)", "detail": "Removed exact duplicates.", "affected": dup})

    for col, meta in semantics.items():
        kind = meta["kind"]
        if kind in ("currency_text", "numeric_text"):
            source = out[col].copy(); parsed = source.map(parse_number)
            failed = int((source.notna() & parsed.isna()).sum())
            out[col] = parsed
            log.append({"step": "Type conversion", "column": col, "detail": f"{kind} → numeric; parse failures={failed}.", "affected": int(source.notna().sum())})
        elif kind == "percentage_text":
            source = out[col].copy(); parsed = source.astype(str).str.replace("%", "", regex=False).map(parse_number) / 100
            parsed[source.isna()] = np.nan; out[col] = parsed
            log.append({"step": "Type conversion", "column": col, "detail": "percentage text → decimal numeric.", "affected": int(source.notna().sum())})
        elif kind == "date_text":
            source = out[col].copy(); parsed = parse_mixed_date(source)
            failed = int((source.notna() & parsed.isna()).sum()); out[col] = parsed
            log.append({"step": "Type conversion", "column": col, "detail": f"text → datetime; parse failures={failed}.", "affected": int(source.notna().sum())})
    return out, log


def add_derived(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    out, log = df.copy(), []
    occur = next((c for c in out.columns if "occur" in c.lower() and pd.api.types.is_datetime64_any_dtype(out[c])), None)
    declar = next((c for c in out.columns if "declar" in c.lower() and pd.api.types.is_datetime64_any_dtype(out[c])), None)
    if occur and declar:
        out["claim_delay_days"] = (out[declar] - out[occur]).dt.days
        log.append({"step": "Derived field", "column": "claim_delay_days", "detail": f"{declar} - {occur}", "affected": int(out["claim_delay_days"].notna().sum())})
    damage = next((c for c in out.columns if "damage" in c.lower() and pd.api.types.is_numeric_dtype(out[c])), None)
    indemn = next((c for c in out.columns if ("indemn" in c.lower() or "settle" in c.lower()) and pd.api.types.is_numeric_dtype(out[c])), None)
    if damage and indemn:
        out["indemnification_rate"] = out[indemn] / out[damage].replace(0, np.nan)
        log.append({"step": "Derived field", "column": "indemnification_rate", "detail": f"{indemn} / {damage}", "affected": int(out["indemnification_rate"].notna().sum())})
    return out, log


def profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "column": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()),
        "missing_%": round(float(df[c].isna().mean() * 100), 2), "unique": int(df[c].nunique(dropna=True)),
        "sample": " | ".join(df[c].dropna().astype(str).head(3).tolist())
    } for c in df.columns])


def issues(df: pd.DataFrame, semantics: Dict[str, dict]) -> pd.DataFrame:
    rows, n = [], max(len(df), 1)
    dup = int(df.duplicated().sum())
    if dup:
        rows.append({"severity": "High", "column": "(rows)", "issue": "Exact duplicate rows", "count": dup, "rate": dup/n, "recommended_action": "Remove exact duplicates"})
    mapping = {
        "date_text": ("Date stored as text / mixed formats", "Convert to datetime"),
        "currency_text": ("Currency stored as text", "Convert to numeric"),
        "numeric_text": ("Numeric stored as text", "Convert to numeric"),
        "percentage_text": ("Percentage stored as text", "Convert to decimal numeric"),
    }
    for c in df.columns:
        miss = int(df[c].isna().sum())
        if miss:
            rate = miss/n; severity = "High" if rate >= .4 else ("Medium" if rate >= .1 else "Low")
            rows.append({"severity": severity, "column": c, "issue": "Missing values", "count": miss, "rate": rate, "recommended_action": "Review missing-value policy"})
        kind = semantics[c]["kind"]
        if kind in mapping:
            issue, action = mapping[kind]
            rows.append({"severity": "Medium", "column": c, "issue": issue, "count": None, "rate": semantics[c]["confidence"], "recommended_action": action})
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for a in dates:
        for b in dates:
            if a != b and any(k in a for k in ("occur", "incident", "order", "start")) and any(k in b for k in ("declar", "report", "ship", "end")):
                v = int((df[a].notna() & df[b].notna() & (df[b] < df[a])).sum())
                rows.append({"rule": f"{b} >= {a}", "violations": v, "status": "Pass" if v == 0 else "Review", "note": "Chronological consistency"})
    amount_cols = [c for c in df.select_dtypes(include=np.number).columns if any(k in c.lower() for k in ("amount", "sales", "revenue", "price", "cost", "profit", "premium", "damage", "indemn"))]
    for c in amount_cols:
        v = int((df[c] < 0).sum())
        rows.append({"rule": f"{c} >= 0", "violations": v, "status": "Pass" if v == 0 else "Review", "note": "Non-negative amount"})
    damage = next((c for c in amount_cols if "damage" in c.lower()), None)
    indemn = next((c for c in amount_cols if "indemn" in c.lower()), None)
    if damage and indemn:
        v = int((df[damage].notna() & df[indemn].notna() & (df[indemn] > df[damage])).sum())
        rows.append({"rule": f"{indemn} <= {damage}", "violations": v, "status": "Pass" if v == 0 else "Review", "note": "Candidate insurance rule; confirm policy terms"})
    return pd.DataFrame(rows or [{"rule": "No domain rules detected", "violations": 0, "status": "Info", "note": "Add business-specific rules if needed"}]).drop_duplicates("rule")


def apply_missing(df: pd.DataFrame, policies: Dict[str, str]) -> Tuple[pd.DataFrame, List[dict]]:
    out, log = df.copy(), []
    for c, policy in policies.items():
        if c not in out or policy == "Keep missing": continue
        before = int(out[c].isna().sum())
        if policy == "Fill median" and pd.api.types.is_numeric_dtype(out[c]): out[c] = out[c].fillna(out[c].median())
        elif policy == "Fill 0" and pd.api.types.is_numeric_dtype(out[c]): out[c] = out[c].fillna(0)
        elif policy == "Fill mode":
            m = out[c].mode(dropna=True)
            if len(m): out[c] = out[c].fillna(m.iloc[0])
        elif policy == "Drop rows": out = out[out[c].notna()].copy()
        after = int(out[c].isna().sum()) if c in out else 0
        log.append({"step": "Missing policy", "column": c, "detail": f"{policy}; missing {before} → {after}.", "affected": abs(before-after)})
    return out.reset_index(drop=True), log


def numeric_cols(df): return df.select_dtypes(include=np.number).columns.tolist()
def date_cols(df): return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
def category_cols(df): return [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_bool_dtype(df[c])]


def metric_candidates(df):
    nums = numeric_cols(df); keys = ("sales", "revenue", "profit", "amount", "price", "cost", "quantity", "premium", "damage", "indemn", "value", "rate")
    preferred = [c for c in nums if any(k in c.lower() for k in keys)]
    return preferred + [c for c in nums if c not in preferred and not any(k in c.lower() for k in ("id", "code", "postal", "zip"))]


def dimension_candidates(df):
    max_unique = min(100, max(20, int(len(df)*.25)))
    return [c for c in category_cols(df) if 2 <= df[c].nunique(dropna=True) <= max_unique]


def compact(v):
    if v is None or pd.isna(v): return "—"
    v = float(v)
    if abs(v) >= 1e9: return f"{v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"{v/1e6:.2f}M"
    if abs(v) >= 1e3: return f"{v/1e3:.2f}K"
    return f"{v:,.2f}"


def quick_ask(df, q: str, metric: Optional[str], dim: Optional[str], dcol: Optional[str]) -> str:
    q = q.lower()
    if any(k in q for k in ("missing", "thiếu", "null", "data quality", "chất lượng")):
        m = df.isna().sum(); m = m[m>0].sort_values(ascending=False)
        return "Không phát hiện missing values." if len(m)==0 else "Missing values:\n" + "\n".join(f"- `{c}`: {int(v)} ({v/len(df):.1%})" for c,v in m.head(10).items())
    if any(k in q for k in ("bao nhiêu dòng", "rows", "records", "bản ghi")): return f"Dataset có **{len(df):,}** dòng và **{df.shape[1]}** cột."
    if any(k in q for k in ("tổng", "sum", "total")) and metric: return f"Tổng `{metric}` = **{compact(df[metric].sum())}**."
    if any(k in q for k in ("trung bình", "average", "mean")) and metric: return f"Trung bình `{metric}` = **{compact(df[metric].mean())}**."
    if any(k in q for k in ("top", "cao nhất", "lớn nhất", "mạnh nhất")) and metric and dim:
        g = df.groupby(dim, dropna=False)[metric].sum().sort_values(ascending=False).head(10)
        return f"Top `{dim}` theo `{metric}`:\n" + "\n".join(f"- **{k}**: {compact(v)}" for k,v in g.items())
    if any(k in q for k in ("trend", "xu hướng", "tháng", "mùa vụ")) and metric and dcol:
        m = df[[dcol, metric]].dropna().set_index(dcol).resample("ME")[metric].sum()
        if len(m): return f"Tháng cao nhất: **{m.idxmax().strftime('%Y-%m')}** ({compact(m.max())}); thấp nhất: **{m.idxmin().strftime('%Y-%m')}** ({compact(m.min())})."
    return "Quick Ask hỗ trợ data quality, row count, total/average, top group và trend. Với truy vấn chi tiết hơn, dùng SQL Query."


def run_sql(df, sql):
    con = duckdb.connect(database=":memory:"); con.register("data", df)
    try: return con.execute(sql).df()
    finally: con.close()


def workbook_bytes(df, prof, issue_df, validation, log_df):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
        prof.to_excel(writer, index=False, sheet_name="Data_Profile")
        issue_df.to_excel(writer, index=False, sheet_name="Detected_Issues")
        validation.to_excel(writer, index=False, sheet_name="Validation")
        log_df.to_excel(writer, index=False, sheet_name="Cleaning_Log")
    return out.getvalue()


st.markdown('<div class="app-title">Data Analyst Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Data quality, cleaning, analysis, querying and export in one workspace</div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Dataset")
    uploaded = st.file_uploader("CSV / Excel / Parquet", type=["csv", "xlsx", "xls", "parquet"])

if not uploaded:
    st.info("Upload a dataset to begin.")
    st.stop()

try:
    raw = load_file(uploaded.getvalue(), uploaded.name)
except Exception as exc:
    st.error(f"Không thể đọc file: {exc}"); st.stop()

standardized, log1 = standardize(raw)
semantics_before = detect_semantics(standardized)
issues_before = issues(standardized, semantics_before)
auto_cleaned, log2 = clean_safe(standardized, semantics_before)
auto_cleaned, log3 = add_derived(auto_cleaned)

if "missing_policies" not in st.session_state: st.session_state.missing_policies = {}
missing_columns = [c for c in auto_cleaned.columns if auto_cleaned[c].isna().any()]
policies = {c: st.session_state.missing_policies.get(c, "Keep missing") for c in missing_columns}
final_df, log4 = apply_missing(auto_cleaned, policies)

prof = profile(final_df)
issue_df = issues(final_df, detect_semantics(final_df))
validation = validate(final_df)
log_df = pd.DataFrame(log1 + log2 + log3 + log4)
metrics, dims, dates = metric_candidates(final_df), dimension_candidates(final_df), date_cols(final_df)

with st.sidebar:
    st.divider(); st.subheader("Analysis")
    metric = st.selectbox("Metric", ["(None)"] + metrics); metric = None if metric == "(None)" else metric
    dim = st.selectbox("Dimension", ["(None)"] + dims); dim = None if dim == "(None)" else dim
    dcol = st.selectbox("Date", ["(None)"] + dates); dcol = None if dcol == "(None)" else dcol
    top_n = st.slider("Top N", 5, 25, 10)

m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Rows", f"{len(final_df):,}"); m2.metric("Columns", final_df.shape[1]); m3.metric("Missing", int(final_df.isna().sum().sum()))
m4.metric("Duplicates", int(final_df.duplicated().sum())); m5.metric("Date columns", len(date_cols(final_df))); m6.metric("File size", f"{len(uploaded.getvalue())/(1024*1024):.1f} MB")

tabs = st.tabs(["Overview", "Data Quality", "Cleaning Studio", "Analysis", "Query Data", "Export"])

with tabs[0]:
    st.subheader("Dataset Overview"); st.dataframe(final_df.head(200), use_container_width=True, height=430)
    a,b = st.columns(2)
    with a:
        st.markdown("**Numeric columns**"); st.write(", ".join(numeric_cols(final_df)) or "—")
        st.markdown("**Categorical columns**"); st.write(", ".join(category_cols(final_df)) or "—")
    with b:
        st.markdown("**Date columns**"); st.write(", ".join(date_cols(final_df)) or "—")
        st.markdown("**Derived fields**"); st.write(", ".join(c for c in ("claim_delay_days", "indemnification_rate") if c in final_df.columns) or "—")
    if numeric_cols(final_df): st.dataframe(final_df[numeric_cols(final_df)].describe().T, use_container_width=True)

with tabs[1]:
    st.subheader("Column Profile"); st.dataframe(prof, use_container_width=True, height=420)
    st.subheader("Detected Issues")
    if len(issue_df):
        show = issue_df.copy(); show["rate"] = show["rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else ""); st.dataframe(show, use_container_width=True)
    else: st.success("No outstanding structural issues detected.")
    st.subheader("Business Rule Validation"); st.dataframe(validation, use_container_width=True)

with tabs[2]:
    st.subheader("Cleaning Studio")
    a,b,c,d = st.columns(4); a.metric("Issues before", len(issues_before)); b.metric("Issues after", len(issue_df)); c.metric("Date columns", len(date_cols(final_df))); d.metric("Numeric columns", len(numeric_cols(final_df)))
    st.markdown("#### Automatic safe cleaning")
    st.write("Column standardization, null/text normalization, exact duplicate removal, mixed-date parsing, currency/numeric conversion and relevant derived fields are applied only when detection confidence is high.")
    st.markdown("#### Missing-value policy")
    if not missing_columns: st.success("No missing values require a policy.")
    else:
        selected = {}; cols = st.columns(2)
        for i,c in enumerate(missing_columns):
            with cols[i%2]:
                opts = ["Keep missing", "Fill median", "Fill 0", "Drop rows"] if pd.api.types.is_numeric_dtype(auto_cleaned[c]) else ["Keep missing", "Fill mode", "Drop rows"]
                current = policies.get(c, "Keep missing"); current = current if current in opts else "Keep missing"
                selected[c] = st.selectbox(f"{c} — {int(auto_cleaned[c].isna().sum())} missing ({auto_cleaned[c].isna().mean():.1%})", opts, index=opts.index(current), key=f"policy_{c}")
        if st.button("Apply missing-value policies", type="primary"):
            st.session_state.missing_policies.update(selected); st.rerun()
    st.markdown("#### Cleaning Log")
    if len(log_df): st.dataframe(log_df, use_container_width=True, height=330)
    else: st.info("No cleaning changes were required.")

with tabs[3]:
    st.subheader("Exploratory Analysis")
    if metric:
        s = pd.to_numeric(final_df[metric], errors="coerce"); a,b,c,d = st.columns(4)
        a.metric(f"Total {metric}", compact(s.sum())); b.metric(f"Average {metric}", compact(s.mean())); c.metric(f"Median {metric}", compact(s.median())); d.metric(f"Max {metric}", compact(s.max()))
        st.plotly_chart(px.histogram(final_df, x=metric, nbins=40, title=f"Distribution of {metric}"), use_container_width=True)
    if metric and dim:
        g = final_df.groupby(dim, dropna=False)[metric].sum().sort_values(ascending=False).head(top_n).reset_index()
        fig = px.bar(g, x=metric, y=dim, orientation="h", title=f"Top {top_n} {dim} by {metric}"); fig.update_layout(yaxis={"categoryorder":"total ascending"}); st.plotly_chart(fig, use_container_width=True)
    if metric and dcol:
        monthly = final_df[[dcol,metric]].dropna().set_index(dcol).resample("ME")[metric].sum().reset_index()
        st.plotly_chart(px.line(monthly, x=dcol, y=metric, markers=True, title=f"{metric} over time"), use_container_width=True)
    nums = [c for c in numeric_cols(final_df) if final_df[c].nunique(dropna=True)>1]
    if len(nums)>=2: st.plotly_chart(px.imshow(final_df[nums].corr(numeric_only=True), text_auto=".2f", aspect="auto", title="Correlation Matrix"), use_container_width=True)

with tabs[4]:
    st.subheader("Query Data"); st.markdown("#### Quick Ask")
    if "qa_history" not in st.session_state: st.session_state.qa_history = []
    for msg in st.session_state.qa_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    q = st.chat_input("Ask about the data...")
    if q:
        st.session_state.qa_history.append({"role":"user","content":q}); ans = quick_ask(final_df,q,metric,dim,dcol); st.session_state.qa_history.append({"role":"assistant","content":ans}); st.rerun()
    st.markdown("#### SQL Query"); st.caption("Table name: `data`")
    sql = st.text_area("SQL", value="SELECT * FROM data LIMIT 50", height=150)
    if st.button("Run SQL"):
        try:
            result = run_sql(final_df, sql); st.success(f"{len(result):,} rows returned."); st.dataframe(result, use_container_width=True, height=360)
            st.download_button("Download query result", result.to_csv(index=False).encode("utf-8-sig"), "query_result.csv", "text/csv")
        except Exception as exc: st.error(f"SQL error: {exc}")

with tabs[5]:
    st.subheader("Export")
    st.download_button("Download cleaned CSV", final_df.to_csv(index=False).encode("utf-8-sig"), "cleaned_data.csv", "text/csv", use_container_width=True)
    st.download_button("Download analysis workbook", workbook_bytes(final_df, prof, issue_df, validation, log_df), "data_analyst_output.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    if len(log_df): st.download_button("Download cleaning log", log_df.to_csv(index=False).encode("utf-8-sig"), "cleaning_log.csv", "text/csv", use_container_width=True)
