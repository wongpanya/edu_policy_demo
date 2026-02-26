
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

PRIMARY_BLUE = "#1D4ED8"
PRIMARY_ORANGE = "#F97316"
WHITE = "#FFFFFF"
TEXT_DARK = "#0F172A"

st.set_page_config(page_title="Education Equity Policy Demo (Dataset A)", page_icon="🟦🟧", layout="wide")

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 1.25rem; }}
    .tag {{
        display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px;
        background: {PRIMARY_ORANGE}; color: {WHITE}; font-weight: 700; font-size: 0.85rem;
    }}
    .card {{
        border: 1px solid rgba(15, 23, 42, 0.10); border-radius: 16px;
        padding: 14px 16px; background: {WHITE};
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }}
    .card h3 {{ margin: 0 0 6px 0; color: {TEXT_DARK}; }}
    .card .big {{ font-size: 1.6rem; font-weight: 900; }}
    .muted {{ color: rgba(15, 23, 42, 0.65); }}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_XLSX = Path(__file__).resolve().parent / "Dataset_A_2558_2567.xlsx"

@st.cache_data(show_spinner=False)
def load_dataset_a(xlsx_path: str):
    xlsx_path = Path(xlsx_path)
    macro = pd.read_excel(xlsx_path, sheet_name="macro_targets")
    student = pd.read_excel(xlsx_path, sheet_name="student_year")
    schools = pd.read_excel(xlsx_path, sheet_name="schools")
    return macro, student, schools

def age_band_from_grade(grade_code: str) -> str:
    if isinstance(grade_code, str) and grade_code.startswith("V"):
        return "อาชีวะ (V1–V4)"
    if not isinstance(grade_code, str) or not grade_code.startswith("G"):
        return "ไม่ทราบ"
    g = int(grade_code[1:])
    if 1 <= g <= 3: return "ประถมต้น (G1–G3)"
    if 4 <= g <= 6: return "ประถมปลาย (G4–G6)"
    if 7 <= g <= 9: return "มัธยมต้น (G7–G9)"
    if 10 <= g <= 12: return "มัธยมปลาย (G10–G12)"
    return "ไม่ทราบ"

def safe_mean(x):
    x = pd.to_numeric(x, errors="coerce")
    m = np.nanmean(x)
    return float(m) if np.isfinite(m) else np.nan

def safe_rate(x):
    return float(pd.to_numeric(x, errors="coerce").mean())

def quantile_gap(series, q_hi=0.9, q_lo=0.1):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 10:
        return np.nan
    return float(s.quantile(q_hi) - s.quantile(q_lo))

def get_score_cols(df):
    return [c for c in df.columns if c.startswith("score_")]

def pick_primary_scores(score_cols):
    prefs = ["score_reading", "score_math", "score_literacy", "score_numeracy"]
    chosen = [c for c in prefs if c in score_cols]
    if len(chosen) >= 2: return chosen[:2]
    if len(chosen) == 1:
        others = [c for c in score_cols if c != chosen[0]]
        return chosen + (others[:1] if others else [])
    return score_cols[:2]

st.sidebar.markdown(f"<span class='tag'>Policy Demo</span>", unsafe_allow_html=True)
st.sidebar.write("โหลด Dataset A (xlsx) เพื่อเริ่มวิเคราะห์")

xlsx_path = None
if DEFAULT_XLSX.exists():
    xlsx_path = str(DEFAULT_XLSX)
    st.sidebar.success("พบไฟล์ Dataset_A_2558_2567.xlsx ในโฟลเดอร์แอป")
else:
    st.sidebar.warning("ไม่พบไฟล์ Dataset_A_2558_2567.xlsx ในโฟลเดอร์แอป")

uploaded = st.sidebar.file_uploader("อัปโหลด Dataset_A_2558_2567.xlsx (ถ้าต้องการ)", type=["xlsx"])
if uploaded is not None:
    tmp = Path("/tmp/Dataset_A_uploaded.xlsx")
    tmp.write_bytes(uploaded.getvalue())
    xlsx_path = str(tmp)

if not xlsx_path:
    st.stop()

macro, student, schools = load_dataset_a(xlsx_path)
student["age_band"] = student["grade_code"].astype(str).map(age_band_from_grade)

score_cols = get_score_cols(student)
primary_scores = pick_primary_scores(score_cols)

# Filters
years = sorted(student["academic_year"].dropna().unique().tolist())
sel_years = st.sidebar.multiselect("ปีการศึกษา", years, default=[max(years)] if years else [])
if not sel_years:
    st.warning("กรุณาเลือกอย่างน้อย 1 ปีการศึกษา")
    st.stop()

age_bands = ["ประถมต้น (G1–G3)", "ประถมปลาย (G4–G6)", "มัธยมต้น (G7–G9)", "มัธยมปลาย (G10–G12)", "อาชีวะ (V1–V4)"]
sel_age = st.sidebar.multiselect("ช่วงวัย/ระดับ", age_bands, default=age_bands)

regions = sorted(student["region"].dropna().unique().tolist())
sel_regions = st.sidebar.multiselect("ภูมิภาค", regions, default=regions)

sel_urban = st.sidebar.multiselect("พื้นที่", ["เมือง (urban=1)", "ชนบท/นอกเมือง (urban=0)"], default=["เมือง (urban=1)", "ชนบท/นอกเมือง (urban=0)"])
urban_vals = []
if "เมือง (urban=1)" in sel_urban: urban_vals.append(1)
if "ชนบท/นอกเมือง (urban=0)" in sel_urban: urban_vals.append(0)
if not urban_vals:
    st.warning("กรุณาเลือก urban อย่างน้อย 1 แบบ")
    st.stop()

sel_ses = st.sidebar.multiselect("ฐานะครอบครัว (SES quintile)", [1,2,3,4,5], default=[1,2,3,4,5])
sel_vuln = st.sidebar.multiselect("กลุ่มเปราะบาง", ["ทั่วไป", "เปราะบาง"], default=["ทั่วไป","เปราะบาง"])

df = student.copy()
df = df[df["academic_year"].isin(sel_years)]
df = df[df["age_band"].isin(sel_age)]
df = df[df["region"].isin(sel_regions)]
df = df[df["urban"].isin(urban_vals)]
df = df[df["ses_quintile"].isin(sel_ses)]
if set(sel_vuln) != {"ทั่วไป","เปราะบาง"}:
    df = df[df["vulnerable_flag"].eq(1 if "เปราะบาง" in sel_vuln else 0)]

CATEGORY_MAP = {
    "Socioeconomic Data": ["ses_quintile","vulnerable_flag","region","urban"],
    "Access to Education": ["enrolled","attendance_rate","device_access","internet_access","online_participation_rate"],
    "Learning Outcomes": score_cols,
    "Resources & Budget (proxy)": ["school_id"],
    "Policy & Governance (scenario)": ["device_access","internet_access","attendance_rate","online_participation_rate"],
    "Culture & Attitudes (future data)": [],
}
DIM4_MAP = {
    "ฐานะครอบครัว": ["ses_quintile","vulnerable_flag"],
    "การเข้าถึง": ["enrolled","attendance_rate","device_access","internet_access","online_participation_rate"],
    "คุณภาพการเรียน": score_cols,
    "ทรัพยากรและนโยบาย": ["school_id"],
}

def build_group_table(df_in: pd.DataFrame, group_cols):
    sch = schools[["school_id","school_quality_z","urban"]].drop_duplicates("school_id")
    x = df_in.merge(sch[["school_id","school_quality_z"]], on="school_id", how="left")
    out = x.groupby(group_cols, dropna=False).apply(lambda g: pd.Series({
        "n": len(g),
        "enroll_rate": safe_rate(g["enrolled"]),
        "dropout_rate": safe_rate(g["dropout"]),
        "promotion_rate": safe_rate(g.loc[g["enrolled"].eq(1) & g["dropout"].eq(0), "promoted"]) if len(g) else np.nan,
        "attendance_mean": safe_mean(g["attendance_rate"]),
        "online_part_mean": safe_mean(g["online_participation_rate"]),
        "device_mean": safe_mean(g["device_access"]),
        "internet_mean": safe_mean(g["internet_access"]),
        "school_quality_z_mean": safe_mean(g["school_quality_z"]),
        f"{primary_scores[0]}_mean": safe_mean(g[primary_scores[0]]) if primary_scores else np.nan,
        f"{primary_scores[1]}_mean": safe_mean(g[primary_scores[1]]) if len(primary_scores) > 1 else np.nan,
        "score_gap_p90_p10_primary1": quantile_gap(g[primary_scores[0]]) if primary_scores else np.nan,
        "score_gap_p90_p10_primary2": quantile_gap(g[primary_scores[1]]) if len(primary_scores) > 1 else np.nan,
    })).reset_index()
    return out

def equity_risk_index(row):
    dr = row.get("dropout_rate", 0) or 0
    er = row.get("enroll_rate", 0) or 0
    att = row.get("attendance_mean", 0.9) or 0.9
    dev = row.get("device_mean", 0.5) or 0.5
    s1 = row.get(f"{primary_scores[0]}_mean", np.nan)
    if pd.isna(s1): s1 = 50

    r_dropout = np.clip(dr / 0.10, 0, 2)
    r_enroll  = np.clip((1-er) / 0.05, 0, 2)
    r_access  = np.clip((0.90-att)/0.10, 0, 2) + np.clip((0.50-dev)/0.25, 0, 2)
    r_learn   = np.clip((55 - s1)/15, 0, 2)

    score = 35*r_dropout + 15*r_enroll + 25*r_access + 25*r_learn
    return float(np.clip(score, 0, 100))

st.title("🟦🟧 Demo เว็บ: วิเคราะห์เชิงนโยบายเพื่อลดความเหลื่อมล้ำทางการศึกษา")
st.caption("Dataset A (2558–2567) • แยกช่วงวัย • แยก SES/พื้นที่ • Drill-down ดูรายการ • Policy what-if")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='card'><h3>จำนวนระเบียน</h3><div class='big'>{:,}</div><div class='muted'>ตามตัวกรอง</div></div>".format(len(df)), unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><h3>อัตราเข้าเรียน</h3><div class='big'>{:.1f}%</div><div class='muted'>enrolled</div></div>".format(100*df["enrolled"].mean()), unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'><h3>อัตราออกกลางคัน</h3><div class='big'>{:.2f}%</div><div class='muted'>dropout</div></div>".format(100*df["dropout"].mean()), unsafe_allow_html=True)
with c4:
    s_col = primary_scores[0] if primary_scores else None
    m = df[s_col].mean() if s_col else np.nan
    st.markdown("<div class='card'><h3>คะแนนเฉลี่ย (ตัวแทน)</h3><div class='big'>{:.1f}</div><div class='muted'>{}</div></div>".format(float(m) if pd.notna(m) else 0.0, (s_col or "N/A").replace("score_","")), unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["EnrollScope", "LearnPulse", "PersistPath", "EquityLens Lab"])

with tab1:
    st.subheader("EnrollScope — การเข้าถึงการศึกษาและการอยู่ในระบบ")
    group_cols = ["age_band","region","urban","ses_quintile","vulnerable_flag"]
    gtab = build_group_table(df, group_cols)
    gtab["access_risk"] = (1-gtab["enroll_rate"])*0.4 + (0.95-gtab["attendance_mean"]).clip(lower=0)*0.4 + (0.5-gtab["device_mean"]).clip(lower=0)*0.2
    gtab = gtab.sort_values(["access_risk","n"], ascending=[False,False])

    left, right = st.columns([1,1])
    with left:
        st.markdown("### 📌 กลุ่มที่เข้าถึงยาก (Top 10)")
        st.dataframe(gtab.head(10)[group_cols + ["n","enroll_rate","attendance_mean","device_mean","internet_mean","online_part_mean","access_risk"]], use_container_width=True)
    with right:
        st.markdown("### 🔎 Drill-down")
        options = gtab.head(60).copy()
        options["group_key"] = (
            options["age_band"].astype(str)+" | "+options["region"].astype(str)+" | urban="+options["urban"].astype(str)+
            " | SES="+options["ses_quintile"].astype(str)+" | vuln="+options["vulnerable_flag"].astype(str)
        )
        chosen = st.selectbox("เลือกกลุ่มเพื่อดูรายละเอียด", options["group_key"].tolist(), index=0)
        row = options.loc[options["group_key"]==chosen].iloc[0].to_dict()
        st.json({k: row[k] for k in row if k != "group_key"})

        mask = (
            (df["age_band"]==row["age_band"]) &
            (df["region"]==row["region"]) &
            (df["urban"]==row["urban"]) &
            (df["ses_quintile"]==row["ses_quintile"]) &
            (df["vulnerable_flag"]==row["vulnerable_flag"])
        )
        sample_cols = ["student_id","academic_year","grade_code","enrolled","attendance_rate","device_access","internet_access","online_participation_rate"]
        st.markdown("ตัวอย่างระเบียน (สุ่ม 30 แถว)")
        st.dataframe(df.loc[mask, sample_cols].sample(min(30, int(mask.sum())), random_state=7), use_container_width=True)

with tab2:
    st.subheader("LearnPulse — คุณภาพการเรียนรู้และช่องว่าง")
    score_choice = st.selectbox("เลือกวิชา/คะแนนเพื่อวิเคราะห์", score_cols, index=score_cols.index(primary_scores[0]) if primary_scores and primary_scores[0] in score_cols else 0)
    base = df[df["enrolled"]==1].copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("คะแนนเฉลี่ย", f"{base[score_choice].mean():.1f}")
    with c2:
        q1 = base[base["ses_quintile"]==1][score_choice].mean()
        q5 = base[base["ses_quintile"]==5][score_choice].mean()
        st.metric("ช่องว่าง SES (Q5 - Q1)", f"{(q5-q1):.1f}")
    with c3:
        urb0 = base[base["urban"]==0][score_choice].mean()
        urb1 = base[base["urban"]==1][score_choice].mean()
        st.metric("ช่องว่าง เมือง-ชนบท (urban1 - urban0)", f"{(urb1-urb0):.1f}")

    group_cols = ["age_band","region","urban","ses_quintile","vulnerable_flag"]
    g = base.groupby(group_cols, dropna=False)[score_choice].agg(["mean","count"]).reset_index()
    g = g[g["count"]>=50].sort_values("mean", ascending=True)
    g["group_key"] = (
        g["age_band"].astype(str)+" | "+g["region"].astype(str)+" | urban="+g["urban"].astype(str)+
        " | SES="+g["ses_quintile"].astype(str)+" | vuln="+g["vulnerable_flag"].astype(str)
    )
    st.markdown("### 📌 กลุ่มที่คะแนนต่ำสุด (Top 10)")
    st.dataframe(g.head(10)[group_cols+["count","mean"]], use_container_width=True)

    st.markdown("### 🔎 Drill-down: distribution + access drivers")
    if len(g) > 0:
        chosen = st.selectbox("เลือกกลุ่มเพื่อดูรายละเอียด (คะแนน)", g.head(60)["group_key"].tolist(), index=0)
        r = g.loc[g["group_key"]==chosen].iloc[0]
        mask = (
            (base["age_band"]==r["age_band"]) &
            (base["region"]==r["region"]) &
            (base["urban"]==r["urban"]) &
            (base["ses_quintile"]==r["ses_quintile"]) &
            (base["vulnerable_flag"]==r["vulnerable_flag"])
        )
        colA, colB = st.columns([1,1])
        with colA:
            st.write({
                "n": int(mask.sum()),
                "mean": float(base.loc[mask, score_choice].mean()),
                "p10": float(base.loc[mask, score_choice].quantile(0.10)),
                "p50": float(base.loc[mask, score_choice].quantile(0.50)),
                "p90": float(base.loc[mask, score_choice].quantile(0.90)),
            })
            st.bar_chart(base.loc[mask, score_choice].dropna().clip(0,100), height=220)
        with colB:
            st.write({
                "attendance_mean": float(base.loc[mask, "attendance_rate"].mean()),
                "device_mean": float(base.loc[mask, "device_access"].mean()),
                "internet_mean": float(base.loc[mask, "internet_access"].mean()),
                "online_part_mean": float(base.loc[mask, "online_participation_rate"].mean()),
            })
            st.dataframe(base.loc[mask, ["student_id","grade_code","attendance_rate","device_access","internet_access","online_participation_rate",score_choice]].sample(min(30, int(mask.sum())), random_state=7), use_container_width=True)

with tab3:
    st.subheader("PersistPath — dropout / promotion / ความเสี่ยงสะสม")
    group_cols = ["age_band","region","urban","ses_quintile","vulnerable_flag"]
    gtab = build_group_table(df, group_cols)
    gtab["risk_index"] = gtab.apply(equity_risk_index, axis=1)
    gtab = gtab.sort_values(["risk_index","n"], ascending=[False,False]).reset_index(drop=True)

    left, right = st.columns([1,1])
    with left:
        st.markdown("### 📌 กลุ่มเสี่ยงหลุดระบบสูง (Top 10)")
        st.dataframe(gtab.head(10)[group_cols + ["n","enroll_rate","dropout_rate","promotion_rate","risk_index"]], use_container_width=True)
    with right:
        st.markdown("### 🔎 Drill-down: ดูคะแนน + access")
        options = gtab.head(80).copy()
        options["group_key"] = (
            options["age_band"].astype(str)+" | "+options["region"].astype(str)+" | urban="+options["urban"].astype(str)+
            " | SES="+options["ses_quintile"].astype(str)+" | vuln="+options["vulnerable_flag"].astype(str)
        )
        chosen = st.selectbox("เลือกกลุ่มเพื่อดูรายละเอียด (persist)", options["group_key"].tolist(), index=0)
        row = options.loc[options["group_key"]==chosen].iloc[0]
        mask = (
            (df["age_band"]==row["age_band"]) &
            (df["region"]==row["region"]) &
            (df["urban"]==row["urban"]) &
            (df["ses_quintile"]==row["ses_quintile"]) &
            (df["vulnerable_flag"]==row["vulnerable_flag"])
        )
        show_cols = ["student_id","academic_year","grade_code","enrolled","dropout","promoted","attendance_rate","device_access","internet_access","online_participation_rate"] + primary_scores
        st.dataframe(df.loc[mask, show_cols].sample(min(60, int(mask.sum())), random_state=7), use_container_width=True)

with tab4:
    st.subheader("EquityLens Lab — ใครเสียมากที่สุด + Policy what-if + Drill-down")
    group_cols = ["age_band","region","urban","ses_quintile","vulnerable_flag"]
    gtab = build_group_table(df, group_cols)
    gtab["risk_index"] = gtab.apply(equity_risk_index, axis=1)
    gtab = gtab.sort_values(["risk_index","n"], ascending=[False,False]).reset_index(drop=True)
    gtab["group_key"] = (
        gtab["age_band"].astype(str)+" | "+gtab["region"].astype(str)+" | urban="+gtab["urban"].astype(str)+
        " | SES="+gtab["ses_quintile"].astype(str)+" | vuln="+gtab["vulnerable_flag"].astype(str)
    )

    st.markdown("### 🧭 ตารางสรุป “ใครเสียมากที่สุด” (Top 15)")
    st.dataframe(gtab.head(15)[group_cols + ["n","enroll_rate","dropout_rate",f"{primary_scores[0]}_mean","attendance_mean","device_mean","risk_index"]], use_container_width=True)

    st.markdown("### 🧪 Policy Simulator (What-if)")
    chosen = st.selectbox("เลือกกลุ่มเพื่อจำลองนโยบาย", gtab.head(120)["group_key"].tolist(), index=0)
    row = gtab.loc[gtab["group_key"]==chosen].iloc[0]

    mask = (
        (df["age_band"]==row["age_band"]) &
        (df["region"]==row["region"]) &
        (df["urban"]==row["urban"]) &
        (df["ses_quintile"]==row["ses_quintile"]) &
        (df["vulnerable_flag"]==row["vulnerable_flag"])
    )
    grp = df.loc[mask].copy()

    k1, k2, k3 = st.columns(3)
    with k1:
        delta_device = st.slider("เพิ่ม device_access +%", 0, 30, 10, 5)
    with k2:
        delta_internet = st.slider("เพิ่ม internet_access +%", 0, 30, 10, 5)
    with k3:
        delta_att = st.slider("เพิ่ม attendance_rate + จุด", 0, 10, 2, 1)

    feat_cols = ["ses_quintile","vulnerable_flag","urban","device_access","internet_access","attendance_rate","online_participation_rate"]

    base_train = student[student["academic_year"].isin(sel_years)].copy()
    for c in feat_cols + ["dropout"] + primary_scores:
        if c in base_train.columns:
            base_train[c] = pd.to_numeric(base_train[c], errors="coerce")

    grp_s = grp.copy()
    grp_s["device_access"] = np.clip(pd.to_numeric(grp_s["device_access"], errors="coerce") + delta_device/100.0, 0, 1)
    grp_s["internet_access"] = np.clip(pd.to_numeric(grp_s["internet_access"], errors="coerce") + delta_internet/100.0, 0, 1)
    grp_s["attendance_rate"] = np.clip(pd.to_numeric(grp_s["attendance_rate"], errors="coerce") + delta_att/100.0, 0, 1)
    grp_s["online_participation_rate"] = np.clip(
        pd.to_numeric(grp_s["online_participation_rate"], errors="coerce") + 0.25*(delta_device/100.0) + 0.25*(delta_internet/100.0) + 0.15*(delta_att/100.0),
        0, 1
    )

    def predict_dropout(train_df, target_df):
        if not SKLEARN_OK:
            base = float(pd.to_numeric(target_df["dropout"], errors="coerce").mean())
            adj = (
                -0.08*(float(target_df["attendance_rate"].mean()) - float(grp["attendance_rate"].mean())) -
                -0.05*(float(target_df["device_access"].mean()) - float(grp["device_access"].mean()))
            )
            return float(np.clip(base + adj, 0, 1))
        d = train_df.dropna(subset=feat_cols+["dropout"]).copy()
        X = d[feat_cols].values
        y = d["dropout"].astype(int).values
        clf = LogisticRegression(max_iter=400, C=0.8, solver="liblinear")
        clf.fit(X, y)
        p = clf.predict_proba(target_df[feat_cols].fillna(0).values)[:,1]
        return float(np.mean(p))

    def predict_score(train_df, target_df, score_col):
        if score_col not in train_df.columns:
            return np.nan
        d = train_df[train_df["enrolled"]==1].dropna(subset=feat_cols+[score_col]).copy()
        if len(d) < 500:
            return float(pd.to_numeric(target_df[score_col], errors="coerce").mean())
        if not SKLEARN_OK:
            return float(pd.to_numeric(target_df[score_col], errors="coerce").mean())
        X = d[feat_cols].values
        y = d[score_col].values
        reg = Ridge(alpha=1.0)
        reg.fit(X, y)
        pred = reg.predict(target_df[feat_cols].fillna(0).values)
        return float(np.mean(np.clip(pred, 0, 100)))

    baseline_dropout = float(pd.to_numeric(grp["dropout"], errors="coerce").mean())
    baseline_enroll = float(pd.to_numeric(grp["enrolled"], errors="coerce").mean())
    baseline_score1 = float(pd.to_numeric(grp[primary_scores[0]], errors="coerce").mean()) if primary_scores else np.nan

    pred_dropout = predict_dropout(base_train, grp_s)
    pred_score1 = predict_score(base_train, grp_s, primary_scores[0]) if primary_scores else np.nan

    a, b, c = st.columns(3)
    with a:
        st.metric("Dropout (ก่อน)", f"{baseline_dropout*100:.2f}%")
        st.metric("Dropout (หลังจำลอง)", f"{pred_dropout*100:.2f}%", delta=f"{(pred_dropout-baseline_dropout)*100:+.2f}%")
    with b:
        st.metric("คะแนนเฉลี่ย (ก่อน)", f"{baseline_score1:.1f}")
        if not np.isnan(pred_score1):
            st.metric("คะแนนเฉลี่ย (หลังจำลอง)", f"{pred_score1:.1f}", delta=f"{(pred_score1-baseline_score1):+.1f}")
    with c:
        st.metric("Enrollment rate (ก่อน)", f"{baseline_enroll*100:.1f}%")
        st.metric("Access changes", f"device +{delta_device}%, internet +{delta_internet}%, attendance +{delta_att} จุด")

    st.markdown("### 📄 รายการข้อมูลใน Dashboard (Drill-down)")
    drill_cols = ["student_id","academic_year","grade_code","region","urban","ses_quintile","vulnerable_flag",
                  "enrolled","dropout","promoted","attendance_rate","device_access","internet_access","online_participation_rate"] + primary_scores
    st.dataframe(grp[drill_cols].sample(min(200, len(grp)), random_state=7), use_container_width=True)

with st.expander("📚 หมวดข้อมูล (6 หมวด) + 4 มิติหลัก (ดูตัวแปรที่ใช้)"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**6 หมวดข้อมูลที่ควรศึกษา**")
        for k, v in CATEGORY_MAP.items():
            st.write(f"- {k}: {', '.join(v) if v else '— (ยังไม่มีใน Dataset A)'}")
    with col2:
        st.markdown("**4 มิติหลักสำหรับลดความเหลื่อมล้ำ**")
        for k, v in DIM4_MAP.items():
            st.write(f"- {k}: {', '.join(v) if v else '—'}")
