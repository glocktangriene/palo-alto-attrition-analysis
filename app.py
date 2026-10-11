"""
Workforce Attrition Patterns & Risk Hotspot Analysis - Palo Alto Networks
Run:  streamlit run app.py
Data: Palo_Alto_Networks.csv (same folder, or upload from the sidebar)
"""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Attrition Hotspots | Palo Alto Networks",
                   page_icon="📉", layout="wide")

# ------------------------------------------------------------------ data
TENURE_BINS = [-1, 1, 3, 5, 10, 100]
TENURE_LABELS = ["0-1 yrs", "2-3 yrs", "4-5 yrs", "6-10 yrs", "11+ yrs"]
AGE_BINS = [17, 24, 34, 44, 54, 100]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]
EDU_MAP = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
MIN_GROUP = 30          # minimum headcount before a segment is flagged as a hotspot
HOTSPOT_MULT = 1.25     # flagged if rate >= 1.25x organisation baseline


@st.cache_data
def load_data(src) -> pd.DataFrame:
    df = pd.read_csv(src)

    # --- validation & cleaning -------------------------------------------
    df["Attrition"] = (df["Attrition"].astype(str).str.strip().str.lower()
                       .map({"1": 1, "0": 0, "yes": 1, "no": 0, "true": 1, "false": 0}))
    df = df.dropna(subset=["Attrition", "Department", "JobRole"]).copy()
    df["Attrition"] = df["Attrition"].astype(int)
    for c in ["Department", "JobRole", "BusinessTravel", "EducationField",
              "Gender", "MaritalStatus", "OverTime"]:
        df[c] = df[c].astype(str).str.strip().str.replace("_", " ")
    df["BusinessTravel"] = df["BusinessTravel"].str.replace("-", " ")   # "Non-Travel" -> "Non Travel"

    # --- derived fields ----------------------------------------------------
    df["AgeGroup"] = pd.cut(df["Age"], AGE_BINS, labels=AGE_LABELS)
    df["TenureBand"] = pd.cut(df["YearsAtCompany"], TENURE_BINS, labels=TENURE_LABELS)
    df["CareerStage"] = pd.cut(df["YearsAtCompany"], [-1, 2, 9, 100],
                               labels=["Early (0-2 yrs)", "Mid (3-9 yrs)", "Senior (10+ yrs)"])
    df["PromotionGap"] = pd.cut(df["YearsSinceLastPromotion"], [-1, 1, 3, 6, 100],
                                labels=["0-1 yrs", "2-3 yrs", "4-6 yrs", "7+ yrs"])
    df["DistanceBand"] = pd.cut(df["DistanceFromHome"], [0, 5, 10, 20, 100],
                                labels=["1-5", "6-10", "11-20", "21+"])
    df["EducationLevel"] = df["Education"].map(EDU_MAP)
    df["HighWorkload"] = (df["OverTime"].eq("Yes") | df["BusinessTravel"].eq("Travel Frequently"))
    return df


def rate_table(d: pd.DataFrame, col) -> pd.DataFrame:
    g = (d.groupby(col, observed=True)["Attrition"]
         .agg(Exits="sum", Headcount="count", Rate="mean").reset_index())
    g["Rate"] = (g["Rate"] * 100).round(1)
    return g


def bar(d, col, title, order=None, horizontal=False):
    t = rate_table(d, col)
    if order is not None:
        t[col] = pd.Categorical(t[col], order, ordered=True)
        t = t.sort_values(col)
    elif horizontal:
        t = t.sort_values("Rate")
    fig = px.bar(t, x="Rate" if horizontal else col, y=col if horizontal else "Rate",
                 text="Rate", hover_data=["Exits", "Headcount"], title=title,
                 color="Rate", color_continuous_scale="Reds",
                 orientation="h" if horizontal else "v")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, yaxis_title="Attrition rate (%)" if not horizontal else None,
                      xaxis_title=None if not horizontal else "Attrition rate (%)",
                      margin=dict(t=50, b=10))
    return fig


def heat(d, row, col, title):
    p = d.pivot_table(index=row, columns=col, values="Attrition", aggfunc="mean", observed=True) * 100
    n = d.pivot_table(index=row, columns=col, values="Attrition", aggfunc="count", observed=True)
    p = p.where(n >= 10)   # hide very thin cells
    fig = px.imshow(p.round(1), text_auto=True, aspect="auto", color_continuous_scale="Reds",
                    title=title, labels=dict(color="Attrition %"))
    fig.update_layout(margin=dict(t=50, b=10))
    return fig


# ------------------------------------------------------------------ load
st.sidebar.title("Filters")
up = st.sidebar.file_uploader("Upload dataset (optional)", type="csv")
try:
    full = load_data(up if up else "Palo_Alto_Networks.csv")
except FileNotFoundError:
    st.error("Place Palo_Alto_Networks.csv next to app.py or upload it from the sidebar.")
    st.stop()

depts = st.sidebar.multiselect("Department", sorted(full["Department"].unique()),
                               default=sorted(full["Department"].unique()))
role_opts = sorted(full.loc[full["Department"].isin(depts), "JobRole"].unique())
roles = st.sidebar.multiselect("Job role", role_opts, default=role_opts)
t_min, t_max = int(full["YearsAtCompany"].min()), int(full["YearsAtCompany"].max())
tenure = st.sidebar.slider("Tenure (years at company)", t_min, t_max, (t_min, t_max))
ot = st.sidebar.multiselect("Overtime", ["Yes", "No"], default=["Yes", "No"])
travel = st.sidebar.multiselect("Business travel", ["Non Travel", "Travel Rarely", "Travel Frequently"],
                                default=["Non Travel", "Travel Rarely", "Travel Frequently"])
early_x = st.sidebar.number_input("Early-tenure cut-off (X years)", 1, 10, 2)

df = full[full["Department"].isin(depts) & full["JobRole"].isin(roles)
          & full["YearsAtCompany"].between(*tenure)
          & full["OverTime"].isin(ot) & full["BusinessTravel"].isin(travel)]

st.title("Workforce Attrition Patterns & Risk Hotspots")
st.caption("Palo Alto Networks · diagnostic view of where attrition is concentrated")
if df.empty:
    st.warning("No employees match the current filters.")
    st.stop()

base = full["Attrition"].mean() * 100   # organisation-wide baseline (unfiltered)
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Department & Role Heatmaps",
                                  "Demographic Explorer", "Tenure & Workload"])

# ------------------------------------------------------------------ overview
with tab1:
    exits, hc = int(df["Attrition"].sum()), len(df)
    early = df[df["YearsAtCompany"] <= early_x]
    hi, lo = df[df["HighWorkload"]], df[~df["HighWorkload"]]
    wai = (hi["Attrition"].mean() / lo["Attrition"].mean()) if len(hi) and len(lo) and lo["Attrition"].mean() > 0 else float("nan")

    c = st.columns(5)
    c[0].metric("Attrition rate", f"{df['Attrition'].mean()*100:.1f}%",
                f"{df['Attrition'].mean()*100 - base:+.1f} pts vs org", delta_color="inverse")
    c[1].metric("Employees", f"{hc:,}")
    c[2].metric("Exited", f"{exits:,}")
    c[3].metric(f"Early-tenure attrition (≤{early_x} yrs)",
                f"{early['Attrition'].mean()*100:.1f}%" if len(early) else "n/a")
    c[4].metric("Workload Attrition Index", f"{wai:.2f}x" if wai == wai else "n/a",
                help="Attrition rate of employees who work overtime OR travel frequently, "
                     "divided by the rate of everyone else. >1 means workload-exposed staff leave more often.")

    l, r = st.columns(2)
    split = df["Attrition"].map({0: "Retained", 1: "Exited"}).value_counts().reset_index()
    split.columns = ["Status", "Employees"]
    l.plotly_chart(px.pie(split, names="Status", values="Employees", hole=.55, title="Retained vs exited",
                          color="Status", color_discrete_map={"Retained": "#4C78A8", "Exited": "#E45756"}),
                   use_container_width=True)
    r.plotly_chart(bar(df, "Department", "Attrition rate by department"), use_container_width=True)

# ------------------------------------------------------------------ heatmaps
with tab2:
    st.plotly_chart(bar(df, "JobRole", "Attrition rate by job role", horizontal=True),
                    use_container_width=True)
    l, r = st.columns(2)
    l.plotly_chart(heat(df, "Department", "TenureBand", "Department × tenure band (%)"), use_container_width=True)
    r.plotly_chart(heat(df, "JobRole", "TenureBand", "Job role × tenure band (%)"), use_container_width=True)
    st.plotly_chart(heat(df, "JobRole", "OverTime", "Job role × overtime (%)"), use_container_width=True)

    st.subheader("High-risk segments")
    st.caption(f"Flagged when attrition ≥ {HOTSPOT_MULT}× the organisation baseline ({base:.1f}%) "
               f"and headcount ≥ {MIN_GROUP}.")
    rows = []
    for col in ["Department", "JobRole"]:
        t = rate_table(df, col).rename(columns={col: "Segment"})
        t.insert(0, "Dimension", col)
        rows.append(t)
    seg = pd.concat(rows)
    seg = seg[(seg["Headcount"] >= MIN_GROUP) & (seg["Rate"] >= base * HOTSPOT_MULT)].sort_values("Rate", ascending=False)
    st.dataframe(seg, hide_index=True, use_container_width=True) if len(seg) else st.info("No segments flagged under current filters.")

# ------------------------------------------------------------------ demographics
with tab3:
    st.caption("Use the sidebar filters, plus the extra cuts below.")
    f1, f2, f3 = st.columns(3)
    g_sel = f1.multiselect("Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))
    a_sel = f2.multiselect("Age group", AGE_LABELS, default=AGE_LABELS)
    e_sel = f3.multiselect("Education level", list(EDU_MAP.values()), default=list(EDU_MAP.values()))
    dd = df[df["Gender"].isin(g_sel) & df["AgeGroup"].isin(a_sel) & df["EducationLevel"].isin(e_sel)]
    if dd.empty:
        st.warning("No employees match these demographic filters.")
    else:
        l, r = st.columns(2)
        l.plotly_chart(bar(dd, "AgeGroup", "By age group", order=AGE_LABELS), use_container_width=True)
        r.plotly_chart(bar(dd, "Gender", "By gender"), use_container_width=True)
        l, r = st.columns(2)
        l.plotly_chart(bar(dd, "MaritalStatus", "By marital status"), use_container_width=True)
        r.plotly_chart(bar(dd, "EducationLevel", "By education level", order=list(EDU_MAP.values())),
                       use_container_width=True)
        st.plotly_chart(bar(dd, "EducationField", "By education field", horizontal=True), use_container_width=True)

# ------------------------------------------------------------------ tenure & workload
with tab4:
    l, r = st.columns(2)
    l.plotly_chart(bar(df, "TenureBand", "By years at company", order=TENURE_LABELS), use_container_width=True)
    r.plotly_chart(bar(df, "CareerStage", "Early vs mid vs senior career",
                       order=["Early (0-2 yrs)", "Mid (3-9 yrs)", "Senior (10+ yrs)"]), use_container_width=True)
    l, r = st.columns(2)
    l.plotly_chart(bar(df, "PromotionGap", "By years since last promotion",
                       order=["0-1 yrs", "2-3 yrs", "4-6 yrs", "7+ yrs"]), use_container_width=True)
    r.plotly_chart(bar(df, "DistanceBand", "By distance from home (km/miles as recorded)",
                       order=["1-5", "6-10", "11-20", "21+"]), use_container_width=True)
    l, r = st.columns(2)
    l.plotly_chart(bar(df, "OverTime", "Overtime vs no overtime"), use_container_width=True)
    r.plotly_chart(bar(df, "BusinessTravel", "By business travel frequency",
                       order=["Non Travel", "Travel Rarely", "Travel Frequently"]), use_container_width=True)
    st.plotly_chart(heat(df, "BusinessTravel", "OverTime", "Travel × overtime (%) - compounding workload"),
                    use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Diagnostic analysis only: rates show association, not causation.")
