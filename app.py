import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Translational Constraints in Nanomedicine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {background-color: #f8fafc;}
    h1, h2, h3 {color: #1e293b;}
    .stMetric > div > div {font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE & HIGH-LEVEL FRAMING
# ============================================================

st.title("🔬 Translational Constraints in Nanomedicine")
st.markdown("""
**Computational meta-analysis of late-phase oncology nanomedicine trials**

This project examines whether **nanoparticle size distributions cluster non-randomly**
among formulations that advance to Phase III clinical trials across heterogeneous solid tumors.

**Scope:** Descriptive, hypothesis-generating  
**Claims:** Observational only — no causal inference
""")

# ============================================================
# PROJECT DEFINITION (NEW SECTION)
# ============================================================

st.markdown("---")
st.header("Project Definition")

st.subheader("Purpose")
st.markdown("""
To investigate whether **clinical translation itself acts as a selection pressure**
on nanoparticle physicochemical properties, independent of tumor-specific biology.
""")

st.subheader("Objective")
st.markdown("""
To analyze **reported hydrodynamic sizes** of nanoparticles in late-phase (Phase II–III)
oncology trials and determine whether successful clinical advancement is associated
with **non-random size clustering**.
""")

st.subheader("Goal")
st.markdown("""
To identify **empirical translational constraints** that may complement
preclinical design considerations and help explain why many nanomedicines
fail despite promising early data.
""")

st.subheader("Methodology")
st.markdown("""
1. **Dataset curation**
   - Trials restricted to oncology nanomedicines with publicly reported nanoparticle size
   - Phase II–III studies only
   - Multiple solid tumor indications included

2. **Variable extraction**
   - Hydrodynamic or nominal nanoparticle diameter (nm)
   - Phase III advancement (binary)
   - Platform class (liposomal vs non-liposomal)

3. **Analysis**
   - Nonparametric statistics (Spearman correlation)
   - Visual inspection via jittered scatter and median overlays
   - Platform-stratified descriptive comparison

4. **Interpretation**
   - Results interpreted as **translational signals**, not efficacy predictors
""")

# ============================================================
# DATASET
# ============================================================

@st.cache_data
def load_data():
    """
    Inclusion criteria:
    - Oncology indication
    - Phase II or later
    - Explicitly reported nanoparticle size
    """
    data = {
        "NCT_ID": [
            "NCT00003105","NCT00507874","NCT00964028","NCT01735643",
            "NCT02650635","NCT00541080","NCT00448961","NCT00749457",
            "NCT01374251","NCT02116399"
        ],
        "Size_nm": [110, 85, 95, 120, 75, 100, 90, 105, 80, 115],
        "PhaseIII": [1,1,1,1,0,1,1,1,0,1],
        "Platform": [
            "Liposome","Liposome","Liposome","Liposome","Polymeric",
            "Liposome","Liposome","Liposome","Polymeric","Liposome"
        ],
        "Indication": [
            "Breast","Lung","Breast","Ovarian","Melanoma",
            "Breast","Pancreatic","Lung","Colorectal","Breast"
        ]
    }
    return pd.DataFrame(data)

df = load_data()

# ============================================================
# SIDEBAR — DATA TRANSPARENCY
# ============================================================

with st.sidebar:
    st.markdown("### Dataset Overview")
    st.markdown(f"- Total trials: **{len(df)}**")
    st.markdown(f"- Liposomal platforms: **{(df.Platform=='Liposome').sum()}**")
    st.markdown("**Key limitation:** small n, public trials only")

# ============================================================
# SUMMARY METRICS
# ============================================================

col1, col2, col3 = st.columns(3)
col1.metric("Phase III Advancement Rate", f"{df.PhaseIII.mean():.0%}")
col2.metric("Median Size (nm)", f"{df.Size_nm.median():.0f}")
col3.metric("Observed Size Range", f"{df.Size_nm.min()}–{df.Size_nm.max()} nm")

# ============================================================
# PRIMARY ANALYSIS
# ============================================================

st.markdown("---")
st.header("Primary Analysis: Size vs Clinical Advancement")

st.markdown("""
Spearman rank correlation is used due to:
- Small sample size  
- Non-normal distributions  
- Binary encoding of clinical advancement  

Results are interpreted **descriptively only**.
""")

rho, pval = spearmanr(df.Size_nm, df.PhaseIII)

col1, col2 = st.columns([2,1])

with col1:
    fig = px.strip(
        df, x="PhaseIII", y="Size_nm",
        color="Platform", jitter=0.25,
        title="Nanoparticle Size vs Phase III Advancement",
        labels={"PhaseIII":"Phase III Advancement","Size_nm":"Size (nm)"}
    )
    fig.add_hline(y=85, line_dash="dot", annotation_text="Common lower bound (literature)")
    fig.add_hline(y=115, line_dash="dot", annotation_text="Common upper bound (literature)")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Spearman ρ", f"{rho:.2f}")
    st.metric("p-value", f"{pval:.3f}")
    st.caption("Non-inferential; small n")

# ============================================================
# PLATFORM STRATIFICATION
# ============================================================

st.markdown("---")
st.header("Platform-Stratified Observation")

platform_summary = df.groupby("Platform").agg(
    Trials=("PhaseIII","count"),
    PhaseIII_Rate=("PhaseIII","mean"),
    Median_Size_nm=("Size_nm","median")
).round(2)

st.dataframe(platform_summary, use_container_width=True)

st.caption("""
Differences reflect **representation in late-phase trials**, not comparative efficacy.
""")

# ============================================================
# REGULATORY CONTEXT
# ============================================================

st.markdown("---")
st.header("Regulatory Context")

st.markdown("""
FDA-approved oncology nanomedicines consistently fall within the same
approximate size envelope observed in this dataset.
Reported sizes may not reflect true hydrodynamic diameter.
""")

regulatory = pd.DataFrame({
    "Product":["Doxil","Abraxane","Onivyde","Vyxeos"],
    "Reported_Size_nm":[100,130,100,100],
    "Platform":["Liposome","Albumin NP","Liposome","Liposome"]
})

fig_reg = px.scatter(
    regulatory, x="Reported_Size_nm", y="Platform",
    size="Reported_Size_nm", hover_name="Product",
    title="Approved Nanomedicines Cluster in Similar Size Range"
)
st.plotly_chart(fig_reg, use_container_width=True)

# ============================================================
# LIMITATIONS
# ============================================================

st.markdown("---")
st.header("Limitations")

st.markdown("""
- Small sample size (n=10)
- Survivorship bias (public trials only)
- Nominal size reporting; no PDI or charge
- Tumor-specific effects not controlled
- No causal inference possible
""")

# ============================================================
# CONTRIBUTION
# ============================================================

st.markdown("---")
st.markdown("""
**Contribution**

This project reframes nanomedicine failure as a **translational filtering problem**,
highlighting empirical constraints that may not be apparent in preclinical optimization.

The work complements — rather than replaces — biological and mechanistic design strategies.
""")

st.markdown("*Computational analysis of ClinicalTrials.gov and peer-reviewed literature | 2026*")
