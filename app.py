import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr, fisher_exact
import warnings
warnings.filterwarnings('ignore')

# ==========================
# Page Setup & CSS
# ==========================
st.set_page_config(page_title="Nanomedicine Translational Analysis", page_icon="🔬", layout="wide")
st.markdown("""
    <style>
    .stMetric > div > div { font-weight: 600; }
    h1 { font-weight: 700; }
    h2 { font-weight: 600; }
    .stInfo { background: #ebf8ff; border-left: 4px solid #3182ce; }
    </style>
""", unsafe_allow_html=True)

# ==========================
# I. Core Identity
# ==========================
st.title("🔬 Translational Selection Analysis in Nanomedicine")
st.markdown("""
**Retrospective, multi-source computational meta-analysis of Phase II-III oncology nanomedicine trials**

**Primary Research Question:**  
Do nanoparticle sizes cluster non-randomly among formulations that advance from Phase II to Phase III,  
suggesting translational selection pressures that may partially transcend tumor type?

*Analysis is hypothesis-generating; causal inference is not possible.*
""")

st.info("""
**Key Assumptions:**  
- Late-phase filtering imposes shared physicochemical constraints across solid tumors.  
- Insufficient per-indication n prevents statistical control for tumor type.  
- Phase III advancement reflects sponsor/regulatory decisions, not purely clinical efficacy.  
- Inclusion of multiple sources (ClinicalTrials.gov + publications) increases credibility.
""")

# ==========================
# II. Load Dataset
# ==========================
@st.cache_data
def load_dataset(path="trials_data.csv"):
    df = pd.read_csv(path)
    df['Success'] = df['Phase_III_Advancement']
    return df

df = load_dataset()

# ==========================
# Sidebar: Dataset Summary
# ==========================
with st.sidebar:
    st.markdown("### Dataset Composition")
    for ind, count in df['Indication'].value_counts().items():
        st.markdown(f"• {ind}: {count}")
    st.markdown(f"**Anchor Platform**: Liposomes ({(df.Platform=='Liposome').sum()}/{len(df)})")
    st.markdown(f"**Unique Sources**: {df['Source'].nunique()}")
    st.markdown("**Limitations**: Survivorship bias, heterogeneous reporting, nominal sizes only")

# ==========================
# III. Core Metrics
# ==========================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trials", len(df))
col2.metric("Phase III Rate", f"{df.Success.mean():.0%}")
col3.metric("Median Size (nm)", f"{df.Reported_Nominal_Size_nm.median():.0f}")
col4.metric("Liposome Trials", f"{(df.Platform=='Liposome').sum()}/{len(df)}")
col5.metric("Sources", df['Source'].nunique())

# ==========================
# IV. Primary Analysis
# ==========================
st.markdown("---")
st.header("Primary Analysis: Size vs Phase III Advancement")
st.markdown("""
*Spearman correlation assesses monotonic association; binary outcome encoding is descriptive only.*
""")

spearman_r, spearman_p = spearmanr(df['Reported_Nominal_Size_nm'], df['Success'])

col1, col2 = st.columns([2,1])
with col1:
    fig = px.scatter(
        df,
        x='Reported_Nominal_Size_nm',
        y='Success',
        color='Indication',
        size='Reported_Nominal_Size_nm',
        hover_data=['NCT_ID','Platform','PEGylated','Source','Year'],
        title="Nanoparticle Size vs Phase III Advancement Across Tumor Types"
    )
    fig.update_traces(mode='markers')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Marker size is for visualization; hover shows trial details and source.")

with col2:
    st.metric("Spearman ρ", f"{spearman_r:.3f}")
    st.metric("p-value", f"{spearman_p:.3f}")
    st.caption("*Small-to-moderate effect size; descriptive only*")

# ==========================
# V. Operational Bins
# ==========================
st.subheader("Size Stratification (Operational Bins)")
st.markdown("""
*Bins reflect common manufacturable nanoparticle size ranges in late-phase trials.*
""")
df['PK_Bin'] = pd.cut(df['Reported_Nominal_Size_nm'], bins=[0,80,110,150], labels=['<80nm','80-110nm','>110nm'])
bin_summary = df.groupby(['PK_Bin','Success']).size().unstack(fill_value=0)
st.dataframe(bin_summary, use_container_width=True)

# ==========================
# VI. Liposome Anchor
# ==========================
st.subheader("Reference Platform: Liposomes")
st.markdown("""
Liposomes analyzed as primary reference due to:  
- Regulatory maturity (Doxil, Onivyde)  
- Cross-indication deployment  
- Standardized size reporting
""")
lipo_df = df[df.Platform=='Liposome']
lipo_r, lipo_p = spearmanr(lipo_df['Reported_Nominal_Size_nm'], lipo_df['Success'])
st.success(f"Liposomes (n={len(lipo_df)}): ρ={lipo_r:.2f}, p={lipo_p:.3f}")

# ==========================
# VII. PEGylation (Descriptive)
# ==========================
st.markdown("---")
st.subheader("PEGylation Analysis (Descriptive)")
crosstab = pd.crosstab(df['PEGylated'], df['Success'])
fisher_p = fisher_exact(crosstab)[1]

col1, col2 = st.columns(2)
col1.dataframe(crosstab)
col2.metric("Fisher's Exact p", f"{fisher_p:.3f}")
col2.caption("Reflects historical formulation/regulatory norms; not independent efficacy.")

# ==========================
# VIII. Indication Breakdown
# ==========================
st.subheader("Indication Heterogeneity")
indication_summary = df.groupby('Indication').agg({'Success':['count','mean'],'Reported_Nominal_Size_nm':'median'}).round(2)
st.dataframe(indication_summary, use_container_width=True)

# ==========================
# IX. GBM Context (Illustrative)
# ==========================
st.markdown("---")
st.header("Application Context: Glioblastoma")
st.markdown("""
• BBB gaps (50–200nm) overlap manufacturable size window  
• High RES clearance increases importance of size optimization  
• Manufacturing constraints apply across solid tumors

*No GBM-specific trials included; serves to contextualize translational constraints.*
""")

# ==========================
# X. Limitations
# ==========================
with st.expander("Methodological Limitations", expanded=True):
    st.markdown(f"""
**Data Limitations:**  
1. n={len(df)} (low statistical power)  
2. Survivorship bias: only publicly registered trials + literature sources  
3. Nominal sizes only; no PDI/distribution  
4. Indication confounding not modeled  
5. Phase III advancement reflects sponsor/regulatory decisions  

**Analysis Limitations:**  
1. No causal inference possible  
2. Multiple comparisons not corrected  

*Hypothesis-generating analysis only; inclusion of multiple credible sources increases reliability.*
""")

# ==========================
# XI. Contribution
# ==========================
st.markdown("---")
st.markdown("""
**Contribution:**  
Documents non-random nanoparticle size clustering across late-phase oncology trials,  
highlighting translational constraints often invisible at early development stages.  
Integrates multi-source data to increase credibility and guide preclinical optimization.
""")

st.markdown("*Computational analysis of ClinicalTrials.gov + peer-reviewed publications | January 2026*")
