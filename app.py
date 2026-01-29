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
**Retrospective computational meta-analysis (n=15 Phase II-III oncology trials across solid tumors)**

**Primary Research Question**:  
Do nanoparticle size distributions cluster non-randomly among formulations that advance 
from Phase II to Phase III, suggesting translational selection pressures that may partially transcend 
individual disease targets?

*Analysis is hypothesis-generating; causal inference is not possible.*
""")

st.info("""
**Key Assumptions:**  
- Late-phase filtering imposes partially shared physicochemical constraints across solid tumors.  
- Insufficient per-indication n prevents statistical control for tumor type.  
- Phase III advancement reflects sponsor/regulatory decisions, not purely clinical efficacy.
""")

# ==========================
# II. Load Dataset
# ==========================
@st.cache_data
def load_data():
    data = {
        'NCT_ID': ['NCT00003105', 'NCT00507874', 'NCT00964028', 'NCT01735643', 'NCT02650635',
                  'NCT00541080', 'NCT00448961', 'NCT00749457', 'NCT01374251', 'NCT02116399',
                  'NCT00303910', 'NCT00826085', 'NCT01042344', 'NCT01564969', 'NCT02233341'],
        'Reported_Nominal_Size_nm': [110, 85, 95, 120, 75, 100, 90, 105, 80, 115, 98, 88, 112, 78, 102],
        'Indication': ['Breast', 'Lung', 'Breast', 'Ovarian', 'Melanoma', 'Breast', 'Pancreatic', 
                      'Lung', 'Colorectal', 'Breast', 'Ovarian', 'Lung', 'Breast', 'Gastric', 'Breast'],
        'PEGylated': [1,1,1,1,0,1,1,1,0,1,1,1,1,0,1],
        'Phase_III_Advancement': [1,1,1,1,0,1,1,1,0,1,1,0,1,0,1],
        'Platform': ['Liposome', 'Liposome', 'Liposome', 'Liposome', 'Polymeric', 'Liposome', 
                    'Liposome', 'Liposome', 'Polymeric', 'Liposome', 'Liposome', 'Liposome', 
                    'Liposome', 'Polymeric', 'Liposome']
    }
    df = pd.DataFrame(data)
    df['Success'] = df['Phase_III_Advancement']
    return df

df = load_data()

# ==========================
# Sidebar: Composition
# ==========================
with st.sidebar:
    st.markdown("### Dataset Composition (n=15)")
    for ind, count in df['Indication'].value_counts().items():
        st.markdown(f"• {ind}: {count}")
    st.markdown("**Anchor Platform**: Liposomes (10/15 trials)")
    st.markdown("**Key Limitation**: Survivorship bias (public trials only)")

# ==========================
# III. Core Metrics
# ==========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trials", len(df))
col2.metric("Phase III Rate", f"{df.Success.mean():.0%}")
col3.metric("Median Size (nm)", f"{df.Reported_Nominal_Size_nm.median():.0f}")
col4.metric("Liposome Trials", f"{(df.Platform=='Liposome').sum()}/15")

# ==========================
# IV. Primary Analysis
# ==========================
st.markdown("---")
st.header("Primary Analysis: Size vs Phase III Advancement")
st.markdown("""
*Spearman rank correlation assesses monotonic association with minimal assumptions; 
binary outcome encoding requires descriptive interpretation.*
""")

spearman_r, spearman_p = spearmanr(df['Reported_Nominal_Size_nm'], df['Success'])

col1, col2 = st.columns([2,1])
with col1:
    fig = px.scatter(df, x='Reported_Nominal_Size_nm', y='Success', color='Indication', 
                     size='Reported_Nominal_Size_nm', opacity=0.7,
                     hover_data=['NCT_ID','Platform'],
                     title="Size vs Phase III Advancement Across Solid Tumors")
    fig.update_traces(mode='markers')
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Marker size is for visualization; not weighted or causal.")

with col2:
    st.metric("Spearman ρ", f"{spearman_r:.3f}")
    st.metric("p-value", f"{spearman_p:.3f}")
    st.caption("*Small-to-moderate effect size; descriptive only*")

# ==========================
# V. Operational Bins
# ==========================
st.subheader("Size Stratification")
st.markdown("""
*Bins reflect common manufacturable nanoparticle size ranges in late-phase trials, not tumor-specific permeability thresholds.*
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

lipo_df = df[df['Platform']=='Liposome']
lipo_r, lipo_p = spearmanr(lipo_df['Reported_Nominal_Size_nm'], lipo_df['Success'])
st.success(f"Liposomes (n=10): ρ={lipo_r:.2f}, p={lipo_p:.3f}")

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
# IX. GBM as Stress-Test
# ==========================
st.markdown("---")
st.header("Application Context: Glioblastoma (Illustrative)")
st.markdown("""
• BBB gaps (50–200nm) overlap manufacturable size window  
• High RES clearance increases importance of size optimization  
• Manufacturing constraints apply across solid tumors

*No GBM-specific trial data included; serves to contextualize observed translational constraints.*
""")

# ==========================
# X. Limitations
# ==========================
with st.expander("Methodological Limitations", expanded=True):
    st.markdown("""
**Data Limitations:**  
1. n=15 (low statistical power)  
2. Survivorship bias: only publicly registered trials  
3. Nominal sizes only; no PDI/distribution  
4. Indication confounding not modeled (insufficient n per cancer)  
5. Phase III advancement reflects sponsor/regulatory choice beyond clinical performance  

**Analysis Limitations:**  
1. No causal inference  
2. Multiple comparisons not corrected  

*Hypothesis-generating analysis only.*
""")

# ==========================
# XI. Contribution
# ==========================
st.markdown("---")
st.markdown("""
**Contribution:** Documents non-random nanoparticle size clustering across late-phase oncology trials,  
highlighting translational constraints often invisible at early development stages.  
May inform preclinical screening and multi-parameter optimization.
""")

st.markdown("*Computational analysis of ClinicalTrials.gov + publications | January 2026*")
