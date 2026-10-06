
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
from PIL import Image

st.set_page_config(
    page_title="RailTrack AI",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- PREMIUM UI ----------
st.markdown("""
<style>
/* FORCE LIGHT THEME */
.stApp {
    background:#f7f9fc !important;
    color:#10233e !important;
}

.main {
    background:#f7f9fc !important;
}

.block-container {
    background:#f7f9fc !important;
}

[data-testid="stAppViewContainer"] {
    background:#f7f9fc !important;
}

[data-testid="stHeader"] {
    background:#f7f9fc !important;
}
#MainMenu, footer {visibility:hidden;}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#07111f 0%,#0d1d32 100%);
    border-right: 1px solid #203650;
}
[data-testid="stSidebar"] * {color:#eaf3ff !important;}
.block-container {padding: 1.2rem 2.5rem 3rem 2.5rem; max-width:1450px;}
.hero {
    padding: 28px 32px; border-radius:22px;
    background: linear-gradient(135deg,#0a1830,#123b68 55%,#0b7890);
    color:white; margin-bottom:24px;
    box-shadow:0 12px 35px rgba(7,24,48,.18);
}
.hero h1 {font-size:38px;margin:0 0 8px;font-weight:800;}
.hero p {font-size:16px;margin:0;color:#d8eaff;}
.badge {
    display:inline-block;padding:6px 12px;border-radius:30px;
    background:rgba(255,255,255,.14);font-size:12px;
    margin-bottom:12px;border:1px solid rgba(255,255,255,.18);
}
.card {
    background:white !important;
    color:#10233e !important;
    border:1px solid #e6edf5;
    border-radius:18px;
    padding:20px 22px;
    box-shadow:0 7px 25px rgba(18,44,72,.07);
}

.card * {
    color:#10233e !important;
}
.metric-title {color:#64748b;font-size:13px;font-weight:600;}
.metric-value {font-size:27px;font-weight:800;color:#10233e;margin-top:5px;}
.section-title {font-size:23px;font-weight:800;color:#10233e;margin:18px 0 10px;}
.info-box {
    background:#f1f7ff;border-left:4px solid #1687a7;
    padding:15px 18px;border-radius:10px;margin:12px 0;
}
[data-testid="stFileUploader"] {
    border:1px dashed #7aa9c8;border-radius:14px;padding:8px;background:#f8fbff;
}
.stButton>button {
    border-radius:10px;border:0;padding:10px 20px;
    font-weight:700;background:#0b6285;color:white;
}
.stButton>button:hover {background:#084e6a;color:white;}
div[data-testid="stMetric"] {
    background:#fff !important;
    border:1px solid #e5edf5;
    border-radius:15px;
    padding:14px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p {
    color:#64748b !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color:#10233e !important;
}

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color:#10233e !important;
}
</style>
""", unsafe_allow_html=True)

REQUIRED = [
    "load_kn","frequency_hz","cycles","ballast_thickness_mm",
    "clay_strength_kpa","ballast_density_kgm3",
    "geogrid_present","geogrid_position","settlement_mm"
]
FEATURES = REQUIRED[:-1]
MODEL_PATH = Path("models/random_forest_settlement.pkl")

def clean(df):
    df=df.copy()
    for c in REQUIRED:
        if c != "geogrid_position":
            df[c]=pd.to_numeric(df[c],errors="coerce")
    df["geogrid_position"]=df["geogrid_position"].fillna("none").astype(str)
    return df.dropna(subset=[c for c in REQUIRED if c!="geogrid_position"])

def encode(df):
    x=df[FEATURES].copy()
    x["geogrid_position"]=x["geogrid_position"].astype(str).str.lower()
    return pd.get_dummies(x,columns=["geogrid_position"],dtype=int)

def hero(title, subtitle):
    st.markdown(f"""
    <div class="hero">
      <div class="badge">🚆 RAILWAY ENGINEERING • AI ANALYTICS</div>
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🚆 RailTrack AI")
    st.caption("Settlement Prediction System")
    st.markdown("---")
    page=st.radio("MAIN MENU",[
        "Dashboard",
        "Random Forest Model",
        "PLAXIS Results",
        "Settlement Predictor"
    ])
    st.markdown("---")
    st.caption("AI Engine")
    st.markdown("**Random Forest Regression**")
    st.caption("Engineering analytics dashboard")

if page=="Dashboard":
    hero("Railway Track AI Detector",
         "Intelligent prediction of railway track settlement and subgrade response.")

    a,b,c,d=st.columns(4)
    a.metric("AI MODEL","Random Forest")
    b.metric("OUTPUT","Settlement (mm)")
    c.metric("PLAXIS","JPEG / PNG")
    d.metric("METRICS","R² • RMSE • MAE")

    st.markdown('<div class="section-title">Project Workflow</div>',unsafe_allow_html=True)
    cols=st.columns(5)
    steps=[
        ("01","DATA","PLAXIS / Experimental"),
        ("02","PREPARE","Clean parameters"),
        ("03","TRAIN","Random Forest"),
        ("04","EVALUATE","R² • RMSE • MAE"),
        ("05","PREDICT","Settlement output")
    ]
    for col,(num,name,desc) in zip(cols,steps):
        with col:
            st.markdown(f"""
            <div class="card">
            <div style="font-size:12px;color:#1687a7;font-weight:800">{num}</div>
            <div style="font-size:18px;font-weight:800;color:#10233e">{name}</div>
            <div style="font-size:13px;color:#64748b;margin-top:5px">{desc}</div>
            </div>
            """,unsafe_allow_html=True)

    st.markdown('<div class="section-title">System Overview</div>',unsafe_allow_html=True)
    x,y=st.columns([1.4,1])
    with x:
        st.markdown("""
        <div class="card">
        <h3>What this system does</h3>
        <p style="color:#64748b;line-height:1.7">
        The dashboard uses engineering input parameters to train a
        Random Forest regression model and estimate railway subgrade
        settlement. PLAXIS result screenshots can also be uploaded for
        visual inspection and presentation.
        </p>
        </div>
        """,unsafe_allow_html=True)
    with y:
        st.markdown("""
        <div class="card">
        <h3>Key Inputs</h3>
        <p>Load amplitude</p><p>Loading frequency</p><p>Loading cycles</p>
        <p>Ballast thickness</p><p>Clay strength</p>
        <p>Ballast density & geogrid</p>
        </div>
        """,unsafe_allow_html=True)

elif page=="Random Forest Model":
    hero("Random Forest Model","Train, evaluate and inspect the settlement prediction model.")

    uploaded=st.file_uploader("Upload numerical dataset (CSV)",type=["csv"])
    demo=st.checkbox("Use included demo dataset")
    df=pd.read_csv(uploaded) if uploaded else (
        pd.read_csv("data/demo_settlement_dataset.csv") if demo else None)

    if df is None:
        st.markdown('<div class="info-box">Upload your PLAXIS/experimental CSV dataset to begin.</div>',unsafe_allow_html=True)
        st.stop()

    missing=[c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error("Missing columns: "+", ".join(missing)); st.stop()

    df=clean(df)
    st.success(f"{len(df)} valid records ready for modelling.")
    st.dataframe(df.head(15),use_container_width=True)

    a,b=st.columns(2)
    test_size=a.slider("Test fraction",0.10,0.40,0.20,0.05)
    trees=b.slider("Random Forest trees",50,500,200,50)

    if st.button("🚀 Train Random Forest",type="primary"):
        X=encode(df); y=df["settlement_mm"].astype(float)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=42)
        model=RandomForestRegressor(n_estimators=trees,random_state=42,n_jobs=-1)
        model.fit(Xtr,ytr)
        pred=model.predict(Xte)
        r2=r2_score(yte,pred); rmse=np.sqrt(mean_squared_error(yte,pred)); mae=mean_absolute_error(yte,pred)

        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH,"wb") as f:
            pickle.dump({"model":model,"columns":list(X.columns)},f)

        res=pd.DataFrame({"Actual Settlement (mm)":yte.values,"Predicted Settlement (mm)":pred})
        res.to_csv("results/test_predictions.csv",index=False)

        a,b,c=st.columns(3)
        a.metric("R²",f"{r2:.4f}")
        b.metric("RMSE",f"{rmse:.4f} mm")
        c.metric("MAE",f"{mae:.4f} mm")

        st.markdown('<div class="section-title">Model Performance</div>',unsafe_allow_html=True)
        st.line_chart(res)
        imp=pd.DataFrame({"Feature":X.columns,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False)
        st.markdown('<div class="section-title">Feature Importance</div>',unsafe_allow_html=True)
        st.bar_chart(imp.set_index("Feature"))
        st.download_button("⬇️ Download Predictions",res.to_csv(index=False),"test_predictions.csv","text/csv")

elif page=="PLAXIS Results":
    hero("PLAXIS Result Analysis","Upload PLAXIS 2D output screenshots and compare numerical settlement with experimental or demo data.")

    st.markdown("""
    <div class="info-box">
    <b>PLAXIS + Experimental Comparison</b><br>
    Upload your PLAXIS screenshots, enter the numerical settlement value shown by PLAXIS,
    and compare it phase-by-phase with experimental data or the included demo reference.
    </div>
    """,unsafe_allow_html=True)

    files=st.file_uploader(
        "📤 Upload PLAXIS result images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        help="Upload PLAXIS 2D Output screenshots."
    )

    phase_options=["30 Days","60 Days","90 Days","Other"]
    result_options=["Total displacement uy","Vertical displacement uy","Settlement","Deformation","Other"]

    # Comparison data source is intentionally kept inside the PLAXIS page only.
    st.markdown('<div class="section-title">Comparison Data</div>',unsafe_allow_html=True)
    source=st.radio(
        "Reference data source",
        ["Experimental CSV","Demo comparison data"],
        horizontal=True,
        key="plx_compare_source"
    )

    reference_df=None
    if source=="Experimental CSV":
        exp_file=st.file_uploader(
            "Upload experimental settlement data (CSV)",
            type=["csv"],
            key="experimental_csv",
            help="CSV should contain a phase column and a settlement_mm column. Example: phase,settlement_mm"
        )
        if exp_file:
            try:
                raw_exp=pd.read_csv(exp_file)
                # Accept a few common names without changing the main model dataset.
                rename_map={}
                for col in raw_exp.columns:
                    low=str(col).strip().lower().replace(" ","_")
                    if low in ["phase","stage","time","days"]:
                        rename_map[col]="phase"
                    elif low in ["settlement_mm","settlement","settlement_(mm)","settlement_mm_"]:
                        rename_map[col]="settlement_mm"
                exp=raw_exp.rename(columns=rename_map)
                if "phase" not in exp.columns or "settlement_mm" not in exp.columns:
                    st.error("Experimental CSV needs columns like: phase, settlement_mm")
                else:
                    exp["settlement_mm"]=pd.to_numeric(exp["settlement_mm"],errors="coerce")
                    exp=exp.dropna(subset=["settlement_mm"]).copy()
                    exp["phase"]=exp["phase"].astype(str)
                    reference_df=exp[["phase","settlement_mm"]].rename(columns={"settlement_mm":"Experimental (mm)"})
                    st.success(f"{len(reference_df)} experimental records loaded.")
                    st.dataframe(reference_df,use_container_width=True,hide_index=True)
            except Exception as e:
                st.error(f"Could not read experimental CSV: {e}")
    else:
        # Clearly labelled illustrative data for testing the comparison UI.
        reference_df=pd.DataFrame({
            "phase":["30 Days","60 Days","90 Days"],
            "Experimental (mm)":[7.80,9.70,11.60]
        })
        st.info("Demo comparison data is illustrative only. Replace it with your experimental measurements for project results.")
        st.dataframe(reference_df,use_container_width=True,hide_index=True)

    if files:
        st.markdown('<div class="section-title">PLAXIS Phase & Numerical Result</div>',unsafe_allow_html=True)
        records=[]
        for idx,file in enumerate(files):
            c1,c2,c3,c4=st.columns([1.0,1.35,1.1,2.0])
            with c1:
                phase=st.selectbox("Phase",phase_options,key=f"plx_phase_{idx}")
            with c2:
                result_type=st.selectbox("Result",result_options,key=f"plx_result_{idx}")
            with c3:
                plx_value=st.number_input(
                    "PLAXIS value (mm)",
                    min_value=0.0,max_value=100000.0,value=0.0,step=0.01,
                    key=f"plx_value_{idx}",
                    help="Enter the numerical displacement/settlement read from PLAXIS Output. Leave 0 if not available."
                )
            with c4:
                custom=st.text_input(
                    "Description",key=f"plx_desc_{idx}",
                    placeholder="e.g. reinforced track / clay subgrade"
                )
            records.append({
                "phase":phase,
                "result_type":result_type,
                "plaxis_mm":float(plx_value),
                "description":custom.strip() if custom.strip() else "—",
                "file":file.name
            })

        st.markdown('<div class="section-title">PLAXIS Results Gallery</div>',unsafe_allow_html=True)
        for start_idx in range(0,len(files),2):
            cols=st.columns(2)
            for j,file in enumerate(files[start_idx:start_idx+2]):
                idx=start_idx+j
                with cols[j]:
                    r=records[idx]
                    title=f"{r['phase']} • {r['result_type']}"
                    if r["description"]!="—":
                        title += f" • {r['description']}"
                    st.markdown(f"""
                    <div class="card">
                        <div style="font-size:12px;color:#1687a7;font-weight:800">PLAXIS 2D OUTPUT</div>
                        <div style="font-size:20px;font-weight:800;color:#10233e;margin:5px 0 12px">📌 {title}</div>
                        <div style="font-size:12px;color:#64748b;margin-bottom:10px">File: {file.name}</div>
                    """,unsafe_allow_html=True)
                    st.image(Image.open(file),use_container_width=True)
                    if r["plaxis_mm"]>0:
                        st.metric("PLAXIS numerical result",f"{r['plaxis_mm']:.2f} mm")
                    else:
                        st.caption("No numerical PLAXIS value entered for this image.")
                    st.markdown("</div>",unsafe_allow_html=True)

        plx_df=pd.DataFrame(records)
        st.markdown('<div class="section-title">Phase-wise Comparison</div>',unsafe_allow_html=True)

        if reference_df is not None:
            # Match by phase. For "Other", the user can still see the PLAXIS record in the table.
            ref=reference_df.copy()
            ref["phase"]=ref["phase"].astype(str)
            comp=plx_df[["phase","result_type","plaxis_mm","description","file"]].copy()
            comp=comp.merge(ref,on="phase",how="left")
            comp=comp.rename(columns={"plaxis_mm":"PLAXIS (mm)"})
            comp["Difference (mm)"]=np.where(
                comp["Experimental (mm)"].notna() & (comp["PLAXIS (mm)"]>0),
                comp["PLAXIS (mm)"]-comp["Experimental (mm)"],
                np.nan
            )
            comp["Absolute Error (mm)"]=comp["Difference (mm)"].abs()
            comp["Error (%)"]=np.where(
                comp["Experimental (mm)"].notna() & (comp["Experimental (mm)"]!=0) & (comp["PLAXIS (mm)"]>0),
                comp["Absolute Error (mm)"]/comp["Experimental (mm)"]*100,
                np.nan
            )
            st.dataframe(comp,use_container_width=True,hide_index=True)

            valid=comp.dropna(subset=["Experimental (mm)"]).copy()
            valid=valid[valid["PLAXIS (mm)"]>0]
            if not valid.empty:
                a,b,c=st.columns(3)
                a.metric("Compared phases",str(len(valid)))
                b.metric("Mean absolute error",f"{valid['Absolute Error (mm)'].mean():.2f} mm")
                b2=valid["Experimental (mm)"].replace(0,np.nan)
                c.metric("Mean % error",f"{(valid['Absolute Error (mm)']/b2*100).mean():.2f}%")

                chart_df=valid[["phase","Experimental (mm)","PLAXIS (mm)"]].set_index("phase")
                st.line_chart(chart_df)
                st.caption("Comparison uses the numerical PLAXIS values entered above; the screenshot itself is not numerically interpreted.")
        else:
            st.info("Upload experimental CSV or choose Demo comparison data to generate the comparison table.")

        st.markdown('<div class="section-title">PLAXIS File Summary</div>',unsafe_allow_html=True)
        st.dataframe(plx_df,use_container_width=True,hide_index=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:55px">
            <div style="font-size:55px">🖼️</div>
            <h3>PLAXIS Output Gallery</h3>
            <p style="color:#64748b">Upload JPG, JPEG or PNG screenshots, then enter the numerical PLAXIS settlement/displacement value to compare it with experimental or demo data.</p>
        </div>
        """,unsafe_allow_html=True)

elif page=="Settlement Predictor":
    hero("Settlement Predictor","Estimate subgrade settlement using the trained Random Forest model.")

    if not MODEL_PATH.exists():
        st.markdown('<div class="info-box">Train the Random Forest model first.</div>',unsafe_allow_html=True)
        st.stop()

    with open(MODEL_PATH,"rb") as f:
        bundle=pickle.load(f)
    model,columns=bundle["model"],bundle["columns"]

    st.markdown('<div class="section-title">Engineering Parameters</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    load=a.number_input("Load amplitude (kN)",0.0,10000.0,100.0)
    freq=b.number_input("Loading frequency (Hz)",0.0,100.0,2.0)
    cycles=c.number_input("Loading cycles",1.0,1000000.0,1000.0)
    a,b,c=st.columns(3)
    ballast=a.number_input("Ballast thickness (mm)",0.0,2000.0,300.0)
    clay=b.number_input("Clay strength (kPa)",0.0,500.0,25.0)
    density=c.number_input("Ballast density (kg/m³)",0.0,3000.0,1800.0)
    a,b=st.columns(2)
    geo=a.selectbox("Geogrid present",[0,1],format_func=lambda x:"Yes" if x else "No")
    pos=b.selectbox("Geogrid position",["none","base","middle","top"])

    if st.button("🔮 Predict Settlement",type="primary"):
        inp=pd.DataFrame([{
            "load_kn":load,"frequency_hz":freq,"cycles":cycles,
            "ballast_thickness_mm":ballast,"clay_strength_kpa":clay,
            "ballast_density_kgm3":density,"geogrid_present":geo,
            "geogrid_position":pos
        }])
        Xnew=encode(inp).reindex(columns=columns,fill_value=0)
        val=float(model.predict(Xnew)[0])
        level="LOW" if val<5 else ("MODERATE" if val<10 else "HIGH")
        st.markdown(f"""
        <div class="card" style="text-align:center;margin-top:25px">
        <div class="metric-title">PREDICTED SETTLEMENT</div>
        <div style="font-size:48px;font-weight:900;color:#0b6285">{val:.3f} mm</div>
        <div style="font-size:18px;font-weight:700">Settlement Indicator: {level}</div>
        </div>
        """,unsafe_allow_html=True)
