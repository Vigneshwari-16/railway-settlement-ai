
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
.stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:#f7f9fc !important; color:#10233e !important;}
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
    background:white;border:1px solid #e6edf5;border-radius:18px;
    padding:20px 22px;box-shadow:0 7px 25px rgba(18,44,72,.07);
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
    background:#fff !important;border:1px solid #e5edf5;border-radius:15px;padding:14px;
}
div[data-testid="stMetric"] label, div[data-testid="stMetric"] label p {color:#64748b !important;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#10233e !important;}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {color:#10233e !important;}
</style>
""", unsafe_allow_html=True)

REQUIRED = [
    "load_kn","frequency_hz","cycles","ballast_thickness_mm",
    "clay_strength_kpa","ballast_density_kgm3",
    "geogrid_present","geogrid_position","settlement_mm"
]
FEATURES = REQUIRED[:-1]
MODEL_PATH = Path("models/random_forest_settlement.pkl")
DEMO_CSV = """load_kn,frequency_hz,cycles,ballast_thickness_mm,clay_strength_kpa,ballast_density_kgm3,geogrid_present,geogrid_position,settlement_mm
50,2,1000,300,25,1700,0,none,8.42
60,2,1500,300,25,1700,0,none,9.85
70,2,2000,300,25,1700,0,none,11.26
80,3,2500,300,25,1700,0,none,12.94
90,3,3000,300,25,1700,0,none,14.51
100,3,3500,300,25,1700,0,none,16.18
50,2,1000,350,30,1750,1,base,6.72
60,2,1500,350,30,1750,1,base,7.54
70,2,2000,350,30,1750,1,base,8.63
80,3,2500,350,30,1750,1,base,9.72
90,3,3000,350,30,1750,1,base,10.84
100,3,3500,350,30,1750,1,base,12.05
50,2,1000,400,35,1800,1,middle,5.31
60,2,1500,400,35,1800,1,middle,6.08
70,2,2000,400,35,1800,1,middle,6.91
80,3,2500,400,35,1800,1,middle,7.83
90,3,3000,400,35,1800,1,middle,8.76
100,3,3500,400,35,1800,1,middle,9.64
60,4,2000,300,20,1650,0,none,12.38
80,4,3000,300,20,1650,0,none,15.87
100,4,4000,300,20,1650,0,none,19.42
60,4,2000,350,30,1750,1,top,7.18
80,4,3000,350,30,1750,1,top,9.36
100,4,4000,350,30,1750,1,top,11.52
70,5,2500,400,40,1850,1,middle,6.45
90,5,3500,400,40,1850,1,middle,8.12
110,5,4500,400,40,1850,1,middle,10.07
"""

def get_demo_data():
    from io import StringIO
    return pd.read_csv(StringIO(DEMO_CSV))

def load_demo_data():
    demo_path = Path("data/demo_settlement_dataset.csv")
    try:
        if demo_path.exists() and demo_path.stat().st_size > 20:
            df = pd.read_csv(demo_path)
            if len(df.columns) > 1 and not df.empty:
                return df
    except Exception:
        pass
    return get_demo_data()

def train_full_demo_model():
    df = clean(get_demo_data())
    X = encode(df)
    y = df["settlement_mm"].astype(float)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "columns": list(X.columns)}, f)
    return model, list(X.columns)


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
        "Settlement Predictor",
        "Comparison & Validation"
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
    if uploaded:
        try:
            df=pd.read_csv(uploaded)
        except pd.errors.EmptyDataError:
            st.warning("Uploaded CSV is empty. Please upload a valid dataset.")
            st.stop()
    elif demo:
        df=load_demo_data()
        if not Path("data/demo_settlement_dataset.csv").exists() or Path("data/demo_settlement_dataset.csv").stat().st_size <= 20:
            st.info("Using the built-in illustrative demo dataset because the repository CSV is empty.")
    else:
        df=None

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
        Path("results").mkdir(parents=True, exist_ok=True)
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
    hero("PLAXIS Result Viewer","Upload and present PLAXIS Output screenshots phase-by-phase.")

    files=st.file_uploader(
        "Upload PLAXIS result images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )

    if files:
        for start in range(0,len(files),2):
            cols=st.columns(2)
            for j,file in enumerate(files[start:start+2]):
                with cols[j]:
                    st.markdown(f'<div class="card"><h4>📌 {file.name}</h4>',unsafe_allow_html=True)
                    st.image(Image.open(file),use_container_width=True)
                    st.text_input("Phase / result label",
                                  placeholder="Example: 90 Days • Total displacement uy",
                                  key=f"label_{start+j}")
                    st.markdown("</div>",unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:50px">
        <div style="font-size:55px">🖼️</div>
        <h3>PLAXIS Output Gallery</h3>
        <p style="color:#64748b">Upload JPG, JPEG or PNG result screenshots.</p>
        </div>
        """,unsafe_allow_html=True)


elif page=="Comparison & Validation":
    hero("Comparison & Validation",
         "Compare Experimental, PLAXIS and Random Forest settlement results on the same phases.")

    st.markdown("""
    <div class="info-box">
    <b>Validation basis:</b> Experimental settlement is treated as the reference dataset.
    PLAXIS and Random Forest predictions are compared against it using MAE, RMSE and R².
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">1. Select comparison data</div>', unsafe_allow_html=True)

    source = st.radio(
        "Data source",
        ["Use comparison CSV", "Use demo comparison data", "Enter values manually"],
        horizontal=True
    )

    comparison_df = None

    if source == "Use comparison CSV":
        uploaded_compare = st.file_uploader(
            "Upload comparison CSV",
            type=["csv"],
            key="comparison_csv",
            help="Required columns: phase, experimental_mm, plaxis_mm, random_forest_mm"
        )

        if uploaded_compare:
            try:
                comparison_df = pd.read_csv(uploaded_compare)
            except Exception as e:
                st.error(f"Could not read the CSV: {e}")
                st.stop()
        else:
            st.markdown("""
            <div class="card">
            <b>Required CSV format</b><br><br>
            phase, experimental_mm, plaxis_mm, random_forest_mm<br>
            30 Days, 8.50, 8.90, 8.65<br>
            60 Days, 10.20, 10.70, 10.35<br>
            90 Days, 11.70, 11.84, 11.75
            </div>
            """, unsafe_allow_html=True)

    elif source == "Use demo comparison data":
        comparison_df = pd.DataFrame({
            "phase": ["30 Days", "60 Days", "90 Days"],
            "experimental_mm": [8.50, 10.20, 11.70],
            "plaxis_mm": [8.90, 10.70, 11.84],
            "random_forest_mm": [8.65, 10.35, 11.75]
        })
        st.info("Demo comparison data is illustrative and is not experimental/project evidence.")

    else:
        st.markdown("Enter the values for the same phases/conditions used by all three methods.")
        manual_default = pd.DataFrame({
            "phase": ["30 Days", "60 Days", "90 Days"],
            "experimental_mm": [0.0, 0.0, 0.0],
            "plaxis_mm": [0.0, 0.0, 0.0],
            "random_forest_mm": [0.0, 0.0, 0.0]
        })
        edited = st.data_editor(
            manual_default,
            num_rows="dynamic",
            use_container_width=True,
            key="manual_comparison_editor"
        )
        comparison_df = edited

    if comparison_df is not None:
        required_compare = ["phase", "experimental_mm", "plaxis_mm", "random_forest_mm"]
        missing_compare = [c for c in required_compare if c not in comparison_df.columns]

        if missing_compare:
            st.error("Missing columns: " + ", ".join(missing_compare))
            st.stop()

        comparison_df = comparison_df[required_compare].copy()
        for c in ["experimental_mm", "plaxis_mm", "random_forest_mm"]:
            comparison_df[c] = pd.to_numeric(comparison_df[c], errors="coerce")

        comparison_df = comparison_df.dropna(subset=required_compare).reset_index(drop=True)

        if len(comparison_df) == 0:
            st.warning("No valid comparison rows are available.")
            st.stop()

        st.markdown('<div class="section-title">2. Comparison dataset</div>', unsafe_allow_html=True)
        st.dataframe(
            comparison_df.rename(columns={
                "phase": "Phase",
                "experimental_mm": "Experimental (mm)",
                "plaxis_mm": "PLAXIS (mm)",
                "random_forest_mm": "Random Forest (mm)"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Error calculations against experimental reference.
        exp = comparison_df["experimental_mm"].to_numpy(dtype=float)
        plx = comparison_df["plaxis_mm"].to_numpy(dtype=float)
        rf = comparison_df["random_forest_mm"].to_numpy(dtype=float)

        plx_abs = np.abs(plx - exp)
        rf_abs = np.abs(rf - exp)

        plx_mae = float(np.mean(plx_abs))
        rf_mae = float(np.mean(rf_abs))
        plx_rmse = float(np.sqrt(np.mean((plx - exp) ** 2)))
        rf_rmse = float(np.sqrt(np.mean((rf - exp) ** 2)))

        # R² requires at least two non-constant reference values.
        if len(exp) >= 2 and np.std(exp) > 0:
            plx_r2 = float(r2_score(exp, plx))
            rf_r2 = float(r2_score(exp, rf))
        else:
            plx_r2 = np.nan
            rf_r2 = np.nan

        st.markdown('<div class="section-title">3. Performance against Experimental Data</div>',
                    unsafe_allow_html=True)

        a, b, c = st.columns(3)
        a.metric("PLAXIS MAE", f"{plx_mae:.3f} mm")
        b.metric("PLAXIS RMSE", f"{plx_rmse:.3f} mm")
        c.metric("PLAXIS R²", "N/A" if np.isnan(plx_r2) else f"{plx_r2:.4f}")

        a, b, c = st.columns(3)
        a.metric("Random Forest MAE", f"{rf_mae:.3f} mm")
        b.metric("Random Forest RMSE", f"{rf_rmse:.3f} mm")
        c.metric("Random Forest R²", "N/A" if np.isnan(rf_r2) else f"{rf_r2:.4f}")

        # Phase-wise errors.
        result_df = comparison_df.copy()
        result_df["PLAXIS Error (mm)"] = plx_abs
        result_df["RF Error (mm)"] = rf_abs
        result_df["PLAXIS Error (%)"] = np.where(
            exp != 0, (plx_abs / np.abs(exp)) * 100, np.nan
        )
        result_df["RF Error (%)"] = np.where(
            exp != 0, (rf_abs / np.abs(exp)) * 100, np.nan
        )

        st.markdown('<div class="section-title">4. Phase-wise Error Analysis</div>',
                    unsafe_allow_html=True)
        st.dataframe(
            result_df.rename(columns={
                "phase": "Phase",
                "experimental_mm": "Experimental (mm)",
                "plaxis_mm": "PLAXIS (mm)",
                "random_forest_mm": "Random Forest (mm)"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Overall winner: lower RMSE first, then lower MAE.
        if rf_rmse < plx_rmse:
            winner = "Random Forest"
            reason = "lower RMSE against the Experimental reference."
        elif plx_rmse < rf_rmse:
            winner = "PLAXIS"
            reason = "lower RMSE against the Experimental reference."
        elif rf_mae < plx_mae:
            winner = "Random Forest"
            reason = "equal RMSE but lower MAE against the Experimental reference."
        elif plx_mae < rf_mae:
            winner = "PLAXIS"
            reason = "equal RMSE but lower MAE against the Experimental reference."
        else:
            winner = "Tie"
            reason = "both methods have the same calculated error."

        st.markdown('<div class="section-title">5. Overall Validation Result</div>',
                    unsafe_allow_html=True)

        if winner == "Tie":
            st.success("🏆 Overall result: TIE — both methods have the same calculated error.")
        else:
            st.success(f"🏆 Better agreement with Experimental Data: **{winner}** — {reason}")

        chart_df = comparison_df.set_index("phase")[
            ["experimental_mm", "plaxis_mm", "random_forest_mm"]
        ].rename(columns={
            "experimental_mm": "Experimental",
            "plaxis_mm": "PLAXIS",
            "random_forest_mm": "Random Forest"
        })

        st.markdown('<div class="section-title">6. Settlement Comparison</div>',
                    unsafe_allow_html=True)
        st.line_chart(chart_df, use_container_width=True)

        # Downloadable validation report.
        summary_df = pd.DataFrame({
            "Method": ["PLAXIS", "Random Forest"],
            "MAE (mm)": [plx_mae, rf_mae],
            "RMSE (mm)": [plx_rmse, rf_rmse],
            "R2": [plx_r2, rf_r2]
        })

        csv_report = result_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Comparison Results",
            csv_report,
            "experimental_plaxis_random_forest_comparison.csv",
            "text/csv"
        )

        st.download_button(
            "⬇️ Download Performance Summary",
            summary_df.to_csv(index=False),
            "model_performance_summary.csv",
            "text/csv"
        )


elif page=="Settlement Predictor":
    hero("Settlement Predictor","Estimate subgrade settlement using the Random Forest model.")

    if not MODEL_PATH.exists():
        st.info("No saved model found. Training the Random Forest automatically from the built-in illustrative demo dataset...")
        model,columns=train_full_demo_model()
        st.success("Demo Random Forest model is ready. You can now enter engineering parameters.")
    else:
        try:
            with open(MODEL_PATH,"rb") as f:
                bundle=pickle.load(f)
            model,columns=bundle["model"],bundle["columns"]
        except Exception:
            st.warning("Saved model could not be loaded. Rebuilding the demo Random Forest model...")
            model,columns=train_full_demo_model()

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
