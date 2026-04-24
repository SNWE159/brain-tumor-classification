import streamlit as st
import numpy as np
from PIL import Image
import time
import os
import io

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI · Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Lazy imports (heavy libs load only when needed) ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Download all four models from HuggingFace and cache them."""
    from huggingface_hub import hf_hub_download
    import tensorflow as tf

    REPO_ID = "Siya-34/brain-tumor-classification"
    MODEL_FILES = {
        "EfficientNetB0": "effnet_best.keras",
        "ResNet-50":      "resnet50_brain_tumor_final.keras",
        "VGG-16":         "vgg16_brain_tumor_final.keras",
        "MobileNetV2":    "mobilenetv2_best.keras",
    }

    models = {}
    for name, filename in MODEL_FILES.items():
        path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        models[name] = tf.keras.models.load_model(path)
    return models

# ── Class labels ────────────────────────────────────────────────────────────
CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CLASS_INFO = {
    "Glioma": {
        "icon": "⚠️",
        "color": "#FF4B4B",
        "desc": "A tumor arising from glial cells in the brain or spine. Can be low-grade (slow-growing) or high-grade (aggressive).",
        "severity": "High",
    },
    "Meningioma": {
        "icon": "🔶",
        "color": "#FFA500",
        "desc": "A tumor forming on the membranes surrounding the brain and spinal cord. Usually benign and slow-growing.",
        "severity": "Moderate",
    },
    "No Tumor": {
        "icon": "✅",
        "color": "#00C853",
        "desc": "No evidence of tumor detected in the MRI scan. Results appear within normal range.",
        "severity": "None",
    },
    "Pituitary": {
        "icon": "🔵",
        "color": "#2979FF",
        "desc": "A tumor in the pituitary gland at the brain's base. Often benign; can affect hormone production.",
        "severity": "Moderate",
    },
}

# ── Image preprocessing ──────────────────────────────────────────────────────
def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Ensemble prediction ──────────────────────────────────────────────────────
def predict_ensemble(models: dict, img_array: np.ndarray):
    all_probs = []
    individual = {}
    for name, model in models.items():
        probs = model.predict(img_array, verbose=0)[0]
        all_probs.append(probs)
        pred_idx = int(np.argmax(probs))
        individual[name] = {
            "label":      CLASS_LABELS[pred_idx],
            "confidence": float(probs[pred_idx]) * 100,
            "probs":      probs.tolist(),
        }
    avg_probs = np.mean(all_probs, axis=0)
    final_idx = int(np.argmax(avg_probs))
    return {
        "label":       CLASS_LABELS[final_idx],
        "confidence":  float(avg_probs[final_idx]) * 100,
        "all_probs":   avg_probs.tolist(),
        "individual":  individual,
    }

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ───────────────────────────────────────── */
:root {
    --bg:        #070B14;
    --surface:   #0D1424;
    --border:    rgba(99,179,237,0.15);
    --cyan:      #63B3ED;
    --cyan-glow: rgba(99,179,237,0.25);
    --red:       #FF4B4B;
    --orange:    #FFA500;
    --green:     #00C853;
    --blue:      #2979FF;
    --text:      #E2E8F0;
    --muted:     #718096;
    --radius:    16px;
}

/* ── Global Reset ─────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"],
[data-testid="stToolbar"],
footer { display: none !important; }

/* hide Streamlit default elements */
#MainMenu { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px; }

/* ── Hero Banner ──────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 3.5rem 2rem 2rem;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(99,179,237,0.12) 0%, transparent 70%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.1);
    border: 1px solid var(--cyan);
    color: var(--cyan);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    margin: 0 0 0.5rem;
    background: linear-gradient(135deg, #fff 40%, var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.7;
}
.hero-dots {
    display: flex;
    justify-content: center;
    gap: 6px;
    margin-top: 2rem;
}
.hero-dots span {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--cyan);
    opacity: 0.4;
}
.hero-dots span:nth-child(2) { opacity: 0.7; }
.hero-dots span:nth-child(3) { opacity: 1; }

/* ── Upload Zone ──────────────────────────────────────────── */
.upload-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 2.5rem;
    text-align: center;
    background: var(--surface);
    transition: border-color .3s;
}
.upload-zone:hover { border-color: var(--cyan); }
.upload-icon { font-size: 3rem; margin-bottom: 0.8rem; }
.upload-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.4rem;
}
.upload-sub { color: var(--muted); font-size: 0.88rem; }

/* ── Section Heading ──────────────────────────────────────── */
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1.2rem;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Model Card ───────────────────────────────────────────── */
.model-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
    transition: border-color .25s, transform .2s;
    position: relative;
    overflow: hidden;
}
.model-card:hover {
    border-color: var(--cyan);
    transform: translateY(-2px);
}
.model-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: var(--cyan);
    border-radius: 4px 0 0 4px;
}
.model-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: #fff;
    margin-bottom: 0.4rem;
}
.model-pred {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
}
.model-label {
    font-size: 1.05rem;
    font-weight: 600;
}
.model-conf {
    font-size: 0.85rem;
    color: var(--muted);
    background: rgba(255,255,255,0.04);
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
}
.bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--cyan), #a8d8f0);
    transition: width 1s cubic-bezier(.4,0,.2,1);
}

/* ── Final Result Card ────────────────────────────────────── */
.result-card {
    border-radius: 20px;
    padding: 2.4rem 2.8rem;
    margin: 2rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.6rem; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 0.3rem;
}
.result-conf {
    font-size: 1rem;
    opacity: 0.8;
    margin-bottom: 1rem;
}
.result-desc {
    font-size: 0.92rem;
    line-height: 1.7;
    max-width: 480px;
    margin: 0 auto 1.2rem;
    opacity: 0.85;
}
.severity-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    border: 1px solid currentColor;
}

/* ── Probability Distribution ─────────────────────────────── */
.prob-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.9rem;
    margin-top: 0.8rem;
}
.prob-item {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.prob-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.prob-name { font-size: 0.85rem; font-weight: 500; }
.prob-pct  { font-size: 0.85rem; font-weight: 700; }
.prob-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 5px;
    overflow: hidden;
}
.prob-bar { height: 100%; border-radius: 999px; }

/* ── Disclaimer ───────────────────────────────────────────── */
.disclaimer {
    background: rgba(255,200,0,0.05);
    border: 1px solid rgba(255,200,0,0.2);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #FFD600;
    font-size: 0.84rem;
    line-height: 1.65;
    margin-top: 2rem;
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;
}

/* ── Streamlit widget overrides ───────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan) !important;
}
.stButton>button {
    background: linear-gradient(135deg, var(--cyan), #3182CE) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(99,179,237,0.3) !important;
    transition: opacity .2s, transform .2s !important;
}
.stButton>button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stImage"] img {
    border-radius: var(--radius);
    border: 1px solid var(--border);
}
.stSpinner > div { border-top-color: var(--cyan) !important; }

/* ── Stats strip ──────────────────────────────────────────── */
.stats-strip {
    display: flex;
    gap: 1.2rem;
    margin-bottom: 2rem;
}
.stat-box {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--cyan);
}
.stat-label { font-size: 0.78rem; color: var(--muted); margin-top: 0.15rem; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🧠 Deep Learning · Medical Imaging</div>
    <h1>NeuroScan AI</h1>
    <p>Upload a brain MRI scan and let four state-of-the-art neural networks
       analyze it in seconds — then combine their wisdom into one confident diagnosis.</p>
    <div class="hero-dots"><span></span><span></span><span></span></div>
</div>
""", unsafe_allow_html=True)

# ── STATS STRIP ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-strip">
    <div class="stat-box">
        <div class="stat-num">4</div>
        <div class="stat-label">Neural Networks</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">4</div>
        <div class="stat-label">Tumor Classes</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">Ensemble</div>
        <div class="stat-label">Fusion Strategy</div>
    </div>
    <div class="stat-box">
        <div class="stat-num">MRI</div>
        <div class="stat-label">Input Modality</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    st.markdown('<div class="section-heading">📁 Upload MRI Scan</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded MRI", use_container_width=True)

        st.markdown(f"""
        <div style="background:var(--surface);border:1px solid var(--border);
                    border-radius:12px;padding:0.9rem 1.2rem;margin-top:0.8rem;
                    font-size:0.84rem;color:var(--muted);">
            📄 <strong style="color:#fff;">{uploaded.name}</strong><br>
            Dimensions: {img.width} × {img.height} px &nbsp;|&nbsp; Mode: {img.mode}
        </div>
        """, unsafe_allow_html=True)

        run_btn = st.button("🔬 Run Ensemble Analysis", use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">🫁</div>
            <div class="upload-title">Drop your MRI scan here</div>
            <div class="upload-sub">Supports JPG · PNG · BMP · TIFF</div>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

    # Model info accordion
    with st.expander("ℹ️ About the Models"):
        st.markdown("""
        | Model | Architecture | Specialty |
        |---|---|---|
        | EfficientNetB0 | EfficientNet | Accuracy & efficiency |
        | ResNet-50 | Residual Net | Deep feature extraction |
        | VGG-16 | VGGNet | Fine-grained texture |
        | MobileNetV2 | MobileNet | Speed & lightweight |

        **Ensemble method:** Soft-voting — averaged class probabilities from all four models produce a single, more robust prediction.
        """)

# ── RIGHT COLUMN ──────────────────────────────────────────────────────────────
with right_col:
    if uploaded and run_btn:
        # ── Load Models ──
        with st.spinner("⚡ Loading neural networks (first run may take ~30 s)…"):
            models = load_models()

        # ── Preprocess & Predict ──
        img_array = preprocess_image(img)

        progress_bar = st.progress(0, text="Running models…")
        results = {}
        all_probs_list = []

        import tensorflow as tf

        model_names = list(models.keys())
        for i, (name, model) in enumerate(models.items()):
            progress_bar.progress((i + 1) / len(models),
                                  text=f"Analyzing with {name}…")
            probs = model.predict(img_array, verbose=0)[0]
            all_probs_list.append(probs)
            pred_idx = int(np.argmax(probs))
            results[name] = {
                "label":      CLASS_LABELS[pred_idx],
                "confidence": float(probs[pred_idx]) * 100,
                "probs":      probs.tolist(),
            }
            time.sleep(0.15)  # brief pause for UX

        progress_bar.empty()

        # Ensemble
        avg_probs   = np.mean(all_probs_list, axis=0)
        final_idx   = int(np.argmax(avg_probs))
        final_label = CLASS_LABELS[final_idx]
        final_conf  = float(avg_probs[final_idx]) * 100
        info        = CLASS_INFO[final_label]

        # ── FINAL RESULT ──────────────────────────────────────────────
        st.markdown('<div class="section-heading">🎯 Ensemble Diagnosis</div>',
                    unsafe_allow_html=True)

        sev_colors = {"None": "#00C853", "Moderate": "#FFA500", "High": "#FF4B4B"}
        sev_col    = sev_colors.get(info["severity"], "#888")

        st.markdown(f"""
        <div class="result-card"
             style="background:linear-gradient(135deg,
                    {info['color']}18 0%, {info['color']}08 100%);
                    border:2px solid {info['color']}55;">
            <div class="result-icon">{info['icon']}</div>
            <div class="result-label" style="color:{info['color']};">{final_label}</div>
            <div class="result-conf">Ensemble confidence: <strong>{final_conf:.1f}%</strong></div>
            <div class="result-desc">{info['desc']}</div>
            <span class="severity-badge" style="color:{sev_col};border-color:{sev_col}40;">
                Severity: {info['severity']}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── CLASS PROBABILITIES ───────────────────────────────────────
        st.markdown('<div class="section-heading">📊 Class Probabilities</div>',
                    unsafe_allow_html=True)

        prob_colors = {
            "Glioma":     "#FF4B4B",
            "Meningioma": "#FFA500",
            "No Tumor":   "#00C853",
            "Pituitary":  "#2979FF",
        }

        html_probs = '<div class="prob-grid">'
        for label, prob in zip(CLASS_LABELS, avg_probs):
            pct = prob * 100
            col = prob_colors[label]
            ico = CLASS_INFO[label]["icon"]
            html_probs += f"""
            <div class="prob-item">
                <div class="prob-header">
                    <span class="prob-name">{ico} {label}</span>
                    <span class="prob-pct" style="color:{col};">{pct:.1f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar"
                         style="width:{pct:.1f}%;background:{col};"></div>
                </div>
            </div>"""
        html_probs += "</div>"
        st.markdown(html_probs, unsafe_allow_html=True)

        # ── INDIVIDUAL MODEL CARDS ────────────────────────────────────
        st.markdown('<div class="section-heading">🤖 Individual Model Predictions</div>',
                    unsafe_allow_html=True)

        for name, res in results.items():
            lbl  = res["label"]
            conf = res["confidence"]
            col  = prob_colors[lbl]
            ico  = CLASS_INFO[lbl]["icon"]
            st.markdown(f"""
            <div class="model-card">
                <div class="model-name">{name}</div>
                <div class="model-pred">
                    <span class="model-label" style="color:{col};">{ico} {lbl}</span>
                    <span class="model-conf">{conf:.1f}% confidence</span>
                </div>
                <div class="bar-wrap">
                    <div class="bar-fill" style="width:{conf:.1f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── DISCLAIMER ────────────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer">
            <span style="font-size:1.2rem;">⚠️</span>
            <span><strong>Medical Disclaimer:</strong> This tool is intended for
            research and educational purposes only. It is not a substitute for
            professional medical diagnosis, advice, or treatment. Always consult
            a qualified healthcare professional for medical decisions.</span>
        </div>
        """, unsafe_allow_html=True)

    elif not uploaded:
        # placeholder right panel
        st.markdown("""
        <div style="height:100%;display:flex;flex-direction:column;
                    justify-content:center;align-items:center;
                    padding:4rem 2rem;text-align:center;">
            <div style="font-size:5rem;margin-bottom:1.5rem;opacity:0.25;">🧠</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;
                        font-weight:700;color:#fff;opacity:0.4;margin-bottom:0.6rem;">
                Awaiting MRI Input
            </div>
            <div style="color:#718096;font-size:0.9rem;max-width:300px;line-height:1.6;">
                Upload a brain MRI scan on the left and click
                <em>Run Ensemble Analysis</em> to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)
