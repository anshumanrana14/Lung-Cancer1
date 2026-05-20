import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LungAI — Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #05080f;
    color: #e8eaf0;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 40% at 50% -10%, rgba(0,180,255,0.07) 0%, transparent 70%),
        radial-gradient(ellipse 60% 30% at 100% 80%, rgba(0,255,150,0.04) 0%, transparent 60%),
        #05080f;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }

/* ── Typography Global ── */
h1, h2, h3, h4, p, span, label, div {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hero Header ── */
.hero-section {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    position: relative;
}

.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    color: #00b4d8;
    text-transform: uppercase;
    margin-bottom: 1rem;
    opacity: 0.9;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800;
    line-height: 0.95;
    margin: 0 0 1.2rem;
    background: linear-gradient(135deg, #ffffff 0%, #a8d8ea 50%, #00b4d8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    color: rgba(200,215,230,0.65);
    max-width: 460px;
    margin: 0 auto;
    line-height: 1.7;
}

.hero-divider {
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, #00b4d8, #00ff94);
    margin: 2rem auto 0;
    border-radius: 2px;
}

/* ── Stat Bar ── */
.stat-bar {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    padding: 1.5rem 0 2rem;
    flex-wrap: wrap;
}

.stat-item {
    text-align: center;
    position: relative;
}

.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00d4f5;
    line-height: 1;
}

.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(200,215,230,0.45);
    margin-top: 0.3rem;
}

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.033);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,180,255,0.35), transparent);
}

.card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(200,215,230,0.5);
    margin: 0 0 1.2rem;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    width: 100%;
}

[data-testid="stFileUploader"] > div {
    border: 1.5px dashed rgba(0,180,255,0.3) !important;
    border-radius: 14px !important;
    background: rgba(0,180,255,0.03) !important;
    transition: all 0.25s ease;
    padding: 2.5rem 1rem !important;
}

[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(0,180,255,0.6) !important;
    background: rgba(0,180,255,0.06) !important;
}

[data-testid="stFileUploader"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    color: rgba(200,215,230,0.6) !important;
}

/* ── Image Display ── */
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.07);
}

/* ── Predict Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0077b6, #00b4d8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,180,255,0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(0,180,255,0.4) !important;
    background: linear-gradient(135deg, #0090cc, #00c8f0) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result Panels ── */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.2rem;
    border-radius: 100px;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-bottom: 0.4rem;
}

.badge-malignant {
    background: rgba(255,60,80,0.15);
    border: 1px solid rgba(255,60,80,0.4);
    color: #ff6b7a;
}

.badge-benign {
    background: rgba(0,255,148,0.1);
    border: 1px solid rgba(0,255,148,0.35);
    color: #00e589;
}

.badge-normal {
    background: rgba(0,180,255,0.1);
    border: 1px solid rgba(0,180,255,0.35);
    color: #00c8f0;
}

.badge-other {
    background: rgba(180,130,255,0.1);
    border: 1px solid rgba(180,130,255,0.35);
    color: #c49aff;
}

.confidence-value {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
    color: #ffffff;
}

.confidence-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(200,215,230,0.45);
    margin-top: 0.3rem;
}

/* ── Probability Bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.9rem;
}

.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(200,215,230,0.6);
    width: 90px;
    flex-shrink: 0;
}

.prob-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 3px;
    overflow: hidden;
    position: relative;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.25, 1, 0.5, 1);
}

.prob-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: rgba(200,215,230,0.75);
    width: 48px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Warning Banner ── */
.warning-banner {
    background: rgba(255,190,50,0.07);
    border: 1px solid rgba(255,190,50,0.2);
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    margin-top: 1.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: rgba(255,210,100,0.8);
    line-height: 1.6;
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
}

/* ── Model Info Tag ── */
.model-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(200,215,230,0.4);
}

.dot-green { width: 6px; height: 6px; background: #00e589; border-radius: 50%; display: inline-block; }
.dot-blue  { width: 6px; height: 6px; background: #00c8f0; border-radius: 50%; display: inline-block; }

/* ── Streamlit overrides ── */
[data-testid="stSpinner"] { color: #00b4d8 !important; }
[data-testid="column"] { gap: 0 !important; }
.stAlert { border-radius: 10px !important; }

/* spinner text */
[data-testid="stSpinner"] p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    color: rgba(200,215,230,0.55) !important;
}

/* ── Scanline Overlay ── */
.scanline {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 9999;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.015) 2px,
        rgba(0,0,0,0.015) 4px
    );
}

/* ── Hide streamlit extras ── */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Column spacing ── */
[data-testid="stHorizontalBlock"] { gap: 1.2rem; }
</style>

<div class="scanline"></div>
""", unsafe_allow_html=True)


# ─── Model Setup ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

CLASS_NAMES = ["Benign", "Malignant", "Normal", "Other"]
CLASS_COLORS = {
    "Benign":    ("#00e589", "badge-benign"),
    "Malignant": ("#ff6b7a", "badge-malignant"),
    "Normal":    ("#00c8f0", "badge-normal"),
    "Other":     ("#c49aff", "badge-other"),
}
BAR_COLORS = {
    "Benign":    "linear-gradient(90deg,#00c870,#00e589)",
    "Malignant": "linear-gradient(90deg,#e03048,#ff6b7a)",
    "Normal":    "linear-gradient(90deg,#0099cc,#00c8f0)",
    "Other":     "linear-gradient(90deg,#9060e0,#c49aff)",
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_image(image, model, device):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return (
        CLASS_NAMES[predicted.item()],
        confidence.item() * 100,
        probs.cpu().numpy()[0]
    )


# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">Lung Cancer<br>Detection</h1>
    <p class="hero-sub">Upload a CT scan image.</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ─── Main Layout ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown('<div class="card"><div class="card-title">📂 Image Input</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop scan here or click to browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        st.markdown(f"""
        <div style="margin-top:0.8rem; display:flex; gap:0.6rem; flex-wrap:wrap;">
            <span class="model-tag"><span class="dot-green"></span>{uploaded_file.name}</span>
            <span class="model-tag"><span class="dot-blue"></span>{image.size[0]}×{image.size[1]}px</span>
            <span class="model-tag"><span class="dot-blue"></span>{image.mode}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card"><div class="card-title">🔬 Analysis Output</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: rgba(200,215,230,0.25);">
            <div style="font-size:2.5rem; margin-bottom:0.8rem;">🫁</div>
            <div style="font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.15em; text-transform:uppercase;">
                Awaiting image upload
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        predict_btn = st.button("⟶ Run Analysis", use_container_width=True)

        if predict_btn:
            with st.spinner("Analyzing tissue sample..."):
                try:
                    model, device = load_model()
                    time.sleep(0.4)  # brief pause for UX
                    prediction, confidence, probs = predict_image(image, model, device)

                    color, badge_class = CLASS_COLORS[prediction]
                    bar_gradient = BAR_COLORS[prediction]

                    # ── Result Badge ──
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0 1.2rem;">
                        <div style="font-family:'DM Mono',monospace; font-size:0.63rem; letter-spacing:0.2em;
                                    text-transform:uppercase; color:rgba(200,215,230,0.4); margin-bottom:0.5rem;">
                            Classification Result
                        </div>
                        <span class="result-badge {badge_class}">
                            {'⚠' if prediction == 'Malignant' else '✓'} &nbsp;{prediction}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Confidence Score ──
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown(f"""
                        <div>
                            <div class="confidence-value">{confidence:.1f}<span style="font-size:1.4rem; opacity:0.5">%</span></div>
                            <div class="confidence-label">Confidence Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        top_idx = np.argmax(probs)
                        second_idx = np.argsort(probs)[-2]
                        st.markdown(f"""
                        <div>
                            <div style="font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:0.15em;
                                        text-transform:uppercase; color:rgba(200,215,230,0.4); margin-bottom:0.4rem;">
                                Runner-up
                            </div>
                            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
                                        color:{CLASS_COLORS[CLASS_NAMES[second_idx]][0]};">
                                {CLASS_NAMES[second_idx]}
                            </div>
                            <div style="font-family:'DM Mono',monospace; font-size:0.75rem;
                                        color:rgba(200,215,230,0.5);">
                                {probs[second_idx]*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<div style='margin-top:1.4rem;'></div>", unsafe_allow_html=True)

                    # ── Probability Distribution ──
                    st.markdown("""
                    <div style="font-family:'DM Mono',monospace; font-size:0.63rem; letter-spacing:0.2em;
                                text-transform:uppercase; color:rgba(200,215,230,0.4); margin-bottom:0.9rem;">
                        Probability Distribution
                    </div>
                    """, unsafe_allow_html=True)

                    for cls_name, prob in zip(CLASS_NAMES, probs):
                        pct = prob * 100
                        bar_col = BAR_COLORS[cls_name]
                        is_top = cls_name == prediction
                        label_style = f"color:{CLASS_COLORS[cls_name][0]};" if is_top else ""
                        st.markdown(f"""
                        <div class="prob-row">
                            <div class="prob-label" style="{label_style}">{cls_name}</div>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill"
                                     style="width:{pct:.1f}%; background:{bar_col};
                                            {'box-shadow: 0 0 8px rgba(0,180,255,0.4);' if is_top else ''}">
                                </div>
                            </div>
                            <div class="prob-pct">{pct:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                except FileNotFoundError:
                    st.markdown("""
                    <div style="background:rgba(255,60,80,0.08); border:1px solid rgba(255,60,80,0.3);
                                border-radius:10px; padding:1rem 1.2rem; margin-top:0.5rem;
                                font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#ff6b7a;">
                        ⚠ <strong>model.pth not found.</strong><br>
                        <span style="opacity:0.7;">Place your trained model file in the same directory as app.py and restart.</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding:2.5rem 1rem; color:rgba(200,215,230,0.2);">
                <div style="font-family:'DM Mono',monospace; font-size:0.72rem; letter-spacing:0.15em; text-transform:uppercase;">
                    Click "Run Analysis" to begin
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="warning-banner">
    <span>⚠</span>
    <span>This tool is intended for <strong>research and educational purposes only</strong>.
    It is not a substitute for professional medical diagnosis.
    Always consult a licensed physician for clinical decisions.</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:2rem; padding-bottom:2rem;
            font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.15em;
            text-transform:uppercase; color:rgba(200,215,230,0.2);">
    LungAI · ResNet-18 · 4-Class Classification · For Research Only
</div>
""", unsafe_allow_html=True)
