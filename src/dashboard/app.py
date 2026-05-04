"""
NSCLC Multimodal Prediction Dashboard
======================================
Streamlit-based research dashboard for AI-Based Prediction of NSCLC Recurrence
as a Treatment Response Proxy Using Multimodal Histopathology and Clinical Data.

Run from project root:
    streamlit run src/dashboard/app.py
"""

import os
import sys
import json
import numpy as np
import torch
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import shap
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.resnet_encoder import build_model as build_resnet
from src.models.fusion_model import WeightedFusionModel          # ← FIX 1: import correct class
from src.preprocess.clinical_processor import ClinicalProcessor
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.explainability.shap_explainer import ClinicalPredictionWrapper

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NSCLC AI Prediction System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
    color: white;
}
section[data-testid="stSidebar"] * { color: white !important; }

section[data-testid="stSidebar"] .stButton > button {
    color: white !important;
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 8px !important;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.16) !important;
    border-color: rgba(255,255,255,0.35) !important;
}
section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stButton > button span,
section[data-testid="stSidebar"] .stButton > button div {
    color: white !important;
}

.nav-card {
    background: white;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1.5px solid #e8eaf0;
    height: 180px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.nav-card .title {
    font-size: 17px; font-weight: 700;
    color: #1a237e; margin-bottom: 6px;
}
.nav-card .desc { font-size: 13px; color: #5f6368; line-height: 1.4; }

.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1565C0 60%, #1976D2 100%);
    border-radius: 16px;
    padding: 48px 40px;
    color: white;
    margin-bottom: 32px;
}
.hero h1 { font-size: 28px; font-weight: 800; margin: 0 0 10px 0; }
.hero p  { font-size: 15px; opacity: 0.88; margin: 0; line-height: 1.6; }

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 4px solid #1565C0;
}
.metric-card .val { font-size: 32px; font-weight: 800; color: #1565C0; }
.metric-card .lbl { font-size: 13px; color: #5f6368; margin-top: 4px; }

.badge-high {
    background: #ffebee; color: #c62828;
    border-radius: 8px; padding: 16px 24px;
    font-size: 20px; font-weight: 700;
    text-align: center; border-left: 5px solid #c62828;
}
.badge-low {
    background: #e8f5e9; color: #2e7d32;
    border-radius: 8px; padding: 16px 24px;
    font-size: 20px; font-weight: 700;
    text-align: center; border-left: 5px solid #2e7d32;
}

.section-header {
    font-size: 22px; font-weight: 700;
    color: #1a237e; margin-bottom: 4px;
    border-left: 4px solid #1565C0;
    padding-left: 12px;
}
.section-sub {
    font-size: 14px; color: #5f6368;
    margin-bottom: 20px; padding-left: 16px;
}

.info-box {
    background: #e3f2fd; border-radius: 10px;
    padding: 16px 20px; border-left: 4px solid #1565C0;
    font-size: 14px; color: #1a237e;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### NSCLC AI Prediction System")
    st.markdown("---")
    nav_items = [
        ("Home",               ":material/home:"),
        ("Predict Recurrence", ":material/biotech:"),
        ("Model Results",      ":material/bar_chart:"),
        ("Explainability",     ":material/psychology:"),
        ("About",              ":material/info:"),
    ]
    for page_name, nav_icon in nav_items:
        if st.button(page_name, key=f"nav_{page_name}",
                     icon=nav_icon, use_container_width=True):
            st.session_state.page = page_name

    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px;opacity:0.7'>Plymouth Index: 10953013<br>"
        "BSc (Hons) Software Engineering<br>PUSL3190 Computing Project</div>",
        unsafe_allow_html=True,
    )

page = st.session_state.page

# ─── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── ResNet encoder ──────────────────────────────────────────────────────
    resnet = build_resnet(num_classes=2, pretrained=False, embedding_dim=256).to(device)
    ckpt   = torch.load("outputs/models/resnet_best.pth", map_location=device)
    resnet.load_state_dict(ckpt["model_state"])
    resnet.eval()

    # ── FIX 2: instantiate WeightedFusionModel to match the saved checkpoint ──
    # fusion_best.pth was saved from WeightedFusionModel which has:
    #   image_branch, clinical_branch, alpha
    # NOT LateFusionModel which has image_projector, clinical_encoder, fusion_head
    fusion = WeightedFusionModel(
        image_embedding_dim=256,
        clinical_input_dim=6,
    ).to(device)
    f_ckpt = torch.load("outputs/models/fusion_best.pth", map_location=device)
    fusion.load_state_dict(f_ckpt["model_state"])
    fusion.eval()

    img_emb  = np.load("data/features/image_embeddings.npy").astype(np.float32)
    mean_emb = img_emb.mean(axis=0)
    return resnet, fusion, device, mean_emb


@st.cache_resource
def load_processor():
    # ── FIX 3: use deduplicated CSV (155 unique patients), not original 234-row file
    p = ClinicalProcessor("data/metadata/tcga_clinical_master_deduped.csv")
    p.get_features_and_labels(fit_scaler=True)
    return p


@st.cache_data
def load_results():
    with open("outputs/results/fusion_comparison.json") as f:
        return json.load(f)


# ─── Helpers ──────────────────────────────────────────────────────────────────

STAGE_OPTIONS = {
    "Stage I": 1, "Stage IA": 1, "Stage IB": 1,
    "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2,
    "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3,
    "Stage IV": 4,
}

import torchvision.transforms as T


def preprocess_image(pil_img):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(pil_img).unsqueeze(0)


def get_image_embedding(resnet, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        resnet.set_mode("encoder")
        emb = resnet(image_tensor)
    return emb


def predict_recurrence(fusion_model, img_emb, clin_feat, device):
    img_t  = torch.tensor(img_emb,   dtype=torch.float32).to(device)
    clin_t = torch.tensor(clin_feat, dtype=torch.float32).to(device)
    if img_t.dim() == 1:
        img_t  = img_t.unsqueeze(0)
        clin_t = clin_t.unsqueeze(0)
    with torch.no_grad():
        prob = fusion_model(img_t, clin_t).item()
    return prob


def make_gradcam_figure(resnet, image_tensor, original_np, device):
    image_tensor = image_tensor.to(device)
    resnet.set_mode("classifier")
    gradcam = GradCAM(resnet, target_layer="layer4")
    heatmap, pred_class = gradcam.generate(image_tensor)
    overlaid = overlay_heatmap(original_np, heatmap, alpha=0.45)
    gradcam.remove_hooks()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_np)
    axes[0].set_title("Original Patch", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
    axes[1].axis("off")
    axes[2].imshow(overlaid)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")
    plt.suptitle("Grad-CAM: Tissue Regions Driving Prediction",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig, ["LUAD", "LUSC"][pred_class]


def make_shap_figure(fusion_model, clin_feat, processor, mean_emb, device):
    wrapper = ClinicalPredictionWrapper(fusion_model, mean_emb, device)
    X_all, _, feat_names, *_ = processor.get_features_and_labels(fit_scaler=False)

    bg_idx    = np.random.choice(len(X_all), size=min(15, len(X_all)), replace=False)
    bg        = X_all[bg_idx]
    explainer = shap.KernelExplainer(wrapper.predict, bg)
    shap_vals = explainer.shap_values(
        clin_feat.reshape(1, -1), nsamples=80, silent=True
    )

    sorted_i = np.argsort(np.abs(shap_vals).flatten())
    colors   = ["#c62828" if v > 0 else "#1565C0" for v in shap_vals.flatten()]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(
        [feat_names[i] for i in sorted_i],
        shap_vals.flatten()[sorted_i],
        color=[colors[i] for i in sorted_i],
        edgecolor="white",
        height=0.55,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on recurrence probability)", fontsize=10)
    ax.set_title("Clinical Feature Contribution (SHAP)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def prob_gauge(prob):
    fig, ax = plt.subplots(figsize=(5, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.barh(0.5, 1,    height=0.35, color="#f0f0f0", left=0)
    color = "#c62828" if prob > 0.5 else "#2e7d32"
    ax.barh(0.5, prob, height=0.35, color=color,     left=0)
    ax.text(
        min(prob + 0.02, 0.92), 0.5,
        f"{prob * 100:.1f}%",
        va="center", fontsize=16, fontweight="bold", color=color,
    )
    ax.text(0,   0.1, "0%",   fontsize=9, color="#999")
    ax.text(1,   0.1, "100%", fontsize=9, color="#999", ha="right")
    ax.text(0.5, 0.92, "Recurrence Probability",
            ha="center", fontsize=11, fontweight="bold", color="#1a237e")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>NSCLC Multimodal AI Prediction System</h1>
        <p>AI-Based Prediction of NSCLC Recurrence as a Treatment Response Proxy<br>
        Using Multimodal Histopathology and Clinical Data &mdash; PUSL3190 Computing Project</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        results    = load_results()
        fusion_auc = results["fusion"]["auc_mean"]
    except Exception:
        fusion_auc = 0.558

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="val">{fusion_auc:.3f}</div>
            <div class="lbl">Weighted Fusion AUC</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="val">99.9%</div>
            <div class="lbl">Subtype Accuracy</div></div>""",
            unsafe_allow_html=True)
    with c3:
        # ── FIX 4: correct patient count
        st.markdown("""<div class="metric-card">
            <div class="val">155</div>
            <div class="lbl">Unique Patients (deduped)</div></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <div class="val">10K</div>
            <div class="lbl">Image Patches</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Navigate to a Section")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""<div class="nav-card">
            <div class="title">Predict Recurrence</div>
            <div class="desc">Upload a histopathology patch and enter clinical data
            to get a recurrence prediction with full explainability</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to Predict", key="go_predict", use_container_width=True):
            st.session_state.page = "Predict Recurrence"
            st.rerun()

    with col2:
        st.markdown("""<div class="nav-card">
            <div class="title">Model Results</div>
            <div class="desc">Compare fusion vs image-only vs clinical-only model
            performance with metrics and ROC curves</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to Results", key="go_results", use_container_width=True):
            st.session_state.page = "Model Results"
            st.rerun()

    with col3:
        st.markdown("""<div class="nav-card">
            <div class="title">Explainability</div>
            <div class="desc">View Grad-CAM tissue heatmaps and SHAP clinical
            feature importance across all patients</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to Explainability", key="go_xai", use_container_width=True):
            st.session_state.page = "Explainability"
            st.rerun()

    with col4:
        st.markdown("""<div class="nav-card">
            <div class="title">About</div>
            <div class="desc">Project background, research gap, system architecture
            and academic details</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Go to About", key="go_about", use_container_width=True):
            st.session_state.page = "About"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        <b>How to use:</b> Navigate to <b>Predict Recurrence</b> to upload a
        histopathology patch and enter patient clinical data. The system will
        predict recurrence risk and explain which tissue regions and clinical
        features drove the prediction.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT RECURRENCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Predict Recurrence":
    st.markdown('<div class="section-header">Predict Recurrence Risk</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Upload a histopathology image patch and enter patient clinical data</div>',
        unsafe_allow_html=True,
    )

    # ── Research disclaimer ──────────────────────────────────────────────────
    st.info(
        "**Research tool only.** This system is a proof-of-concept trained on n=155 "
        "deduplicated TCGA patients. Predictions are not validated for clinical use. "
        "Image features are proxy-matched by subtype; patient-specific slide linkage "
        "is a documented future work item.",
        icon="ℹ️",
    )

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### Histopathology Image")
        uploaded_img = st.file_uploader(
            "Upload a histopathology patch (JPG or PNG, 224×224 or larger)",
            type=["jpg", "jpeg", "png"],
            key="img_upload",
        )
        if uploaded_img:
            pil_img = Image.open(uploaded_img).convert("RGB")
            st.image(pil_img, caption="Uploaded patch", use_container_width=True)

        st.markdown("#### Clinical Data")
        with st.form("clinical_form"):
            age     = st.slider("Patient Age", 30, 90, 65)
            gender  = st.selectbox("Gender", ["Male", "Female"])
            subtype = st.selectbox("NSCLC Subtype", ["LUAD", "LUSC"])
            stage   = st.selectbox("Tumour Stage", list(STAGE_OPTIONS.keys()))
            days_fu = st.slider("Days of Follow-up", 0, 3000, 365)
            submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    with col_result:
        if submitted and uploaded_img:
            with st.spinner("Running multimodal prediction..."):
                try:
                    resnet, fusion, device, mean_emb = load_models()
                    processor = load_processor()

                    pil_img    = Image.open(uploaded_img).convert("RGB")
                    img_tensor = preprocess_image(pil_img).to(device)
                    img_np     = np.array(pil_img.resize((224, 224)))

                    # ── Input validation: subtype confidence gate ──────────────────────
                    with torch.no_grad():
                        resnet.set_mode("classifier")
                        logits = resnet(img_tensor)
                        probs  = torch.softmax(logits, dim=1)
                        conf, pred_idx = probs.max(dim=1)
                        conf = conf.item()
                        pred_subtype_label = ["LUAD", "LUSC"][pred_idx.item()]

                    CONFIDENCE_THRESHOLD = 0.70

                    if conf < CONFIDENCE_THRESHOLD:
                        st.warning(
                            f"⚠️ **Low tissue confidence ({conf*100:.1f}%).** "
                            f"The uploaded image does not appear to be a valid "
                            f"histopathology patch (expected LUAD or LUSC tissue). "
                            f"Prediction has been blocked. Please upload a lung "
                            f"histopathology image patch."
                        )
                        st.info(
                            "This validation step is intentional. The ResNet-50 encoder "
                            "was trained exclusively on histopathological tissue classes "
                            "(LC25000 dataset). Out-of-distribution inputs produce "
                            "meaningless embeddings and unreliable predictions."
                        )
                        st.stop()
                    # ── End input validation ───────────────────────────────────────────

                    img_emb = get_image_embedding(resnet, img_tensor, device)

                    raw = np.array([[
                        float(age),
                        0.0 if gender == "Male" else 1.0,
                        0.0 if subtype == "LUAD" else 1.0,
                        float(STAGE_OPTIONS[stage]),
                        float(days_fu),
                    ]], dtype=np.float32)
                    clin_scaled = processor.scaler.transform(raw).astype(np.float32)

                    prob = predict_recurrence(
                        fusion, img_emb.cpu().numpy(), clin_scaled, device
                    )

                    st.markdown("#### Prediction Result")
                    st.pyplot(prob_gauge(prob))

                    # Show which subtype the encoder detected
                    st.caption(
                        f"Image validated — predicted subtype: **{pred_subtype_label}** "
                        f"(confidence: {conf*100:.1f}%)"
                    )

                    if prob > 0.5:
                        st.markdown(
                            f'<div class="badge-high">High Recurrence Risk &nbsp;|&nbsp; {prob*100:.1f}%</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="badge-low">Low Recurrence Risk &nbsp;|&nbsp; {prob*100:.1f}%</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown("#### Grad-CAM Heatmap")
                    with st.spinner("Generating Grad-CAM..."):
                        fig_gc, _ = make_gradcam_figure(
                            resnet, img_tensor, img_np, device
                        )
                        st.pyplot(fig_gc)
                        st.caption(
                            f"Predicted subtype: {pred_subtype_label} — "
                            "warm colours show discriminative tissue regions"
                        )

                    st.markdown("#### Clinical Feature Importance (SHAP)")
                    with st.spinner("Computing SHAP values..."):
                        fig_shap = make_shap_figure(
                            fusion, clin_scaled[0], processor, mean_emb, device
                        )
                        st.pyplot(fig_shap)
                        st.caption(
                            "Red = increases recurrence risk   "
                            "Blue = decreases recurrence risk"
                        )

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.info(
                        "If you see a state_dict mismatch error, ensure "
                        "`outputs/models/fusion_best.pth` was saved from "
                        "`WeightedFusionModel`. Check that `fusion_model.py` "
                        "exports `WeightedFusionModel` with `image_branch`, "
                        "`clinical_branch`, and `alpha` parameters.",
                        icon="🔧",
                    )

        elif submitted and not uploaded_img:
            st.warning("Please upload a histopathology image patch to run the prediction.")
        else:
            st.markdown("""<div class="info-box">
                Fill in the clinical data form and upload a histopathology patch,
                then click <b>Run Prediction</b> to see the recurrence risk,
                Grad-CAM heatmap, and SHAP explanation.
            </div>""", unsafe_allow_html=True)
            if os.path.exists("outputs/figures/gradcam_examples.png"):
                st.image(
                    "outputs/figures/gradcam_examples.png",
                    caption="Example Grad-CAM outputs",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Model Results":
    st.markdown('<div class="section-header">Model Performance Results</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">5-fold stratified cross-validation on deduplicated cohort (n=155 unique patients, seed=42)</div>',
        unsafe_allow_html=True,
    )

    try:
        results = load_results()

        rows = []
        label_map = {
            "fusion"        : "Weighted Fusion (Image + Clinical)",
            "image_only"    : "Image Only (ResNet-50)",
            "clinical_only" : "Clinical Only (MLP)",
        }
        for mtype, res in results.items():
            rows.append({
                "Model"     : label_map.get(mtype, mtype),
                "AUC"       : f"{res['auc_mean']:.4f} ± {res['auc_std']:.4f}",
                "F1 Score"  : f"{res['f1_mean']:.4f} ± {res['f1_std']:.4f}",
                "Accuracy"  : f"{res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}",
                "Precision" : f"{res['precision_mean']:.4f} ± {res['precision_std']:.4f}",
                "Recall"    : f"{res['recall_mean']:.4f} ± {res['recall_std']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── FIX 5: accurate key finding — clinical-only leads, not fusion
        st.markdown("""<div class="info-box">
            <b>Key finding:</b> Clinical-only (MLP) achieves the highest AUC (0.6199 ± 0.0561),
            while scalar weighted fusion (0.5578 ± 0.0692) underperforms — indicating that
            naively fused proxy-matched image embeddings introduce noise rather than
            complementary signal. The GMU gated fusion (AUC 0.6157) mitigates this
            degradation through adaptive per-patient modality weighting.
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ROC Curves")
            if os.path.exists("outputs/figures/fusion_roc_curves.png"):
                st.image("outputs/figures/fusion_roc_curves.png", use_container_width=True)
        with col2:
            st.markdown("#### Fusion Training Curves")
            if os.path.exists("outputs/figures/fusion_training_curves.png"):
                st.image("outputs/figures/fusion_training_curves.png", use_container_width=True)

        st.markdown("#### Subtype Classifier Training Curves (ResNet-50, LC25000)")
        if os.path.exists("outputs/figures/resnet_training_curves.png"):
            st.image("outputs/figures/resnet_training_curves.png", use_container_width=True)

        if os.path.exists("outputs/results/resnet_results.json"):
            with open("outputs/results/resnet_results.json") as f:
                rn = json.load(f)
            st.markdown("#### Subtype Classifier Results (sanity-check benchmark)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Test Accuracy", f"{rn['test_accuracy']}%")
            c2.metric("Test AUC",      str(rn["test_auc"]))
            c3.metric("Test F1",       str(rn["test_f1"]))
            st.caption(
                "Note: the subtype classifier (LUAD vs LUSC on LC25000) is a "
                "saturated benchmark, not the main research contribution."
            )

    except Exception as e:
        st.error(f"Could not load results: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Explainability":
    st.markdown('<div class="section-header">Model Explainability</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Grad-CAM spatial heatmaps and SHAP clinical feature importance</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["Grad-CAM (Image)", "SHAP (Clinical)"])

    with tab1:
        st.markdown("""
        **Grad-CAM** (Gradient-weighted Class Activation Mapping) highlights which
        regions of the histopathology image patch the model focused on when making its
        subtype classification. Warm colours (red and yellow) indicate regions of high
        importance.
        """)
        if os.path.exists("outputs/figures/gradcam_examples.png"):
            st.image("outputs/figures/gradcam_examples.png", use_container_width=True)

    with tab2:
        st.markdown("""
        **SHAP** (SHapley Additive exPlanations) quantifies the contribution of each
        clinical feature to the recurrence prediction. Tumour stage (stage_numeric) is
        the dominant predictor identified by the deep MLP. Features with higher mean
        absolute SHAP values have greater influence on the model output.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Feature Importance Bar Chart")
            if os.path.exists("outputs/figures/shap_summary_bar.png"):
                st.image("outputs/figures/shap_summary_bar.png", use_container_width=True)
        with col2:
            st.markdown("##### Beeswarm Plot")
            if os.path.exists("outputs/figures/shap_beeswarm.png"):
                st.image("outputs/figures/shap_beeswarm.png", use_container_width=True)

        st.markdown("##### Patient-level Waterfall Plot")
        if os.path.exists("outputs/figures/shap_waterfall_patient0.png"):
            st.image("outputs/figures/shap_waterfall_patient0.png", use_container_width=True)

        if os.path.exists("outputs/results/shap_importance.json"):
            with open("outputs/results/shap_importance.json") as f:
                shap_data = json.load(f)
            st.markdown("##### Feature Importance Ranking")
            imp = shap_data["feature_importance_shap"]
            df  = pd.DataFrame(
                list(imp.items()),
                columns=["Clinical Feature", "Mean Absolute SHAP Value"],
            )
            st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "About":
    st.markdown('<div class="section-header">About This Project</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        ### Project Overview

        This system is a proof-of-concept multimodal deep learning framework for
        predicting recurrence risk in Non-Small Cell Lung Cancer (NSCLC) patients
        as a proxy for treatment response.

        **Research Problem:** Current NSCLC predictive models rely on single data
        modalities — either histopathological images or clinical biomarkers — limiting
        their ability to capture the full complexity of tumour behaviour. Most
        frameworks also require high-end GPU hardware, limiting accessibility.

        **Proposed Solution:** A resource-aware late fusion architecture combining:
        - CNN-based feature extraction from histopathology patches (ResNet-50, offline)
        - Clinical biomarker encoding via a Multi-Layer Perceptron (MLP)
        - Adaptive Gated Multimodal Unit (GMU) for per-patient modality weighting
        - Explainable AI outputs via Grad-CAM and SHAP

        **Important limitation:** Histopathology images are proxy-matched to patients
        by NSCLC subtype (LUAD/LUSC) using the LC25000 dataset, as patient-linked
        TCGA whole-slide images require institutional compute beyond the 6GB GPU
        constraint. This means image embeddings carry subtype-level signal only,
        which is why clinical-only features currently lead on AUC.

        ### Datasets Used
        - **Kaggle LC25000** (Borkowski et al., 2019): 10,000 NSCLC histopathology
          patches (5,000 LUAD + 5,000 LUSC), used for encoder pretraining and
          proxy-matched image embeddings
        - **TCGA-LUAD / TCGA-LUSC** (GDC Portal): 155 unique patient clinical
          records after deduplication (originally 234 rows — data leakage bug
          found and corrected)

        ### Key Results (n=155, 5-fold stratified CV, seed=42)
        - Clinical-only MLP: AUC 0.6199 ± 0.0561 (best unimodal)
        - GMU gated fusion: AUC 0.6157 (95% CI: 0.483–0.669)
        - Weighted fusion: AUC 0.5578 ± 0.0692
        - Subtype classifier: 99.93% accuracy, AUC 1.0 (saturated benchmark)
        - Inference: 3.41 ± 1.83ms, 63,107 trainable parameters, 6GB GPU validated
        - Tumour stage (stage_numeric) identified as dominant clinical predictor (SHAP)
        """)

    with col2:
        st.markdown("""
        ### System Architecture

        | Component | Technology |
        |-----------|-----------|
        | Image Encoder | ResNet-50 (pretrained ImageNet) |
        | Clinical Encoder | 3-layer MLP |
        | Fusion Strategies | Weighted, Concat, GMU gated |
        | Explainability | Grad-CAM + SHAP |
        | Framework | PyTorch |
        | Dashboard | Streamlit |
        | Evaluation | 5-fold stratified CV |
        | Hardware target | 6GB GPU (VRAM-constrained) |

        ### Academic Details

        | Field | Detail |
        |-------|--------|
        | Module | PUSL3190 Computing Project |
        | Degree | BSc (Hons) Software Engineering |
        | University | University of Plymouth |
        | Delivery | SLIIT Sri Lanka |
        | Student ID | 10953013 |
        | Supervisor | Ms. M T A Wickramasinghe |

        ### Research Questions
        1. Does multimodal fusion improve AUC over unimodal baselines under a 6GB GPU constraint?
        2. Does GMU gated fusion improve over scalar-weighted late fusion on n=155?
        3. Which features are most predictive of NSCLC recurrence?
        """)