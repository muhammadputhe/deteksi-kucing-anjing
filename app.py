import io
from pathlib import Path
from html import escape

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
MAIN_ICON_PATH = ASSET_DIR / "Main_icon.png"

st.set_page_config(
    page_title="PawScan | Animal Recognition",
    page_icon=str(MAIN_ICON_PATH) if MAIN_ICON_PATH.exists() else "🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

YOLO_PATH = ROOT / "model" / "Haikal_Laporan 4.pt"
CLASSIFIER_PATH = ROOT / "best_model.h5"
DESKTOP_BG_PATH = ROOT / "background_pawscan.png"
MOBILE_BG_PATH = ROOT / "background_pawscan_mobile.png"
CAT_ICON_PATH = ASSET_DIR / "cat_icon.png"
DOG_ICON_PATH = ASSET_DIR / "dog_icon.png"
EMPTY_STATE_PATH = ASSET_DIR / "Ilustrasi_kosong.jpg"
LOADING_GIF_PATH = ASSET_DIR / "pawscan_loading.gif"
HOME_ICON_PATH = ASSET_DIR / "beranda.png"
ANALYSIS_ICON_PATH = ASSET_DIR / "analisis.png"
HELP_ICON_PATH = ASSET_DIR / "bantuan.png"
DELETE_ICON_PATH = ASSET_DIR / "hapus.png"
SETTINGS_ICON_PATH = ASSET_DIR / "pengaturan.png"
HISTORY_ICON_PATH = ASSET_DIR / "riwayat.png"
DOWNLOAD_ICON_PATH = ASSET_DIR / "unduh_hasil.png"
# Custom interactive method-card assets (normal and selected states).
CNN_CARD_PATH = ASSET_DIR / "card_cnn.png"
CNN_CARD_SELECTED_PATH = ASSET_DIR / "card_cnn_selected.png"
YOLO_CARD_PATH = ASSET_DIR / "card_yolo.png"
YOLO_CARD_SELECTED_PATH = ASSET_DIR / "card_yolo_selected.png"
IMAGE_SIZE = (224, 224)

# PawScan visual identity: navy, blue, white, and soft lavender.
def bg_uri(path):
    import base64
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{encoded}"

desktop_bg = bg_uri(DESKTOP_BG_PATH)
mobile_bg = bg_uri(MOBILE_BG_PATH)
if desktop_bg:
    bg_css = f"""background-color:#F5F8FF;background-image:linear-gradient(rgba(246,249,255,.48),rgba(246,249,255,.50)),url("{desktop_bg}");background-size:cover;background-position:center top;background-attachment:fixed;background-repeat:no-repeat;"""
else:
    bg_css = "background:linear-gradient(145deg,#F8FAFF,#EAF1FF,#F5F0FF);"
mobile_rule = ""
if mobile_bg:
    mobile_rule = f"""@media(max-width:600px){{.stApp{{background-image:linear-gradient(rgba(246,249,255,.48),rgba(246,249,255,.52)),url("{mobile_bg}")!important;background-size:cover!important;background-position:center top!important;background-attachment:scroll!important;background-repeat:no-repeat!important;}}}}"""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@500;600;700;800&display=swap');
html,body,.stApp,p,button,input,textarea,label,[data-testid="stMarkdownContainer"]{{font-family:'Inter',sans-serif;}}
h1,h2,h3,h4,.brand-name,.hero h1,.panel-title,.member-name,.metric-value,.result-answer{{font-family:'Poppins',sans-serif!important;}}
h1,h2,h3,h4{{font-weight:700;}}
.stApp{{{bg_css}color:#172033;}}
[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"]{{background:transparent!important;}}
#MainMenu,footer{{visibility:hidden;}} header[data-testid="stHeader"]{{background:transparent;}}
.block-container{{max-width:1220px;padding:1.5rem 2rem 2rem;}}
.brand{{display:flex;align-items:center;gap:12px;margin:0 0 18px;}}
.brand-icon{{width:54px;height:54px;flex:0 0 54px;display:flex;align-items:center;justify-content:center;background:transparent;border-radius:0;padding:0;box-shadow:none;overflow:visible;}} .brand-icon img{{width:100%;height:100%;object-fit:contain;border-radius:0;}}
.brand-name{{font-size:22px;font-weight:800;color:#172554;line-height:1.2;}}
.brand-sub{{font-size:11px;color:#7B86A1;letter-spacing:.5px;}}
.progress-wrap{{display:flex;gap:8px;margin:12px 0 24px;}}
.progress-step{{flex:1;border-radius:99px;height:6px;background:#DDE5F7;}}
.progress-step.active{{background:linear-gradient(90deg,#4263EB,#8B5CF6);}}
.step-label{{font-size:11px;color:#64708B;font-weight:800;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;}}
.hero{{position:relative;overflow:hidden;padding:34px;border-radius:24px;color:white;background:linear-gradient(120deg,#172554 0%,#2947C7 58%,#4263EB 100%);box-shadow:0 16px 40px #2947c729;}}
.hero h1{{color:white;font-size:clamp(28px,4vw,46px);line-height:1.15;font-weight:800;letter-spacing:-1px;margin:12px 0;}}
.hero p{{color:#E0E8FF;font-size:14px;line-height:1.85;max-width:700px;}}
.hero-kicker{{display:inline-block;border:1px solid #ffffff45;background:#ffffff15;border-radius:99px;padding:6px 11px;font-size:10px;font-weight:800;letter-spacing:.8px;}}
.hero-decoration{{position:absolute;right:28px;top:20px;font-size:78px;opacity:.18;pointer-events:none;}} .feature-icon{{width:44px;height:44px;object-fit:contain;margin:5px 0 8px;}} .asset-action{{width:20px;height:20px;object-fit:contain;vertical-align:middle;margin-right:7px;}} .result-pet-icon{{width:112px;height:112px;object-fit:contain;display:block;margin:0 auto 8px;filter:drop-shadow(0 8px 12px #2947c71a);}} .section-eyebrow{{font-size:10px;font-weight:800;letter-spacing:1.2px;color:#6478B8;text-transform:uppercase;margin-bottom:7px;}} .soft-note{{border-radius:14px;background:linear-gradient(120deg,#F0F5FF,#FAF7FF);padding:14px 16px;border:1px solid #E2E9FF;color:#596783;font-size:12px;line-height:1.8;}} .stRadio label{{font-weight:700;}} .stProgress>div>div>div>div{{background:linear-gradient(90deg,#4263EB,#8B5CF6);}}
.panel{{background:rgba(255,255,255,.94);border:1px solid #E3E9F5;border-radius:20px;padding:22px;box-shadow:0 8px 26px #18274b0a;}}
.panel-title{{font-size:18px;font-weight:800;color:#172554;margin-bottom:6px;}}
.panel-subtitle{{font-size:12px;color:#758096;line-height:1.75;}}
.metric-card{{background:rgba(255,255,255,.94);border:1px solid #E3E9F5;border-radius:17px;padding:18px;min-height:100px;}}
.member-card{{display:flex;align-items:center;gap:14px;padding:16px 18px;border-radius:18px;background:linear-gradient(115deg,#EEF3FF,#F8F6FF);border:1px solid #DCE5FF;box-shadow:0 8px 24px #2947c710;margin:12px 0 18px;}}
.member-avatar{{width:48px;height:48px;flex:0 0 48px;display:flex;align-items:center;justify-content:center;border-radius:15px;background:linear-gradient(135deg,#4263EB,#8B5CF6);color:white;font-size:24px;box-shadow:0 7px 16px #4263eb35;}}
.member-kicker{{font-size:10px;font-weight:800;letter-spacing:1px;color:#6578B5;text-transform:uppercase;}}
.member-name{{font-size:16px;font-weight:800;color:#172554;margin-top:2px;overflow-wrap:anywhere;}}
.member-campus{{font-size:12px;color:#71809F;margin-top:3px;overflow-wrap:anywhere;}}
.member-check{{margin-left:auto;color:#4965ED;font-size:22px;}}
.analysis-loading{{text-align:center;color:#2947C7;font-weight:800;font-size:15px;padding:4px 0 12px;}}
.metric-label{{color:#758096;font-size:12px;margin-bottom:8px;}}
.metric-value{{color:#172033;font-size:19px;font-weight:800;}}
.metric-hint{{color:#8994A8;font-size:11px;margin-top:6px;}}
.pill{{display:inline-block;border-radius:99px;padding:7px 12px;background:#EAF0FF;color:#2947C7;font-size:11px;font-weight:800;}}
.result-banner{{padding:18px;border-radius:16px;background:#EDF2FF;border:1px solid #D8E2FF;margin:12px 0 18px;}}
.result-label{{color:#5265A8;font-size:11px;font-weight:800;letter-spacing:1px;}}
.result-answer{{color:#203A9B;font-size:29px;font-weight:800;margin-top:4px;}}
.conf-row{{display:flex;align-items:center;gap:10px;margin:13px 0;}}
.conf-name{{width:88px;color:#344054;font-size:12px;font-weight:700;}}
.conf-track{{flex:1;height:8px;border-radius:99px;background:#E7EBF4;overflow:hidden;}}
.conf-fill{{height:100%;background:linear-gradient(90deg,#4263EB,#8B5CF6);border-radius:99px;}}
.conf-pct{{width:55px;text-align:right;color:#344054;font-size:12px;font-weight:800;}}
.empty-state{{text-align:center;padding:26px 15px;border:1px dashed #CFD8EB;border-radius:16px;background:#ffffffc9;color:#667085;line-height:1.8;}}
.footer{{text-align:center;color:#8A65B5;font-size:11px;line-height:1.8;padding:24px 0 4px;}}
.stButton button,.stDownloadButton button{{border-radius:12px;font-weight:800;min-height:44px;border:0;}}
.stButton button[kind="primary"]{{background:linear-gradient(100deg,#4263EB,#8B5CF6);color:white;}}
[data-testid="stFileUploaderDropzone"]{{background:#fff;border:1.5px dashed #B9C9F5;border-radius:15px;padding:14px;}}
[data-testid="stFileUploaderDropzone"]:hover{{border-color:#4263EB;background:#F5F7FF;}}
[data-testid="stImage"] img{{border-radius:15px;}}
div[data-baseweb="input"] input{{border-radius:10px;}}
@media(max-width:900px){{.block-container{{padding:1rem 1rem 2rem;}}.hero{{padding:25px;}}.panel{{padding:17px;}}}}
@media(max-width:600px){{.block-container{{padding:.7rem .7rem 1.5rem;}}.hero{{padding:21px 17px;border-radius:17px;}}.hero h1{{font-size:28px;}}.hero p{{font-size:13px;}}.hero-decoration{{font-size:45px;right:8px;top:8px;}}.panel{{padding:14px;}}.footer{{line-height:1.8;}}}}
{mobile_rule}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classifier(path):
    from tensorflow.keras.models import load_model
    return load_model(path, compile=False)

@st.cache_resource
def load_yolo(path):
    from ultralytics import YOLO
    return YOLO(path)

def image_to_rgb(file):
    return ImageOps.exif_transpose(Image.open(file)).convert("RGB")

def classifier_predict(image, model):
    resized = image.resize(IMAGE_SIZE)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    pred = np.asarray(model.predict(np.expand_dims(arr, axis=0), verbose=0)).reshape(-1)
    if pred.size != 2:
        raise ValueError(f"Model H5 menghasilkan {pred.size} output, seharusnya 2 kelas (anjing dan kucing).")
    # Pastikan urutan ini sama dengan urutan kelas saat training model.
    labels = ["Kucing", "Anjing"]
    order = np.argsort(pred)[::-1]
    return [(labels[int(i)], float(pred[int(i)])) for i in order]

def yolo_predict(image, model, confidence):
    names = model.names
    allowed = {"cats": "Kucing", "cat": "Kucing", "kucing": "Kucing", "dogs": "Anjing", "dog": "Anjing", "anjing": "Anjing"}
    class_ids = [int(idx) for idx, name in names.items() if str(name).strip().lower() in allowed]
    result = model.predict(image, imgsz=64
