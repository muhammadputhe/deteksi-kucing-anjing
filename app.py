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

# Prioritaskan best.pt jika tersedia; pertahankan dukungan model lama.
YOLO_CANDIDATES = [
    ROOT / "model" / "best.pt",
    ROOT / "best.pt",
    ROOT / "model" / "Haikal_Laporan 4.pt",
]
YOLO_PATH = next((p for p in YOLO_CANDIDATES if p.exists()), YOLO_CANDIDATES[0])
YOLO_CONFIDENCE = 0.40
CLASSIFIER_PATH = ROOT / "best_model.h5"
DESKTOP_BG_PATH = ROOT / "background_pawscan.png"
MOBILE_BG_PATH = ROOT / "background_pawscan_mobile.png"
CAT_ICON_PATH = ASSET_DIR / "cat_icon.png"
DOG_ICON_PATH = ASSET_DIR / "dog_icon.png"
MASCOT_PATH = ASSET_DIR / "maskot_pawscan.png"  # Opsional: maskot kucing + anjing dalam satu PNG transparan
STEP_CARD_1_PATH = ASSET_DIR / "kartu_01_registrasi.png"
STEP_CARD_2_PATH = ASSET_DIR / "kartu_02_pilih_metode.png"
STEP_CARD_3_PATH = ASSET_DIR / "kartu_03_unggah_hasil.png"
PAW_PRINT_PATH = ASSET_DIR / "paw_print_blue.png"
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
# Subtle paw-print SVG pattern layered over the background. It is embedded in CSS,
# so no extra image asset is required and it scales cleanly on laptop and mobile.
paw_pattern = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='220' viewBox='0 0 240 220'%3E%3Cg fill='%237C91D8' fill-opacity='.16'%3E%3Cellipse cx='52' cy='75' rx='13' ry='17' transform='rotate(-25 52 75)'/%3E%3Cellipse cx='77' cy='57' rx='12' ry='16' transform='rotate(-8 77 57)'/%3E%3Cellipse cx='101' cy='65' rx='12' ry='16' transform='rotate(15 101 65)'/%3E%3Cellipse cx='120' cy='87' rx='11' ry='15' transform='rotate(30 120 87)'/%3E%3Cpath d='M54 103 C40 111 32 132 43 143 C51 151 64 143 76 138 C88 134 101 145 112 140 C128 133 119 112 105 101 C91 90 69 93 54 103Z'/%3E%3C/g%3E%3C/svg%3E"
if desktop_bg:
    bg_css = f"""background-color:#F5F8FF;background-image:linear-gradient(rgba(246,249,255,.48),rgba(246,249,255,.50)),url("{desktop_bg}");background-size:cover;background-position:center top;background-attachment:fixed;background-repeat:no-repeat;"""
else:
    bg_css = f"""background-color:#F5F8FF;background-image:radial-gradient(ellipse at 8% 8%,rgba(185,205,255,.35),transparent 34%),radial-gradient(ellipse at 94% 76%,rgba(221,204,255,.32),transparent 32%),linear-gradient(145deg,#F8FAFF,#EAF1FF,#F5F0FF),url("{paw_pattern}");background-size:auto,auto,auto,240px 220px;background-position:center,center,center,0 0;background-attachment:scroll;"""
# Add a very subtle paw pattern over an existing custom background too.
if desktop_bg:
    bg_css = bg_css.replace('background-image:', f'background-image:url("{paw_pattern}"),', 1).replace('background-size:cover;', 'background-size:240px 220px,cover,cover;', 1).replace('background-position:center top;', 'background-position:0 0,center top,center top;', 1).replace('background-attachment:fixed;', 'background-attachment:scroll,fixed,fixed;', 1).replace('background-repeat:no-repeat;', 'background-repeat:repeat,no-repeat,no-repeat;', 1)
mobile_rule = f'''@media(max-width:600px){{.stApp{{background-attachment:scroll!important;background-size:auto,auto,auto,190px 175px!important;background-position:center,center,center,0 0!important;}}}}'''
if mobile_bg:
    mobile_rule += f"""@media(max-width:600px){{.stApp{{background-image:url("{paw_pattern}"),linear-gradient(rgba(246,249,255,.48),rgba(246,249,255,.52)),url("{mobile_bg}")!important;background-size:190px 175px,cover,cover!important;background-position:0 0,center,center top!important;background-attachment:scroll!important;background-repeat:repeat,no-repeat,no-repeat!important;}}}}"""

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
.hero-decoration{{position:absolute;right:28px;top:20px;font-size:78px;opacity:.18;pointer-events:none;}}
.home-hero{{position:relative;isolation:isolate;overflow:hidden;min-height:300px;padding:34px;border-radius:24px;color:white;background:linear-gradient(120deg,#172554 0%,#2947C7 58%,#4263EB 100%);box-shadow:0 16px 40px #2947c729;}}
.home-hero-copy{{position:relative;z-index:2;max-width:66%;}}
.home-hero p{{max-width:690px;}}
.home-hero-mascot{{position:absolute;z-index:1;right:18px;bottom:-12px;width:32%;height:95%;object-fit:contain;object-position:center bottom;filter:drop-shadow(0 10px 10px #101c4b25);pointer-events:none;}}
.step-card{{height:100%;min-height:145px;padding:20px;border:1px solid #E3E9F5;border-radius:20px;background:rgba(255,255,255,.96);box-shadow:0 8px 26px #18274b0a;transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease;}}
div[data-testid='stImage'] img{{border-radius:20px;box-shadow:0 8px 26px #18274b0a;transition:transform .22s ease,box-shadow .22s ease;}}@media(hover:hover){{div[data-testid='stImage'] img:hover{{transform:translateY(-3px);box-shadow:0 15px 30px #2947c71a;}}}}.step-card-image{{display:block;width:100%;height:auto;border-radius:20px;box-shadow:0 8px 26px #18274b0a;transition:transform .22s ease,box-shadow .22s ease;}}
.step-card-image:hover{{transform:translateY(-3px);box-shadow:0 15px 30px #2947c71a;}}
.paw-decor{{position:fixed;z-index:0;width:54px;height:54px;object-fit:contain;opacity:.12;pointer-events:none;filter:saturate(1.05);}}
.paw-decor.paw-one{{left:12px;top:34%;transform:rotate(-18deg);}}
.paw-decor.paw-two{{right:18px;top:15%;width:42px;height:42px;transform:rotate(16deg);}}
.paw-decor.paw-three{{right:8px;bottom:10%;width:62px;height:62px;transform:rotate(-12deg);}}

.step-card:hover{{transform:translateY(-5px);box-shadow:0 15px 30px #2947c71a;border-color:#C8D5FF;}}
.step-icon{{width:42px;height:42px;display:flex;align-items:center;justify-content:center;border-radius:14px;background:linear-gradient(135deg,#EAF0FF,#F3EDFF);font-size:22px;margin:0 0 12px;}}
.step-card .pill{{display:inline-block;}}
@keyframes pawscanArrow{{0%,100%{{transform:translateX(0)}}50%{{transform:translateX(4px)}}}}
@media(hover:hover){{.stButton>button[kind="primary"]:hover{{transform:translateY(-2px);box-shadow:0 12px 25px #1991ff45;filter:saturate(1.08);}}.stButton>button[kind="primary"]:hover span{{animation:pawscanArrow .65s ease-in-out infinite;}}}}
.stButton>button[kind="primary"]{{transition:transform .2s ease,box-shadow .2s ease,filter .2s ease;border-radius:999px!important;box-shadow:0 7px 20px #1991ff2b;}}
@media(max-width:900px){{.home-hero{{min-height:285px;padding:28px;}}.home-hero-copy{{max-width:68%;}}.home-hero-mascot{{width:34%;right:8px;}}.step-card{{padding:17px;}}}}
@media(max-width:600px){{.home-hero{{min-height:285px;padding:22px 18px;border-radius:20px;}}.home-hero-copy{{max-width:72%;}}.home-hero h1{{font-size:clamp(24px,6vw,32px);}}.home-hero p{{font-size:11px;line-height:1.6;max-width:100%;}}.home-hero-mascot{{width:42%;height:64%;right:-4px;bottom:-4px;opacity:.98;}}.step-card{{min-height:0;padding:16px;}}}}
@media(prefers-reduced-motion:reduce){{.step-card,.stButton>button[kind="primary"]{{transition:none!important;}}.step-card:hover,.stButton>button[kind="primary"]:hover{{transform:none!important;}}}}
.welcome-hero{{position:relative;isolation:isolate;min-height:232px;padding:23px 18px 20px;border-radius:17px;overflow:visible;background:linear-gradient(112deg,#172554 0%,#2446C7 55%,#3659E8 100%);box-shadow:0 12px 28px #2947c729;margin-bottom:12px;}}
.welcome-hero .welcome-copy{{position:relative;z-index:3;width:52%;max-width:245px;}}
.welcome-hero .hero-kicker{{font-size:8px;padding:4px 8px;letter-spacing:.4px;color:#fff;border-color:#ffffff60;background:#ffffff16;}}
.welcome-hero h1{{color:#fff!important;font-size:clamp(23px,2.25vw,30px);line-height:1.08;letter-spacing:-.55px;font-weight:800;margin:13px 0 10px;max-width:225px;text-shadow:0 1px 1px #101c4b12;}}
.welcome-hero p{{font-size:10px;line-height:1.5;max-width:215px;margin:0;color:#F0F4FF;}}
.welcome-mascot-combined{{position:absolute;z-index:2;right:-20px;bottom:-15px;width:60%;height:108%;object-fit:contain;object-position:center bottom;pointer-events:none;filter:drop-shadow(0 5px 5px #101c4b20);}}
.welcome-mascot-pair{{position:absolute;z-index:2;right:-15px;bottom:-13px;width:53%;height:103%;display:flex;align-items:flex-end;justify-content:center;gap:0;pointer-events:none;}}
.welcome-mascot-pair img{{width:55%;height:100%;object-fit:contain;object-position:center bottom;filter:drop-shadow(0 3px 3px #101c4b22);}}
.welcome-hero .hero-decoration{{display:none;}}
@media(max-width:900px){{.welcome-hero{{min-height:232px;padding:20px 16px;}}.welcome-hero .welcome-copy{{width:52%;max-width:235px;}}.welcome-hero h1{{font-size:24px;max-width:220px;}}.welcome-hero p{{max-width:205px;font-size:9.5px;}}.welcome-mascot-combined{{width:60%;height:105%;right:-15px;bottom:-12px;}}.welcome-mascot-pair{{width:53%;height:100%;right:-10px;bottom:-10px;}}}}
@media(max-width:600px){{.welcome-hero{{min-height:225px;padding:17px 13px;border-radius:14px;margin-bottom:18px;}}.welcome-hero .welcome-copy{{width:53%;max-width:210px;}}.welcome-hero h1{{font-size:21px;line-height:1.08;margin:11px 0 8px;max-width:195px;}}.welcome-hero p{{font-size:8.5px;line-height:1.45;max-width:180px;}}.welcome-mascot-combined{{width:59%;height:94%;right:-12px;bottom:-10px;}}.welcome-mascot-pair{{width:52%;height:88%;right:-8px;bottom:-8px;}}}} .feature-icon{{width:44px;height:44px;object-fit:contain;margin:5px 0 8px;}} .asset-action{{width:20px;height:20px;object-fit:contain;vertical-align:middle;margin-right:7px;}} .result-pet-icon{{width:112px;height:112px;object-fit:contain;display:block;margin:0 auto 8px;filter:drop-shadow(0 8px 12px #2947c71a);}} .section-eyebrow{{font-size:10px;font-weight:800;letter-spacing:1.2px;color:#6478B8;text-transform:uppercase;margin-bottom:7px;}} .soft-note{{border-radius:14px;background:linear-gradient(120deg,#F0F5FF,#FAF7FF);padding:14px 16px;border:1px solid #E2E9FF;color:#596783;font-size:12px;line-height:1.8;}} .stRadio label{{font-weight:700;}} .stProgress>div>div>div>div{{background:linear-gradient(90deg,#4263EB,#8B5CF6);}}
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
.footer{{text-align:center;color:#2D4FD7;font-size:11px;line-height:1.8;padding:24px 0 4px;}}
/* Registration refresh: clearer hierarchy, calmer surfaces, and responsive layout. */
.registration-heading{{display:flex;align-items:flex-start;gap:13px;padding:22px 23px;margin-bottom:18px;border:1px solid #E1E8F8;border-radius:21px;background:rgba(255,255,255,.91);box-shadow:0 10px 28px #233b7609;}}
.registration-heading-icon{{display:flex;align-items:center;justify-content:center;flex:0 0 43px;width:43px;height:43px;border-radius:14px;background:linear-gradient(135deg,#EAF1FF,#F1EBFF);font-size:21px;}}
.registration-heading-title{{font-family:'Poppins',sans-serif;font-size:21px;font-weight:800;line-height:1.3;color:#172554;margin:0 0 5px;}}
.registration-heading-copy{{font-size:12px;line-height:1.75;color:#71809F;margin:0;}}
.registration-form-card{{padding:22px 23px 20px;border:1px solid #E2E9F7;border-radius:21px;background:rgba(255,255,255,.91);box-shadow:0 12px 32px #243b760a;}}
.registration-form-card [data-testid="stTextInput"]{{margin-top:2px;margin-bottom:13px;}}
.registration-form-card [data-testid="stTextInput"] label{{font-size:12px;font-weight:700;color:#34446A;margin-bottom:7px;}}
.registration-form-card div[data-baseweb="input"]{{background:#F8FAFF;border:1px solid #E1E8F6;border-radius:12px;transition:border-color .18s ease,box-shadow .18s ease;}}
.registration-form-card div[data-baseweb="input"]:focus-within{{border-color:#7188F7;box-shadow:0 0 0 3px #4263EB16;}}
.registration-form-card div[data-baseweb="input"] input{{background:transparent!important;min-height:43px;color:#172033;}}
.registration-helper{{display:flex;align-items:flex-start;gap:9px;padding:11px 12px;margin-top:3px;border-radius:12px;background:linear-gradient(110deg,#F1F6FF,#FAF7FF);border:1px solid #E5EBFC;color:#6A7895;font-size:11px;line-height:1.7;}}
.registration-helper-icon{{font-size:15px;line-height:1.4;}}
.registration-benefit{{display:flex;gap:8px;align-items:center;margin-top:13px;color:#72809B;font-size:10px;line-height:1.6;}}
.registration-benefit span{{display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;border-radius:7px;background:#E8F8F0;color:#159565;font-size:12px;flex:0 0 20px;}}
.welcome-hero{{box-shadow:0 18px 42px #2947c72b;}}
.welcome-hero:before{{content:"";position:absolute;width:190px;height:190px;right:-60px;top:-90px;border:1px solid #ffffff18;border-radius:50%;pointer-events:none;}}
@media(max-width:900px){{.registration-heading{{padding:19px;}}.registration-form-card{{padding:19px;}}}}
@media(max-width:600px){{.registration-heading{{gap:10px;padding:16px;border-radius:17px;margin-bottom:14px;}}.registration-heading-icon{{width:37px;height:37px;flex-basis:37px;font-size:18px;border-radius:12px;}}.registration-heading-title{{font-size:18px;}}.registration-heading-copy{{font-size:11px;}}.registration-form-card{{padding:17px 15px;border-radius:17px;}}.registration-helper{{font-size:10px;}}}}
.stButton button,.stDownloadButton button{{border-radius:12px;font-weight:800;min-height:44px;border:0;}}
.analysis-heading{{display:flex;align-items:center;gap:14px;padding:22px 24px;border-radius:22px;background:linear-gradient(115deg,#172554 0%,#2947C7 58%,#536DFF 100%);box-shadow:0 14px 32px #2947c726;color:#fff;margin-bottom:16px;}}
.analysis-heading .analysis-heading-icon{{width:64px;height:64px;object-fit:contain;flex:0 0 64px;filter:drop-shadow(0 5px 10px #091a5a55);}}
.analysis-heading h1{{font-size:clamp(23px,3vw,32px);line-height:1.15;color:#fff;margin:0 0 6px;font-weight:800;}}
.analysis-heading p{{color:#E5EBFF;margin:0;font-size:13px;line-height:1.65;}}
.analysis-status{{display:flex;align-items:center;gap:14px;padding:14px 17px;border-radius:17px;margin:0 0 16px;border:1px solid #DCE6FF;background:linear-gradient(115deg,#F4F8FF,#FFFFFF);}}
.analysis-status.ready{{border-color:#A7F3D0;background:linear-gradient(115deg,#F0FDF9,#FFFFFF);}}
.analysis-status.waiting{{border-color:#BFDBFE;background:linear-gradient(115deg,#EFF6FF,#FFFFFF);}}
.analysis-status .status-icon{{width:66px;height:66px;object-fit:contain;flex:0 0 66px;}}
.analysis-status .status-title{{font-family:'Poppins',sans-serif;font-weight:800;font-size:16px;color:#1E3A8A;margin-bottom:3px;}}
.analysis-status.ready .status-title{{color:#047857;}}
.analysis-status .status-copy{{font-size:12px;line-height:1.65;color:#667594;}}
.analysis-upload-panel{{background:rgba(255,255,255,.92);border:1px solid #E1E9FA;border-radius:22px;padding:18px;box-shadow:0 10px 30px #18274b0a;}}
.analysis-tip{{background:linear-gradient(135deg,#F5F8FF,#FBF9FF);border:1px solid #E1E8FB;border-radius:18px;padding:18px 20px;}}
.analysis-tip-title{{font-family:'Poppins',sans-serif;color:#1E327A;font-weight:800;font-size:15px;margin-bottom:8px;}}
.analysis-tip-copy{{font-size:12px;line-height:1.9;color:#667594;}}
.stButton button[kind="primary"]{{background:linear-gradient(100deg,#24B8FF 0%,#0789F9 100%)!important;color:white!important;border:1px solid #ffffff70!important;border-radius:999px!important;min-height:58px!important;font-size:16px!important;font-weight:800!important;box-shadow:0 8px 18px #129af03d, inset 0 1px 0 #ffffff65!important;transition:background .2s ease,transform .2s ease,box-shadow .2s ease!important;position:relative!important;}}
.stButton button[kind="primary"]:hover{{background:linear-gradient(100deg,#168BFF 0%,#4263EB 55%,#8458F6 100%)!important;transform:translateY(-1px);box-shadow:0 10px 24px #6954f04a,0 0 0 2px #ffffffaa!important;}}
.stButton button[kind="primary"]:disabled{{background:linear-gradient(100deg,#D9E7FA,#C8D5EC)!important;color:#FFFFFF!important;border-color:#ffffff55!important;box-shadow:none!important;transform:none!important;opacity:1!important;}}
.st-key-start_analysis_button button{{display:flex!important;align-items:center!important;justify-content:center!important;gap:0!important;padding:0 54px!important;position:relative!important;}}
.st-key-start_analysis_button button::before{{content:"✦";position:absolute;left:clamp(22px,5vw,42px);font-size:29px;line-height:1;color:#fff;text-shadow:0 1px 8px #ffffff75;}}
.st-key-start_analysis_button button::after{{content:"→";position:absolute;right:clamp(22px,5vw,42px);font-size:28px;font-weight:500;line-height:1;color:#fff;}}
.st-key-start_analysis_button button:disabled::before,.st-key-start_analysis_button button:disabled::after{{color:#fff;}}
@media(max-width:600px){{.analysis-heading{{padding:18px 16px;gap:10px;}}.analysis-heading .analysis-heading-icon{{width:48px;height:48px;flex-basis:48px;}}.analysis-status{{padding:12px;gap:10px;}}.analysis-status .status-icon{{width:52px;height:52px;flex-basis:52px;}}}}

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
    """Deteksi tanpa filter kelas awal agar prediksi model tidak tersembunyi."""
    result = model.predict(
        source=np.asarray(image.convert("RGB")),
        imgsz=640,
        conf=confidence,
        iou=0.45,              # Ambang IoU untuk Non-Maximum Suppression (NMS)
        agnostic_nms=False,   # NMS per kelas: kucing dan anjing tetap dibedakan
        max_det=100,           # Batas maksimum deteksi per gambar
        verbose=False,
    )[0]

    # result.plot() dari Ultralytics menghasilkan array BGR; Streamlit perlu RGB.
    annotated = result.plot(line_width=3)[:, :, ::-1]

    allowed = {
        "cat": "Kucing",
        "cats": "Kucing",
        "kucing": "Kucing",
        "dog": "Anjing",
        "dogs": "Anjing",
        "anjing": "Anjing",
    }
    counts, best, raw_detections = {}, {}, []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        raw_label = str(result.names[class_id]).strip().lower()
        normalized = raw_label.replace("_", " ").replace("-", " ")
        tokens = normalized.split()

        label = allowed.get(normalized)
        if label is None:
            if "cat" in tokens or "kucing" in tokens:
                label = "Kucing"
            elif "dog" in tokens or "anjing" in tokens:
                label = "Anjing"

        raw_detections.append({
            "ID kelas": class_id,
            "Label model": raw_label,
            "Confidence": round(score, 4),
        })

        if label is not None:
            counts[label] = counts.get(label, 0) + 1
            best[label] = max(best.get(label, 0.0), score)

    return annotated, counts, best, raw_detections, dict(result.names)

def confidence_rows(items):
    rows = []
    for name, score in items:
        pct = max(0.0, min(100.0, float(score) * 100))
        rows.append(f'<div class="conf-row"><span class="conf-name">{name}</span><span class="conf-track"><span class="conf-fill" style="width:{pct:.1f}%;display:block"></span></span><span class="conf-pct">{pct:.1f}%</span></div>')
    return "".join(rows)

def show_member_card(compact=False):
    """Render the member identity using the current registration values."""
    name = escape(st.session_state.get("full_name", "PawScan Member") or "PawScan Member")
    campus = escape(st.session_state.get("campus", "") or "Kampus belum diisi")
    main_uri = bg_uri(MAIN_ICON_PATH)
    avatar = f'<img src="{main_uri}" alt="PawScan" style="width:100%;height:100%;object-fit:contain;padding:5px;">' if main_uri else '🐾'
    st.markdown(
        f'<div class="member-card"><div class="member-avatar">{avatar}</div>'
        f'<div><div class="member-kicker">PawScan Member</div>'
        f'<div class="member-name">{name}</div>'
        f'<div class="member-campus">{campus}</div></div>'
        f'<div class="member-check">✓</div></div>',
        unsafe_allow_html=True,
    )

def show_empty_illustration():
    if EMPTY_STATE_PATH.exists():
        st.image(str(EMPTY_STATE_PATH), use_container_width=True, caption=None)
    else:
        st.markdown('<div class="empty-state"><div style="font-size:36px">📸</div><b>Gambar belum dipilih</b><br>Unggah foto untuk memulai analisis.</div>', unsafe_allow_html=True)

def show_loading_visual(message):
    if LOADING_GIF_PATH.exists():
        st.image(str(LOADING_GIF_PATH), use_container_width=True)
        st.markdown(f'<div class="analysis-loading">{escape(message)}</div>', unsafe_allow_html=True)
    else:
        st.info(message)

def footer():
    st.markdown('<div class="footer">created by <b>Muhammad Haikal Asy’ary · 2408108010065</b><br>Project UTS Praktikum Pemrograman Big data | Universitas Syiah Kuala</div>', unsafe_allow_html=True)

def go(step):
    st.session_state.step = step
    st.rerun()

if "step" not in st.session_state:
    st.session_state.step = 1
if "full_name" not in st.session_state:
    st.session_state.full_name = ""
if "campus" not in st.session_state:
    st.session_state.campus = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "Klasifikasi CNN"

# Brand + five-stage progress navigation.
main_icon_uri = bg_uri(MAIN_ICON_PATH)
brand_mark = f'<img src="{main_icon_uri}" alt="PawScan icon">' if main_icon_uri else '🐾'
st.markdown(f'<div class="brand"><div class="brand-icon">{brand_mark}</div><div><div class="brand-name">PawScan</div><div class="brand-sub">ANIMAL RECOGNITION · CNN + YOLO</div></div></div>', unsafe_allow_html=True)
step = st.session_state.step
labels = ["Beranda", "Registrasi", "Metode", "Unggah & Analisis", "Hasil"]
progress = "".join(f'<div class="progress-step {"active" if i <= step else ""}"></div>' for i in range(1, 6))
st.markdown(f'<div class="progress-wrap">{progress}</div><div class="step-label">Sesi {step} dari 5 · {labels[step-1]}</div>', unsafe_allow_html=True)

if step == 1:
    home_mascot_uri = bg_uri(MASCOT_PATH)
    home_mascot = f'<img class="home-hero-mascot" src="{home_mascot_uri}" alt="Maskot kucing dan anjing PawScan">' if home_mascot_uri else ''
    st.markdown(f"""
    <div class="home-hero">
      <div class="home-hero-copy">
        <span class="hero-kicker">SMART PET RECOGNITION</span>
        <h1>Kenali Anjing & Kucing<br>dengan <span style="color:#AFC2FF">Kekuatan AI</span></h1>
        <p>Selamat datang di PawScan! Pilih metode klasifikasi CNN untuk mengenali jenis hewan, atau deteksi objek menggunakan YOLO untuk menemukan dan menandai hewan dalam gambar. Ikuti lima sesi sederhana untuk melihat hasilnya.</p>
      </div>
      {home_mascot}
    </div>""", unsafe_allow_html=True)
    st.write("")
    # Kartu langkah menggunakan gambar PNG siap pakai; tidak perlu membangun ulang desain dengan HTML/CSS.
    paw_uri = bg_uri(PAW_PRINT_PATH)
    if paw_uri:
        st.markdown(
            f'<img class="paw-decor paw-one" src="{paw_uri}" alt="" aria-hidden="true">'
            f'<img class="paw-decor paw-two" src="{paw_uri}" alt="" aria-hidden="true">'
            f'<img class="paw-decor paw-three" src="{paw_uri}" alt="" aria-hidden="true">',
            unsafe_allow_html=True,
        )
    card_paths = [STEP_CARD_1_PATH, STEP_CARD_2_PATH, STEP_CARD_3_PATH]
    c1, c2, c3 = st.columns(3, gap="medium")
    for col, card_path in zip((c1, c2, c3), card_paths):
        with col:
            if card_path.exists():
                st.image(str(card_path), use_container_width=True)
            else:
                st.warning(f"Aset kartu tidak ditemukan: {card_path.name}")
    st.write("")
    if st.button("Mulai Sekarang →", type="primary", use_container_width=True):
        go(2)

elif step == 2:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        mascot_uri = bg_uri(MASCOT_PATH)
        cat_uri = bg_uri(CAT_ICON_PATH)
        dog_uri = bg_uri(DOG_ICON_PATH)
        if mascot_uri:
            mascot_markup = f'<img class="welcome-mascot-combined" src="{mascot_uri}" alt="Maskot kucing dan anjing PawScan">'
        else:
            mascot_items = ""
            if cat_uri:
                mascot_items += f'<img src="{cat_uri}" alt="Maskot kucing PawScan">'
            if dog_uri:
                mascot_items += f'<img src="{dog_uri}" alt="Maskot anjing PawScan">'
            mascot_markup = f'<div class="welcome-mascot-pair">{mascot_items}</div>' if mascot_items else ""
        st.markdown(
            '<div class="welcome-hero"><div class="welcome-copy">'
            '<span class="hero-kicker">PAWSCAN MEMBER</span>'
            '<h1>Halo, Sobat Paw!</h1>'
            '<p>Kenali sahabat berbulu dengan teknologi AI. Lengkapi profil singkatmu untuk melanjutkan ke tahap pemilihan metode.</p>'
            '</div>' + mascot_markup + '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="registration-benefit"><span>✓</span><div><b>Proses sederhana</b><br>Isi dua informasi, lalu lanjutkan ke pemilihan metode analisis.</div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="registration-heading">'
            '<div class="registration-heading-icon">✍️</div>'
            '<div><div class="registration-heading-title">Registrasi Singkat</div>'
            '<p class="registration-heading-copy">Kenalan dulu, lalu pilih cara PawScan mengenali hewan pada gambarmu.</p></div></div>'
            '<div class="registration-form-card">',
            unsafe_allow_html=True,
        )
        full_name = st.text_input(
            "Nama lengkap",
            value=st.session_state.full_name,
            placeholder="Masukkan nama lengkap",
            help="Masukkan nama yang ingin ditampilkan pada ringkasan hasil.",
            key="registration_full_name",
        )
        campus = st.text_input(
            "Nama kampus",
            value=st.session_state.campus,
            placeholder="Masukkan nama universitas atau kampus",
            help="Masukkan nama universitas atau institusi kamu.",
            key="registration_campus",
        )
        st.markdown(
            '<div class="registration-helper"><span class="registration-helper-icon">💡</span>'
            '<div><b>Hampir selesai!</b><br>Pastikan kedua kolom sudah benar. Setelah ini, kamu bisa memilih Klasifikasi CNN atau Deteksi Objek YOLO.</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        a, b = st.columns(2, gap="small")
        with a:
            if st.button("← Kembali", use_container_width=True, key="registration_back"):
                go(1)
        with b:
            if st.button("Lanjutkan →", type="primary", use_container_width=True, key="registration_continue"):
                if not full_name.strip() or not campus.strip():
                    st.warning("Nama lengkap dan nama kampus perlu diisi terlebih dahulu.")
                else:
                    st.session_state.full_name = full_name.strip()
                    st.session_state.campus = campus.strip()
                    go(3)

elif step == 3:
    st.markdown("""
    <div class="hero">
      <span class="hero-kicker">SESI 3 · PILIH METODE</span>
      <h1>Bagaimana kamu ingin<br><span style="color:#AFC2FF">mengenali hewan?</span></h1>
      <p>Pilih kartu yang sesuai dengan kebutuhanmu. CNN berfokus mengenali jenis hewan, sedangkan YOLO menemukan lokasi dan jumlah objek. Metode pilihan akan digunakan otomatis pada sesi unggah gambar.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    selected_mode = st.session_state.analysis_mode

    # Embed the selected PNG directly as each button's background so the whole
    # card is clickable and the artwork changes with the selected method.
    cnn_normal_uri = bg_uri(CNN_CARD_PATH)
    cnn_selected_uri = bg_uri(CNN_CARD_SELECTED_PATH)
    yolo_normal_uri = bg_uri(YOLO_CARD_PATH)
    yolo_selected_uri = bg_uri(YOLO_CARD_SELECTED_PATH)

    cnn_card_uri = cnn_selected_uri if selected_mode == "Klasifikasi CNN" else cnn_normal_uri
    yolo_card_uri = yolo_selected_uri if selected_mode == "Deteksi Objek YOLO" else yolo_normal_uri

    # If the asset folder hasn't been copied yet, keep the screen usable with
    # a styled native-button fallback and a clear warning.
    if not all([cnn_normal_uri, cnn_selected_uri, yolo_normal_uri, yolo_selected_uri]):
        st.warning("Sebagian asset kartu belum ditemukan. Ekstrak pawscan_method_cards_idea4.zip dan letakkan empat file PNG di folder assets/.")

    st.markdown(f"""
    <style>
      div[class*="st-key-choose_cnn_card"] button,
      div[class*="st-key-choose_yolo_card"] button {{
        width: 100% !important;
        min-width: 0 !important;
        min-height: 0 !important;
        height: auto !important;
        aspect-ratio: 1.69 / 1 !important;
        padding: 0 !important;
        border: 0 !important;
        border-radius: 22px !important;
        overflow: hidden !important;
        background-color: transparent !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: 100% 100% !important;
        box-shadow: 0 10px 28px #18274b12 !important;
        transition: transform .18s ease, box-shadow .18s ease !important;
      }}
      div[class*="st-key-choose_cnn_card"] button {{
        background-image: url("{cnn_card_uri}") !important;
      }}
      div[class*="st-key-choose_yolo_card"] button {{
        background-image: url("{yolo_card_uri}") !important;
      }}
      div[class*="st-key-choose_cnn_card"] button:hover,
      div[class*="st-key-choose_yolo_card"] button:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 16px 34px #2947c72b !important;
        filter: brightness(1.025);
      }}
      div[class*="st-key-choose_cnn_card"] button > div,
      div[class*="st-key-choose_yolo_card"] button > div {{
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
      }}
      @media(max-width: 700px) {{
        div[class*="st-key-choose_cnn_card"] button,
        div[class*="st-key-choose_yolo_card"] button {{
          aspect-ratio: 1.45 / 1 !important;
          border-radius: 16px !important;
        }}
      }}
    </style>
    """, unsafe_allow_html=True)

    cnn_col, yolo_col = st.columns(2, gap="large")
    with cnn_col:
        if st.button("Pilih metode Klasifikasi CNN", key="choose_cnn_card", use_container_width=True):
            if st.session_state.analysis_mode != "Klasifikasi CNN":
                st.session_state.analysis_mode = "Klasifikasi CNN"
                st.rerun()
    with yolo_col:
        if st.button("Pilih metode Deteksi Objek YOLO", key="choose_yolo_card", use_container_width=True):
            if st.session_state.analysis_mode != "Deteksi Objek YOLO":
                st.session_state.analysis_mode = "Deteksi Objek YOLO"
                st.rerun()

    st.write("")
    selected_description = (
        "Klasifikasi CNN memprediksi kelas utama kucing atau anjing dan menampilkan skor untuk setiap kelas."
        if st.session_state.analysis_mode == "Klasifikasi CNN"
        else "Deteksi Objek YOLO menandai posisi hewan dengan kotak deteksi dan menghitung objek yang dikenali."
    )
    st.markdown(
        f'<div class="soft-note"><b>Metode terpilih: {escape(st.session_state.analysis_mode)}</b><br>{escape(selected_description)}<br><br>Tekan kartu lainnya kapan saja untuk mengganti pilihan.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    back_col, next_col = st.columns(2, gap="medium")
    with back_col:
        if st.button("← Kembali ke Registrasi", use_container_width=True, key="method_back"):
            go(2)
    with next_col:
        if st.button("Lanjut ke Unggah Gambar →", type="primary", use_container_width=True, key="method_continue"):
            go(4)

elif step == 4:
    mode = st.session_state.analysis_mode
    brain_uri = bg_uri(ASSET_DIR / "icon_ai_brain.png")
    status_uri = bg_uri(ASSET_DIR / ("icon_image_ready.png" if st.session_state.get("pawscan_image_ready", False) else "icon_upload.png"))
    st.markdown(
        f'<div class="analysis-heading"><img class="analysis-heading-icon" src="{brain_uri}" alt="AI PawScan">'
        f'<div><div class="hero-kicker">SESI 4 · SMART ANALYSIS</div><h1>Unggah & Analisis</h1>'
        f'<p>Metode aktif: <b>{escape(mode)}</b>. Unggah foto hewan untuk memulai analisis AI.</p></div></div>',
        unsafe_allow_html=True,
    )
    show_member_card(compact=True)
    st.markdown(
        f'<div class="analysis-status waiting"><img class="status-icon" src="{status_uri}" alt="Status unggah">'
        f'<div><div class="status-title">Siapkan foto hewanmu</div>'
        f'<div class="status-copy">Gunakan foto kucing atau anjing yang jelas. Format JPG, JPEG, dan PNG didukung.</div></div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        st.markdown('<div class="analysis-upload-panel"><div class="panel-title">📷 Area gambar</div><div class="panel-subtitle">Pilih gambar yang ingin diperiksa PawScan.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Pilih gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="pawscan_upload", label_visibility="collapsed")
        image = None
        if uploaded is not None:
            try:
                image = image_to_rgb(uploaded)
                st.image(image, caption="Pratinjau gambar", use_container_width=True)
            except Exception:
                st.error("Gambar tidak dapat dibaca. Silakan pilih file gambar lain.")
        else:
            show_empty_illustration()
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        mode_label = "Klasifikasi CNN" if mode == "Klasifikasi CNN" else "Deteksi Objek YOLO"
        mode_detail = "Memprediksi kelas kucing atau anjing dan menampilkan skor model." if mode == "Klasifikasi CNN" else "Menandai lokasi hewan, menampilkan confidence, dan menghitung objek."
        st.markdown(f'<div class="analysis-tip"><div class="analysis-tip-title">✨ Metode aktif · {escape(mode_label)}</div><div class="analysis-tip-copy">{escape(mode_detail)}<br><br>✓ Pastikan hewan terlihat jelas<br>✓ Gunakan pencahayaan yang cukup<br>✓ Hindari gambar yang terlalu buram</div></div>', unsafe_allow_html=True)
        yolo_conf = YOLO_CONFIDENCE
        st.write("")
        if st.button("Mulai Analisis", type="primary", use_container_width=True, disabled=(image is None), key="start_analysis_button"):
            try:
                if mode == "Klasifikasi CNN":
                    if not CLASSIFIER_PATH.exists():
                        st.error(f"Model CNN tidak ditemukan: {CLASSIFIER_PATH.name}. Letakkan file best_model.h5 di folder yang sama dengan app.py.")
                    else:
                        loading_slot = st.empty()
                        with loading_slot.container():
                            show_loading_visual("AI sedang membaca gambar kamu. Mohon tunggu...")
                        with st.spinner("CNN sedang menganalisis gambar..."):
                            model = load_classifier(str(CLASSIFIER_PATH))
                            predictions = classifier_predict(image, model)
                        loading_slot.empty()
                        st.session_state.analysis_result = {"mode": mode, "predictions": predictions, "image": image.copy()}
                        go(5)
                else:
                    if not YOLO_PATH.exists():
                        st.error(f"Model YOLO tidak ditemukan. Letakkan best.pt di folder proyek atau model/, atau model lama di model/. Jalur yang dicoba: {YOLO_PATH}")
                    else:
                        loading_slot = st.empty()
                        with loading_slot.container():
                            show_loading_visual("PawScan sedang memindai objek dengan YOLO. Mohon tunggu...")
                        with st.spinner("YOLO sedang mendeteksi objek..."):
                            model = load_yolo(str(YOLO_PATH))
                            annotated, counts, best, raw_detections, model_names = yolo_predict(image, model, yolo_conf)
                        loading_slot.empty()
                        st.session_state.analysis_result = {"mode": mode, "annotated": annotated, "counts": counts, "best": best, "raw_detections": raw_detections, "model_names": model_names, "yolo_confidence": yolo_conf, "image": image.copy()}
                        go(5)
            except ImportError as exc:
                st.error("Dependensi model belum terpasang. CNN memerlukan TensorFlow; YOLO memerlukan ultralytics.")
                with st.expander("Detail teknis"):
                    st.code(str(exc))
            except Exception as exc:
                st.error("Analisis gagal dijalankan. Periksa file model dan dependensi.")
                with st.expander("Detail kesalahan"):
                    st.code(str(exc))
    st.write("")
    if st.button("← Kembali ke Pilihan Metode"):
        go(3)

elif step == 5:
    result = st.session_state.analysis_result
    if not result:
        st.markdown('<div class="empty-state"><div style="font-size:36px">🔎</div><b>Belum ada hasil analisis</b><br>Kembali ke sesi analisis untuk mengunggah gambar.</div>', unsafe_allow_html=True)
        if st.button("Ke sesi analisis", type="primary"):
            go(4)
    else:
        safe_name = escape(st.session_state.full_name or "Teman PawScan")
        st.markdown(f'<div class="hero"><div class="hero-decoration">✨</div><span class="hero-kicker">ANALISIS SELESAI</span><h1>Hasil PawScan</h1><p>Terima kasih, {safe_name}! Berikut ringkasan hasil analisis gambar Anda.</p></div>', unsafe_allow_html=True)
        st.write("")
        show_member_card()
        st.caption(f"Metode analisis: {result['mode']}")
        st.write("")
        left, right = st.columns([1, 1], gap="large")
        with left:
            if result["mode"] == "Klasifikasi CNN":
                st.image(result["image"], caption="Gambar yang dianalisis", use_container_width=True)
            else:
                # Jika tidak ada kelas hewan yang dikenali, tampilkan foto asli tanpa kotak yang membingungkan.
                has_detection = bool(result.get("best"))
                display_image = result["annotated"] if has_detection else result["image"]
                image_caption = "Gambar dengan kotak deteksi YOLO" if has_detection else "Gambar asli yang dianalisis"
                st.image(display_image, caption=image_caption, use_container_width=True)
                out = io.BytesIO()
                # `result["image"]` bisa berupa objek PIL, sedangkan `result["annotated"]`
                # biasanya berupa array NumPy. Tangani kedua tipe agar unduhan tidak error.
                if isinstance(display_image, Image.Image):
                    display_image.save(out, format="PNG")
                else:
                    Image.fromarray(np.asarray(display_image)).save(out, format="PNG")
                if DOWNLOAD_ICON_PATH.exists():
                    st.image(str(DOWNLOAD_ICON_PATH), width=34)
                download_label = "Unduh gambar hasil YOLO" if has_detection else "Unduh gambar asli"
                st.download_button(download_label, data=out.getvalue(), file_name="hasil_pawscan.png", mime="image/png", use_container_width=True)
        with right:
            if result["mode"] == "Klasifikasi CNN":
                predictions = result["predictions"]
                top_label, top_score = predictions[0]
                icon_path = CAT_ICON_PATH if top_label == "Kucing" else DOG_ICON_PATH
                if icon_path.exists():
                    st.image(str(icon_path), width=140)
                else:
                    st.markdown("<div style='font-size:44px'>" + ("🐱" if top_label == "Kucing" else "🐶") + "</div>", unsafe_allow_html=True)
                st.markdown(f'<div class="result-banner"><div class="result-label">PREDIKSI UTAMA</div><div class="result-answer">{top_label}</div><div style="color:#5265A8;font-size:12px;margin-top:5px">Skor model: {top_score:.1%}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">Perbandingan skor kelas</div>', unsafe_allow_html=True)
                st.markdown(confidence_rows(predictions), unsafe_allow_html=True)
                summary = f"Model CNN memprediksi gambar sebagai {top_label} dengan skor model {top_score:.1%}. "
                summary += "Skor tertinggi menunjukkan kelas yang paling didukung model, bukan jaminan bahwa prediksi selalu benar."
            else:
                counts, best = result["counts"], result["best"]
                total = sum(counts.values())
                if best:
                    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
                    st.markdown(f'<div class="result-banner"><div class="result-label">RINGKASAN DETEKSI</div><div class="result-answer">{total} objek</div><div style="color:#5265A8;font-size:12px;margin-top:5px">Dikenali: {" dan ".join(k for k, _ in ranked)}</div></div>', unsafe_allow_html=True)
                    st.markdown(confidence_rows(ranked), unsafe_allow_html=True)
                    summary = f"YOLO mendeteksi {total} objek. Rinciannya: {counts.get('Anjing', 0)} anjing dan {counts.get('Kucing', 0)} kucing. Nilai confidence tertinggi per kelas ditampilkan di atas."
                else:
                    st.markdown(
                        '<div class="result-banner">'
                        '<div class="result-label">HASIL PEMINDAIAN</div>'
                        '<div class="result-answer">Belum ada hewan terdeteksi</div>'
                        '<div style="color:#5265A8;font-size:13px;margin-top:8px">'
                        'Coba unggah foto kucing atau anjing yang lebih jelas, dengan pencahayaan cukup '
                        'dan tubuh hewan terlihat.'
                        '</div></div>',
                        unsafe_allow_html=True,
                    )
                    summary = (
                        "PawScan belum dapat mengenali kucing atau anjing pada gambar ini. "
                        "Coba unggah gambar lain yang lebih jelas. Hasil ini tidak selalu berarti "
                        "gambar tidak memiliki hewan."
                    )
        st.write("")
        st.markdown('<div class="panel"><div class="panel-title">🧾 Ringkasan Hasil</div><div class="panel-subtitle">', unsafe_allow_html=True)
        st.write(summary)
        st.caption("Catatan: hasil dipengaruhi kualitas gambar, data pelatihan, urutan label, dan model yang digunakan.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Analisis Gambar Lain", type="primary", use_container_width=True):
                st.session_state.analysis_result = None
                go(4)
        with c2:
            if st.button("⌂ Kembali ke Beranda", use_container_width=True):
                st.session_state.analysis_result = None
                go(1)

footer()
