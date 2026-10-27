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
MASCOT_PATH = ASSET_DIR / "maskot_pawscan.png"  # Opsional: maskot kucing + anjing dalam satu PNG transparan
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
.hero-decoration{{position:absolute;right:28px;top:20px;font-size:78px;opacity:.18;pointer-events:none;}}
.welcome-hero{{position:relative;isolation:isolate;min-height:180px;padding:18px 18px 16px;border-radius:15px;overflow:hidden;background:linear-gradient(112deg,#172554 0%,#2947C7 63%,#3659DE 100%);box-shadow:0 10px 24px #2947c729;}}
.welcome-hero .welcome-copy{{position:relative;z-index:2;width:69%;}}
.welcome-hero .hero-kicker{{font-size:8px;padding:4px 7px;letter-spacing:.35px;}}
.welcome-hero h1{{font-size:clamp(19px,2vw,24px);line-height:1.08;letter-spacing:-.35px;margin:10px 0 9px;max-width:255px;}}
.welcome-hero p{{font-size:9px;line-height:1.55;max-width:245px;margin:0;color:#E7ECFF;}}
.welcome-mascot-combined{{position:absolute;z-index:1;right:-5px;bottom:-4px;width:47%;height:88%;object-fit:contain;object-position:center bottom;pointer-events:none;}}
.welcome-mascot-pair{{position:absolute;z-index:1;right:0;bottom:0;width:43%;height:80%;display:flex;align-items:flex-end;justify-content:center;gap:0;pointer-events:none;}}
.welcome-mascot-pair img{{width:53%;height:100%;object-fit:contain;object-position:center bottom;filter:drop-shadow(0 3px 3px #101c4b22);}}
.welcome-hero .hero-decoration{{display:none;}}
@media(max-width:900px){{.welcome-hero{{min-height:172px;padding:17px 15px;}}.welcome-hero .welcome-copy{{width:70%;}}.welcome-hero h1{{font-size:21px;}}}}
@media(max-width:600px){{.welcome-hero{{min-height:165px;padding:16px 13px;border-radius:14px;}}.welcome-hero .welcome-copy{{width:70%;}}.welcome-hero h1{{font-size:19px;margin:9px 0 7px;}}.welcome-hero p{{font-size:8px;line-height:1.45;}}.welcome-mascot-combined{{width:45%;height:80%;right:-5px;}}.welcome-mascot-pair{{width:42%;height:72%;}}}} .feature-icon{{width:44px;height:44px;object-fit:contain;margin:5px 0 8px;}} .asset-action{{width:20px;height:20px;object-fit:contain;vertical-align:middle;margin-right:7px;}} .result-pet-icon{{width:112px;height:112px;object-fit:contain;display:block;margin:0 auto 8px;filter:drop-shadow(0 8px 12px #2947c71a);}} .section-eyebrow{{font-size:10px;font-weight:800;letter-spacing:1.2px;color:#6478B8;text-transform:uppercase;margin-bottom:7px;}} .soft-note{{border-radius:14px;background:linear-gradient(120deg,#F0F5FF,#FAF7FF);padding:14px 16px;border:1px solid #E2E9FF;color:#596783;font-size:12px;line-height:1.8;}} .stRadio label{{font-weight:700;}} .stProgress>div>div>div>div{{background:linear-gradient(90deg,#4263EB,#8B5CF6);}}
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
    result = model.predict(image, imgsz=640, conf=confidence,
                           classes=class_ids if class_ids else None, verbose=False)[0]
    annotated = result.plot(line_width=3)[:, :, ::-1]
    counts, best = {}, {}
    for box in result.boxes:
        raw = str(result.names[int(box.cls)]).strip().lower()
        label = allowed.get(raw)
        if label is None:
            continue
        score = float(box.conf)
        counts[label] = counts.get(label, 0) + 1
        best[label] = max(best.get(label, 0.0), score)
    return annotated, counts, best

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
    st.markdown('<div class="footer">created by <b>Muhammad Haikal Asy’ary · 2408108010065</b><br>AI Animal Recognition · CNN & YOLO | Universitas Syiah Kuala</div>', unsafe_allow_html=True)

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
labels = ["Intro", "Registrasi", "Metode", "Unggah & Analisis", "Hasil"]
progress = "".join(f'<div class="progress-step {"active" if i <= step else ""}"></div>' for i in range(1, 6))
st.markdown(f'<div class="progress-wrap">{progress}</div><div class="step-label">Sesi {step} dari 5 · {labels[step-1]}</div>', unsafe_allow_html=True)

if step == 1:
    st.markdown("""
    <div class="hero">
    <span class="hero-kicker">SMART PET RECOGNITION</span>
    <h1>Kenali Anjing & Kucing<br>dengan <span style="color:#AFC2FF">Kekuatan AI</span></h1>
    <p>Selamat datang di PawScan! Pilih metode klasifikasi CNN untuk mengenali jenis hewan, atau deteksi objek menggunakan YOLO untuk menemukan dan menandai hewan dalam gambar. Ikuti lima sesi sederhana untuk melihat hasilnya.</p>
    </div>""", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3, gap="medium")
    for col, num, title, desc in [
        (c1, "01", "Isi registrasi", "Masukkan nama dan kampus."),
        (c2, "02", "Pilih metode", "Tentukan CNN atau deteksi YOLO."),
        (c3, "03", "Unggah & lihat hasil", "Unggah foto dan baca ringkasan AI."),
    ]:
        with col:
            st.markdown(f'<div class="panel"><div class="pill">LANGKAH {num}</div><div class="panel-title" style="margin-top:13px">{title}</div><div class="panel-subtitle">{desc}</div></div>', unsafe_allow_html=True)
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
            '<h1>Halo, Selamat<br>Bergabung di PawScan!</h1>'
            '<p>Hanya satu langkah lagi untuk memulai petualangan analisis jejak kaki kesayanganmu!</p>'
            '</div>' + mascot_markup + '</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="panel"><div class="panel-title">📝 Registrasi Singkat</div><div class="panel-subtitle">Isi data berikut untuk melanjutkan ke sesi analisis.</div>', unsafe_allow_html=True)
        full_name = st.text_input("Nama lengkap", value=st.session_state.full_name, placeholder="Masukkan nama lengkap")
        campus = st.text_input("Nama kampus", value=st.session_state.campus, placeholder="Masukkan nama kampus")
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        with a:
            if st.button("← Kembali", use_container_width=True):
                go(1)
        with b:
            if st.button("Lanjutkan →", type="primary", use_container_width=True):
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
    st.markdown(f'<div class="panel"><div class="panel-title">📷 Unggah & Analisis Gambar</div><div class="panel-subtitle">Metode yang digunakan: <b>{escape(mode)}</b>. Unggah gambar anjing atau kucing yang jelas untuk memulai analisis.</div></div>', unsafe_allow_html=True)
    show_member_card(compact=True)
    st.write("")
    st.caption("CNN · Model H5 · 224 × 224 RGB · kelas: Anjing dan Kucing" if mode == "Klasifikasi CNN" else "YOLO · Model .pt · kotak deteksi, jumlah objek, dan confidence")
    left, right = st.columns([1, 1], gap="large")
    with left:
        uploaded = st.file_uploader("Pilih gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="pawscan_upload")
        image = None
        if uploaded is not None:
            try:
                image = image_to_rgb(uploaded)
                st.image(image, caption="Pratinjau gambar", use_container_width=True)
            except Exception:
                st.error("Gambar tidak dapat dibaca. Silakan pilih file gambar lain.")
        else:
            show_empty_illustration()
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Sebelum memulai</div><div class="panel-subtitle">• Gunakan foto dengan objek yang terlihat jelas.<br>• Pastikan pencahayaan cukup.<br>• Untuk CNN, satu gambar akan mendapat satu prediksi utama.<br>• Untuk YOLO, hasil bergantung pada bobot model .pt yang tersedia.</div></div>', unsafe_allow_html=True)
        if mode == "Deteksi Objek YOLO":
            yolo_conf = st.slider("Ambang keyakinan YOLO", 0.05, 0.95, 0.25, 0.05)
        else:
            yolo_conf = 0.25
        st.write("")
        if st.button("✨ Analisis Sekarang", type="primary", use_container_width=True, disabled=(image is None)):
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
                        st.error(f"Model YOLO tidak ditemukan: {YOLO_PATH}. Letakkan file Haikal_Laporan 4.pt di folder model/.")
                    else:
                        loading_slot = st.empty()
                        with loading_slot.container():
                            show_loading_visual("PawScan sedang memindai objek dengan YOLO. Mohon tunggu...")
                        with st.spinner("YOLO sedang mendeteksi objek..."):
                            model = load_yolo(str(YOLO_PATH))
                            annotated, counts, best = yolo_predict(image, model, yolo_conf)
                        loading_slot.empty()
                        st.session_state.analysis_result = {"mode": mode, "annotated": annotated, "counts": counts, "best": best, "image": image.copy()}
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
                st.image(result["annotated"], caption="Gambar dengan kotak deteksi YOLO", use_container_width=True)
                out = io.BytesIO()
                Image.fromarray(result["annotated"]).save(out, format="PNG")
                if DOWNLOAD_ICON_PATH.exists():
                    st.image(str(DOWNLOAD_ICON_PATH), width=34)
                st.download_button("Unduh gambar hasil YOLO", data=out.getvalue(), file_name="hasil_pawscan.png", mime="image/png", use_container_width=True)
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
                    st.warning("Tidak ada anjing atau kucing yang terdeteksi pada ambang keyakinan ini. Coba gambar lain atau turunkan ambang keyakinan pada sesi analisis.")
                    summary = "YOLO belum menemukan objek anjing atau kucing pada gambar dengan ambang keyakinan yang digunakan."
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
