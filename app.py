import base64
from pathlib import Path
import io

import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(
    page_title="PawScan | Deteksi Hewan",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "Haikal_Laporan_4.pt"
BACKGROUND_PATH = ROOT / "background_pawscan.png"
LABELS = {"cats": "Kucing", "dogs": "Anjing"}

# Background desktop dan HP dibaca dari folder yang sama dengan app.py.
# Gambar dimasukkan sebagai data URI supaya benar-benar berada di belakang UI
# dan tidak bergantung pada URL relatif browser.
DESKTOP_BG_PATH = ROOT / "background_pawscan.png"
MOBILE_BG_PATH = ROOT / "background_pawscan_mobile.png"

def image_data_uri(path):
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{encoded}"

desktop_bg = image_data_uri(DESKTOP_BG_PATH)
mobile_bg = image_data_uri(MOBILE_BG_PATH)

if desktop_bg:
    background_css = f'''        background-color: #F5F8FF;
        background-image:
            linear-gradient(rgba(246,249,255,.42), rgba(246,249,255,.48)),
            url("{desktop_bg}");
        background-size: cover;
        background-position: center top;
        background-attachment: fixed;
        background-repeat: no-repeat;
    '''
else:
    background_css = '''        background-color: #F5F8FF;
        background-image: linear-gradient(145deg,#F8FAFF,#EAF1FF,#F8FAFF);
    '''

mobile_background_rule = ""
if mobile_bg:
    mobile_background_rule = f'''    @media (max-width: 600px) {{
        .stApp {{
            background-image:
                linear-gradient(rgba(246,249,255,.36), rgba(246,249,255,.42)),
                url("{mobile_bg}") !important;
            background-size: cover !important;
            background-position: center top !important;
            background-attachment: scroll !important;
            background-repeat: no-repeat !important;
        }}
    }}
    '''

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, .stApp, p, h1, h2, h3, h4, button, input, textarea, label {{
    font-family: 'Manrope', sans-serif;
}}
/* Jangan timpa font ikon internal Streamlit. */
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}}

.stApp {{
    {background_css}
    color: #172033;
}}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {{
    background: transparent !important;
}}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.block-container {{
    max-width: 1360px;
    padding: 1.5rem 2rem 2.5rem;
}}

[data-testid="stSidebar"] {{
    background: #10182B;
    border-right: 1px solid #26334D;
}}
[data-testid="stSidebar"] * {{ color: #E8EEFF; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: #B7C3DD;
}}
.brand {{
    display:flex; align-items:center; gap:12px;
    padding:8px 0 25px;
}}
.brand-icon {{
    background:#4263EB; border-radius:14px; padding:10px;
    font-size:24px; line-height:1.2;
}}
.brand-name {{ color:white; font-size:21px; font-weight:800; }}
.brand-sub {{ color:#B7C3DD; font-size:11px; }}
.sidebar-label {{
    color:#8998B8; font-size:10px; font-weight:800;
    letter-spacing:1.5px; margin:15px 0 8px;
}}
.tip {{
    border:1px solid #2A3957; border-radius:13px;
    background:#17223A; color:#B7C3DD;
    padding:14px; font-size:12px; line-height:1.7; margin-top:22px;
}}
.tip strong {{ color:white; }}

.hero {{
    position:relative; overflow:hidden; padding:32px;
    border-radius:22px; color:white;
    background:linear-gradient(120deg,#172554 0%,#2947C7 58%,#4263EB 100%);
    box-shadow:0 15px 38px rgba(41,71,199,.16);
}}
.hero h1 {{
    color:white; font-size:clamp(28px,3.3vw,42px);
    line-height:1.2; font-weight:800; letter-spacing:-1px;
    margin:12px 0;
}}
.hero p {{ color:#E0E8FF; font-size:14px; line-height:1.8; max-width:700px; }}
.hero-kicker {{
    display:inline-block; border:1px solid rgba(255,255,255,.28);
    background:rgba(255,255,255,.10); border-radius:99px;
    padding:6px 11px; font-size:10px; font-weight:800; letter-spacing:.7px;
}}
.hero-decoration {{
    position:absolute; right:25px; top:20px;
    font-size:76px; opacity:.15; pointer-events:none;
}}

.metric-card {{
    background:rgba(255,255,255,.94); border:1px solid #E3E9F5;
    border-radius:17px; padding:19px; min-height:105px;
    box-shadow:0 5px 18px rgba(24,39,75,.04);
}}
.metric-label {{ color:#758096; font-size:12px; margin-bottom:8px; }}
.metric-value {{ color:#172033; font-size:21px; font-weight:800; }}
.metric-hint {{ color:#8994A8; font-size:11px; margin-top:6px; }}

.panel {{
    background:rgba(255,255,255,.95); border:1px solid #E3E9F5;
    border-radius:19px; padding:21px; min-height:96px;
    box-shadow:0 5px 18px rgba(24,39,75,.035);
}}
.panel-title {{ font-size:17px; font-weight:800; color:#172033; margin-bottom:5px; }}
.panel-subtitle {{ font-size:12px; color:#758096; line-height:1.7; }}

[data-testid="stFileUploaderDropzone"] {{
    background:rgba(255,255,255,.94); border:1.5px dashed #B9C9F5;
    border-radius:15px; padding:16px;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color:#4263EB; background:#F5F7FF;
}}
[data-testid="stFileUploaderDropzone"] button {{
    border-radius:10px; border:1px solid #DCE3F2;
    background:white; color:#2947C7; font-weight:700;
}}
[data-testid="stImage"] img {{ border-radius:14px; }}

.result-banner {{
    padding:18px; border-radius:15px; background:#EDF2FF;
    border:1px solid #D8E2FF; margin:12px 0 18px;
}}
.result-label {{ color:#5265A8; font-size:11px; font-weight:800; letter-spacing:1px; }}
.result-answer {{ color:#203A9B; font-size:27px; font-weight:800; margin-top:4px; }}
.conf-row {{ display:flex; align-items:center; gap:10px; margin:15px 0; }}
.conf-name {{ width:68px; color:#344054; font-size:13px; font-weight:700; }}
.conf-track {{ flex:1; height:8px; border-radius:99px; background:#E7EBF4; overflow:hidden; }}
.conf-fill {{ height:100%; background:linear-gradient(90deg,#4263EB,#7192FF); border-radius:99px; }}
.conf-pct {{ width:48px; text-align:right; color:#344054; font-size:12px; font-weight:800; }}
.empty-state {{
    text-align:center; padding:27px 15px; border:1px dashed #CFD8EB;
    border-radius:16px; background:rgba(255,255,255,.85);
    color:#667085; line-height:1.8;
}}
.footer {{ text-align:center; color:#8994A8; font-size:11px; padding-top:25px; }}

.stButton button, .stDownloadButton button {{
    border-radius:10px; font-weight:700; min-height:42px;
}}
@media (max-width:900px) {{
    .block-container {{ padding:1rem 1rem 2rem; }}
    .hero {{ padding:24px; border-radius:19px; }}
    .panel {{ padding:17px; }}
    .metric-card {{ padding:14px; }}
}}
@media (max-width:600px) {{
    .block-container {{ padding:.75rem .75rem 1.5rem; }}
    .hero {{ padding:20px 17px; border-radius:16px; }}
    .hero h1 {{ font-size:27px; letter-spacing:-.6px; }}
    .hero p {{ font-size:13px; }}
    .hero-decoration {{ font-size:43px; right:8px; top:8px; }}
    .hero-kicker {{ font-size:9px; }}
    .metric-card {{ min-height:86px; padding:11px; }}
    .metric-value {{ font-size:17px; }}
    .metric-label {{ font-size:11px; }}
    .panel {{ padding:15px; }}
    [data-testid="stFileUploaderDropzone"] {{ padding:10px; }}
    .footer {{ line-height:1.7; }}
}}
{mobile_background_rule}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def confidence_rows(items):
    parts = []
    for name, confidence in items:
        pct = max(0, min(100, confidence * 100))
        parts.append(
            f'<div class="conf-row">'
            f'<span class="conf-name">{name}</span>'
            f'<span class="conf-track"><span class="conf-fill" '
            f'style="width:{pct:.1f}%;display:block"></span></span>'
            f'<span class="conf-pct">{pct:.1f}%</span></div>'
        )
    return "".join(parts)


with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🐾</div>
            <div><div class="brand-name">PawScan</div>
            <div class="brand-sub">Animal Recognition</div></div>
        </div>
        <div class="sidebar-label">NAVIGASI</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Halaman",
        ["Dashboard", "Tentang Model"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="sidebar-label">PENGATURAN PREDIKSI</div>',
        unsafe_allow_html=True,
    )
    conf = st.slider(
        "Ambang keyakinan", 0.05, 0.95, 0.25, 0.05,
        help="Nilai lebih rendah dapat mendeteksi objek yang kurang jelas, tetapi juga bisa menambah deteksi keliru.",
    )
    st.markdown(
        """
        <div class="tip"><strong>Tips foto</strong><br>
        Gunakan foto yang jelas dengan pencahayaan cukup.
        Pastikan hewan terlihat di dalam gambar.</div>
        """,
        unsafe_allow_html=True,
    )


if page == "Tentang Model":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-decoration">🐾</div>
            <span class="hero-kicker">PAWSCAN</span>
            <h1>Tentang model</h1>
            <p>PawScan menggunakan model YOLO untuk mendeteksi kucing dan anjing dari gambar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            """
            <div class="panel"><div class="panel-title">Cara penggunaan</div>
            <div class="panel-subtitle">1. Unggah foto.<br>
            2. Atur ambang keyakinan bila diperlukan.<br>
            3. Lihat kotak deteksi dan hasil analisis.</div></div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="panel"><div class="panel-title">Informasi teknis</div>
            <div class="panel-subtitle">Antarmuka: Streamlit<br>
            Model: YOLO Ultralytics<br>
            Kelas yang ditampilkan: Kucing dan Anjing<br>
            Format: JPG, JPEG, PNG</div></div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div class="footer">PawScan · Deteksi Kucing dan Anjing</div>', unsafe_allow_html=True)
    st.stop()


st.markdown(
    """
    <div class="hero">
        <div class="hero-decoration">🐾</div>
        <span class="hero-kicker">DETEKSI HEWAN</span>
        <h1>Kenali hewan dalam<br>sekali unggah.</h1>
        <p>Unggah foto kucing atau anjing untuk melihat hasil deteksi,
        kotak objek, dan tingkat keyakinan model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

model = None
model_error = None
class_ids = []

if not MODEL_PATH.exists():
    model_error = f"File model tidak ditemukan: {MODEL_PATH}"
else:
    try:
        model = load_model()
        class_ids = [
            i for i, name in model.names.items()
            if str(name).lower() in LABELS
        ]
    except Exception as exc:
        model_error = str(exc)

m1, m2, m3 = st.columns(3)
with m1:
    status = "Siap digunakan" if model is not None else "Periksa model"
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Status model</div>'
        f'<div class="metric-value" style="font-size:18px">{status}</div>'
        f'<div class="metric-hint">Model YOLO lokal</div></div>',
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Kategori</div>'
        '<div class="metric-value">2 kelas</div>'
        '<div class="metric-hint">Kucing dan anjing</div></div>',
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Ambang keyakinan</div>'
        f'<div class="metric-value">{conf:.0%}</div>'
        f'<div class="metric-hint">Dapat diatur dari sidebar</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
left, right = st.columns([0.95, 1.05], gap="large")

with left:
    st.markdown(
        '<div class="panel"><div class="panel-title">Unggah gambar</div>'
        '<div class="panel-subtitle">Pilih foto yang ingin dianalisis oleh model.</div></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Pilih foto", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Format yang didukung: JPG, JPEG, PNG.",
    )
    image = None
    if uploaded is not None:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            st.image(image, caption="Foto yang diunggah", use_container_width=True)
        except Exception:
            st.error("Foto tidak dapat dibuka. Silakan coba file gambar lain.")
    else:
        st.markdown(
            '<div class="empty-state"><div style="font-size:32px">📷</div>'
            '<strong>Belum ada gambar</strong><br>Pilih foto untuk memulai prediksi.</div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown(
        '<div class="panel"><div class="panel-title">Hasil analisis</div>'
        '<div class="panel-subtitle">Hasil prediksi akan ditampilkan di area ini.</div></div>',
        unsafe_allow_html=True,
    )

    if model is None:
        st.error("Model belum berhasil dimuat. Periksa lokasi file model.")
        with st.expander("Detail kesalahan"):
            st.code(model_error or "Kesalahan tidak diketahui.")
    elif image is None:
        st.markdown(
            '<div class="empty-state"><div style="font-size:32px">🔎</div>'
            '<strong>Menunggu gambar</strong><br>Unggah foto untuk memulai prediksi.</div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            with st.spinner("Sedang menganalisis gambar..."):
                result = model.predict(
                    image, imgsz=640, conf=conf,
                    classes=class_ids if class_ids else None,
                    verbose=False,
                )[0]

            annotated = result.plot(line_width=3)[:, :, ::-1]
            best = {}
            counts = {}

            for box in result.boxes:
                raw_name = str(result.names[int(box.cls)]).lower()
                if raw_name not in LABELS:
                    continue
                name = LABELS[raw_name]
                score = float(box.conf)
                best[name] = max(best.get(name, 0.0), score)
                counts[name] = counts.get(name, 0) + 1

            if not best:
                st.markdown(
                    '<div class="empty-state"><div style="font-size:32px">🔍</div>'
                    '<strong>Hewan belum terdeteksi</strong><br>'
                    'Coba foto lain atau turunkan ambang keyakinan.</div>',
                    unsafe_allow_html=True,
                )
            else:
                ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
                top_name = " dan ".join(name for name, _ in ranked)
                total = sum(counts.values())
                st.markdown(
                    f'<div class="result-banner"><div class="result-label">DETEKSI BERHASIL</div>'
                    f'<div class="result-answer">{top_name}</div>'
                    f'<div style="color:#5265A8;font-size:12px;margin-top:5px">'
                    f'{total} objek terdeteksi</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(confidence_rows(ranked), unsafe_allow_html=True)
                with st.expander("Rincian jumlah objek"):
                    for name in LABELS.values():
                        st.write(f"**{name}:** {counts.get(name, 0)} objek")

            st.image(annotated, caption="Gambar dengan kotak deteksi", use_container_width=True)
            output = io.BytesIO()
            Image.fromarray(annotated).save(output, format="PNG")
            st.download_button(
                "Unduh gambar hasil deteksi",
                data=output.getvalue(),
                file_name="hasil_deteksi_pawscan.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception as exc:
            st.error("Prediksi gagal diproses. Periksa model dan format gambar.")
            with st.expander("Detail kesalahan"):
                st.code(str(exc))

st.markdown(
    '<div class="footer">PawScan · Deteksi Kucing dan Anjing<br>'
    'Hasil dapat dipengaruhi oleh model dan kualitas gambar.</div>',
    unsafe_allow_html=True,
)
