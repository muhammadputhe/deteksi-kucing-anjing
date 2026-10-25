import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="PawScan | Klasifikasi & Deteksi",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
YOLO_PATH = ROOT / "model" / "Haikal_Laporan_4.pt"
CLASSIFIER_PATH = ROOT / "Haikal_Laporan 2 (1).h5"
DESKTOP_BG_PATH = ROOT / "background_pawscan.png"
MOBILE_BG_PATH = ROOT / "background_pawscan_mobile.png"

# Model H5 memiliki input 224x224 RGB dan output 3 kelas.
# Nama kelas asli tidak tersimpan di file H5; atur label sesuai urutan
# class_indices dari proses training agar prediksi diberi nama yang benar.
CLASSIFIER_IMAGE_SIZE = (224, 224)


def image_data_uri(path):
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{encoded}"


desktop_bg = image_data_uri(DESKTOP_BG_PATH)
mobile_bg = image_data_uri(MOBILE_BG_PATH)

if desktop_bg:
    background_css = f"""
        background-color: #F5F8FF;
        background-image:
            linear-gradient(rgba(246,249,255,.48), rgba(246,249,255,.50)),
            url("{desktop_bg}");
        background-size: cover;
        background-position: center top;
        background-attachment: fixed;
        background-repeat: no-repeat;
    """
else:
    background_css = "background: linear-gradient(145deg,#F8FAFF,#EAF1FF,#F8FAFF);"

mobile_background_rule = ""
if mobile_bg:
    mobile_background_rule = f"""
    @media (max-width: 600px) {{
        .stApp {{
            background-image:
                linear-gradient(rgba(246,249,255,.48), rgba(246,249,255,.52)),
                url("{mobile_bg}") !important;
            background-size: cover !important;
            background-position: center top !important;
            background-attachment: scroll !important;
            background-repeat: no-repeat !important;
        }}
    }}
    """

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, .stApp, p, h1, h2, h3, h4, button, input, textarea, label {{
    font-family: 'Manrope', sans-serif;
}}
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}}
.stApp {{ {background_css} color:#172033; }}
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"] {{ background:transparent !important; }}
#MainMenu, footer {{ visibility:hidden; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ max-width:1360px; padding:1.5rem 2rem 2.5rem; }}

[data-testid="stSidebar"] {{ background:#10182B; border-right:1px solid #26334D; }}
[data-testid="stSidebar"] * {{ color:#E8EEFF; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color:#B7C3DD; }}
.brand {{ display:flex; align-items:center; gap:12px; padding:8px 0 24px; }}
.brand-icon {{ background:#4263EB; border-radius:14px; padding:10px; font-size:24px; line-height:1.2; }}
.brand-name {{ color:white; font-size:21px; font-weight:800; }}
.brand-sub {{ color:#B7C3DD; font-size:11px; }}
.sidebar-label {{ color:#8998B8; font-size:10px; font-weight:800; letter-spacing:1.5px; margin:15px 0 8px; }}
.tip {{ border:1px solid #2A3957; border-radius:13px; background:#17223A; color:#B7C3DD; padding:14px; font-size:12px; line-height:1.7; margin-top:20px; }}
.tip strong {{ color:white; }}

.hero {{ position:relative; overflow:hidden; padding:30px; border-radius:22px; color:white; background:linear-gradient(120deg,#172554 0%,#2947C7 58%,#4263EB 100%); box-shadow:0 15px 38px rgba(41,71,199,.16); }}
.hero h1 {{ color:white; font-size:clamp(27px,3.3vw,42px); line-height:1.2; font-weight:800; letter-spacing:-1px; margin:12px 0; }}
.hero p {{ color:#E0E8FF; font-size:14px; line-height:1.8; max-width:760px; }}
.hero-kicker {{ display:inline-block; border:1px solid rgba(255,255,255,.28); background:rgba(255,255,255,.10); border-radius:99px; padding:6px 11px; font-size:10px; font-weight:800; letter-spacing:.7px; }}
.hero-decoration {{ position:absolute; right:25px; top:20px; font-size:72px; opacity:.15; pointer-events:none; }}

.metric-card {{ background:rgba(255,255,255,.94); border:1px solid #E3E9F5; border-radius:17px; padding:18px; min-height:100px; box-shadow:0 5px 18px rgba(24,39,75,.04); }}
.metric-label {{ color:#758096; font-size:12px; margin-bottom:8px; }}
.metric-value {{ color:#172033; font-size:20px; font-weight:800; }}
.metric-hint {{ color:#8994A8; font-size:11px; margin-top:6px; }}
.panel {{ background:rgba(255,255,255,.95); border:1px solid #E3E9F5; border-radius:18px; padding:20px; min-height:82px; box-shadow:0 5px 18px rgba(24,39,75,.035); }}
.panel-title {{ font-size:17px; font-weight:800; color:#172033; margin-bottom:5px; }}
.panel-subtitle {{ font-size:12px; color:#758096; line-height:1.7; }}
[data-testid="stFileUploaderDropzone"] {{ background:rgba(255,255,255,.95); border:1.5px dashed #B9C9F5; border-radius:15px; padding:14px; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color:#4263EB; background:#F5F7FF; }}
[data-testid="stFileUploaderDropzone"] button {{ border-radius:10px; border:1px solid #DCE3F2; background:white; color:#2947C7; font-weight:700; }}
[data-testid="stImage"] img {{ border-radius:14px; }}
.result-banner {{ padding:17px; border-radius:15px; background:#EDF2FF; border:1px solid #D8E2FF; margin:12px 0 18px; }}
.result-label {{ color:#5265A8; font-size:11px; font-weight:800; letter-spacing:1px; }}
.result-answer {{ color:#203A9B; font-size:27px; font-weight:800; margin-top:4px; }}
.conf-row {{ display:flex; align-items:center; gap:10px; margin:14px 0; }}
.conf-name {{ width:100px; color:#344054; font-size:12px; font-weight:700; }}
.conf-track {{ flex:1; height:8px; border-radius:99px; background:#E7EBF4; overflow:hidden; }}
.conf-fill {{ height:100%; background:linear-gradient(90deg,#4263EB,#7192FF); border-radius:99px; }}
.conf-pct {{ width:50px; text-align:right; color:#344054; font-size:12px; font-weight:800; }}
.empty-state {{ text-align:center; padding:25px 15px; border:1px dashed #CFD8EB; border-radius:16px; background:rgba(255,255,255,.88); color:#667085; line-height:1.8; }}
.footer {{ text-align:center; color:#8994A8; font-size:11px; padding-top:24px; }}
.stButton button, .stDownloadButton button {{ border-radius:10px; font-weight:700; min-height:42px; }}
@media (max-width:900px) {{
    .block-container {{ padding:1rem 1rem 2rem; }}
    .hero {{ padding:24px; border-radius:19px; }}
    .panel {{ padding:16px; }}
    .metric-card {{ padding:14px; }}
}}
@media (max-width:600px) {{
    .block-container {{ padding:.7rem .7rem 1.5rem; }}
    .hero {{ padding:20px 16px; border-radius:16px; }}
    .hero h1 {{ font-size:26px; letter-spacing:-.6px; }}
    .hero p {{ font-size:13px; }}
    .hero-decoration {{ font-size:42px; right:8px; top:8px; }}
    .hero-kicker {{ font-size:9px; }}
    .metric-card {{ min-height:82px; padding:11px; }}
    .metric-value {{ font-size:16px; }}
    .metric-label {{ font-size:10px; }}
    .panel {{ padding:14px; }}
    [data-testid="stFileUploaderDropzone"] {{ padding:9px; }}
    .conf-name {{ width:76px; }}
    .footer {{ line-height:1.7; }}
}}
{mobile_background_rule}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_classifier(path_string):
    # Import di dalam fungsi agar aplikasi deteksi YOLO tetap dapat dibuka
    # walaupun TensorFlow belum terpasang, dengan pesan error yang jelas.
    from tensorflow.keras.models import load_model
    return load_model(path_string, compile=False)


@st.cache_resource
def load_yolo(path_string):
    from ultralytics import YOLO
    return YOLO(path_string)


def confidence_rows(items, name_width=100):
    html = []
    for name, confidence in items:
        pct = max(0.0, min(100.0, float(confidence) * 100))
        html.append(
            f'<div class="conf-row"><span class="conf-name" style="width:{name_width}px">{name}</span>'
            f'<span class="conf-track"><span class="conf-fill" style="width:{pct:.1f}%;display:block"></span></span>'
            f'<span class="conf-pct">{pct:.1f}%</span></div>'
        )
    return "".join(html)


def image_to_rgb(uploaded_file):
    return ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")


def classifier_predict(image, model, class_names):
    resized = image.resize(CLASSIFIER_IMAGE_SIZE)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)
    pred = np.asarray(model.predict(batch, verbose=0))
    pred = pred.reshape(-1)
    if pred.size != 3:
        raise ValueError(
            f"Model klasifikasi menghasilkan {pred.size} nilai, sedangkan arsitektur file ini diperkirakan memiliki 3 kelas."
        )
    # File H5 tidak menyimpan nama kelas dataset, jadi label diambil dari input sidebar.
    order = np.argsort(pred)[::-1]
    return [(class_names[int(i)], float(pred[int(i)])) for i in order]


def yolo_predict(image, model, confidence):
    names = model.names
    allowed = {"cats": "Kucing", "cat": "Kucing", "dogs": "Anjing", "dog": "Anjing"}
    class_ids = [
        int(idx) for idx, name in names.items()
        if str(name).strip().lower() in allowed
    ]
    result = model.predict(
        image, imgsz=640, conf=confidence,
        classes=class_ids if class_ids else None,
        verbose=False,
    )[0]
    annotated = result.plot(line_width=3)[:, :, ::-1]
    counts = {}
    best = {}
    for box in result.boxes:
        raw = str(result.names[int(box.cls)]).strip().lower()
        label = allowed.get(raw)
        if label is None:
            continue
        score = float(box.conf)
        counts[label] = counts.get(label, 0) + 1
        best[label] = max(best.get(label, 0.0), score)
    return annotated, counts, best


with st.sidebar:
    st.markdown(
        """
        <div class="brand"><div class="brand-icon">🐾</div><div>
        <div class="brand-name">PawScan</div>
        <div class="brand-sub">Animal Recognition</div></div></div>
        <div class="sidebar-label">NAVIGASI</div>
        """, unsafe_allow_html=True
    )
    page = st.radio(
        "Menu",
        ["Dashboard", "Tentang Model"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="sidebar-label">PENGATURAN YOLO</div>', unsafe_allow_html=True)
    yolo_conf = st.slider("Ambang keyakinan deteksi", 0.05, 0.95, 0.25, 0.05)
    st.markdown('<div class="sidebar-label">LABEL KLASIFIKASI</div>', unsafe_allow_html=True)
    st.caption("Sesuaikan label menurut urutan kelas saat model H5 dilatih.")
    class_names = [
        st.text_input("Nama kelas 1", "Kelas 1", key="class_label_1"),
        st.text_input("Nama kelas 2", "Kelas 2", key="class_label_2"),
        st.text_input("Nama kelas 3", "Kelas 3", key="class_label_3"),
    ]
    st.markdown(
        '<div class="tip"><strong>Tips foto</strong><br>'
        'Gunakan gambar yang jelas dan pencahayaan cukup. '
        'Klasifikasi memilih satu kelas utama; YOLO dapat menandai beberapa objek.</div>',
        unsafe_allow_html=True,
    )


if page == "Tentang Model":
    st.markdown(
        """
        <div class="hero"><div class="hero-decoration">🐾</div>
        <span class="hero-kicker">PAWSCAN</span><h1>Dua jenis analisis gambar</h1>
        <p>Klasifikasi gambar memprediksi kelas utama gambar, sedangkan deteksi objek YOLO mencari lokasi objek dan menampilkan kotak pembatas.</p></div>
        """, unsafe_allow_html=True
    )
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            '<div class="panel"><div class="panel-title">Klasifikasi gambar</div>'
            '<div class="panel-subtitle">Model CNN dari file H5. Input model 224 × 224 piksel, RGB, dan output berjumlah 3 kelas. Nama kelas perlu disesuaikan dengan label training.</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="panel"><div class="panel-title">Deteksi objek YOLO</div>'
            '<div class="panel-subtitle">Menggunakan file .pt lokal untuk mendeteksi lokasi objek. Gambar hasil menampilkan kotak deteksi dan nilai confidence.</div></div>',
            unsafe_allow_html=True
        )
    st.markdown('<div class="footer">PawScan · Klasifikasi gambar dan deteksi objek</div>', unsafe_allow_html=True)
    st.stop()


st.markdown(
    """
    <div class="hero"><div class="hero-decoration">🐾</div>
    <span class="hero-kicker">KLASIFIKASI + DETEKSI OBJEK</span>
    <h1>Satu gambar, dua cara analisis.</h1>
    <p>Pilih klasifikasi gambar untuk memperkirakan kelas utama, atau deteksi YOLO untuk menemukan objek dan melihat kotak pembatasnya.</p></div>
    """, unsafe_allow_html=True
)
st.write("")

# Status model disajikan terpisah agar pengguna mengetahui file mana yang belum tersedia.
classifier_ready = CLASSIFIER_PATH.exists()
yolo_ready = YOLO_PATH.exists()
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Model klasifikasi</div>'
        f'<div class="metric-value" style="font-size:17px">{"Tersedia" if classifier_ready else "Belum tersedia"}</div>'
        f'<div class="metric-hint">File .h5 · 3 kelas</div></div>',
        unsafe_allow_html=True
    )
with s2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Model deteksi YOLO</div>'
        f'<div class="metric-value" style="font-size:17px">{"Tersedia" if yolo_ready else "Belum tersedia"}</div>'
        f'<div class="metric-hint">File .pt lokal</div></div>',
        unsafe_allow_html=True
    )
with s3:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Fitur</div>'
        '<div class="metric-value" style="font-size:17px">2 mode</div>'
        '<div class="metric-hint">Klasifikasi dan deteksi objek</div></div>',
        unsafe_allow_html=True
    )

st.write("")
mode = st.radio(
    "Pilih mode analisis",
    ["Klasifikasi Gambar", "Deteksi Objek YOLO"],
    horizontal=True,
    label_visibility="visible",
)
st.write("")

left, right = st.columns([0.95, 1.05], gap="large")
with left:
    st.markdown(
        '<div class="panel"><div class="panel-title">Unggah gambar</div>'
        '<div class="panel-subtitle">Pilih satu gambar JPG, JPEG, atau PNG untuk dianalisis.</div></div>',
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader(
        "Pilih gambar", type=["jpg", "jpeg", "png"],
        key="pawscan_image_uploader", label_visibility="collapsed",
    )
    image = None
    if uploaded is not None:
        try:
            image = image_to_rgb(uploaded)
            st.image(image, caption="Gambar masukan", use_container_width=True)
        except Exception:
            st.error("Gambar tidak dapat dibaca. Silakan pilih file gambar lain.")
    else:
        st.markdown(
            '<div class="empty-state"><div style="font-size:30px">📷</div>'
            '<strong>Menunggu gambar</strong><br>Unggah foto untuk memulai analisis.</div>',
            unsafe_allow_html=True
        )

with right:
    st.markdown(
        '<div class="panel"><div class="panel-title">Hasil analisis</div>'
        '<div class="panel-subtitle">Hasil akan muncul di sini setelah gambar dipilih.</div></div>',
        unsafe_allow_html=True
    )
    if image is None:
        st.markdown(
            '<div class="empty-state"><div style="font-size:30px">🔎</div>'
            '<strong>Belum ada hasil</strong><br>Pilih gambar di panel sebelah kiri.</div>',
            unsafe_allow_html=True
        )
    elif mode == "Klasifikasi Gambar":
        if not classifier_ready:
            st.error(f"File model klasifikasi tidak ditemukan: {CLASSIFIER_PATH.name}")
        else:
            try:
                with st.spinner("Memuat model klasifikasi dan menganalisis gambar..."):
                    classifier = load_classifier(str(CLASSIFIER_PATH))
                    predictions = classifier_predict(image, classifier, class_names)
                top_label, top_score = predictions[0]
                st.markdown(
                    f'<div class="result-banner"><div class="result-label">PREDIKSI KELAS UTAMA</div>'
                    f'<div class="result-answer">{top_label}</div>'
                    f'<div style="color:#5265A8;font-size:12px;margin-top:5px">Skor model: {top_score:.1%}</div></div>',
                    unsafe_allow_html=True
                )
                st.markdown("**Skor untuk setiap kelas**")
                st.markdown(confidence_rows(predictions), unsafe_allow_html=True)
                st.caption("Catatan: label kelas harus mengikuti urutan kelas saat training. Skor model bukan jaminan kebenaran prediksi.")
            except ImportError:
                st.error("TensorFlow belum terpasang. Jalankan `pip install tensorflow` lalu mulai ulang aplikasi.")
            except Exception as exc:
                st.error("Klasifikasi gagal dijalankan.")
                with st.expander("Detail kesalahan"):
                    st.code(str(exc))
    else:
        if not yolo_ready:
            st.error(
                "File model YOLO tidak ditemukan. Letakkan `Haikal_Laporan_4.pt` "
                "di folder `model/` di samping app.py."
            )
        else:
            try:
                with st.spinner("YOLO sedang mendeteksi objek..."):
                    detector = load_yolo(str(YOLO_PATH))
                    annotated, counts, best = yolo_predict(image, detector, yolo_conf)
                if not best:
                    st.markdown(
                        '<div class="empty-state"><div style="font-size:30px">🔍</div>'
                        '<strong>Belum ada objek terdeteksi</strong><br>'
                        'Coba gambar lain atau ubah ambang keyakinan.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
                    total = sum(counts.values())
                    st.markdown(
                        f'<div class="result-banner"><div class="result-label">DETEKSI BERHASIL</div>'
                        f'<div class="result-answer">{" dan ".join(k for k, _ in ranked)}</div>'
                        f'<div style="color:#5265A8;font-size:12px;margin-top:5px">{total} objek terdeteksi</div></div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(confidence_rows(ranked, 75), unsafe_allow_html=True)
                    with st.expander("Rincian jumlah objek"):
                        for label in ["Kucing", "Anjing"]:
                            st.write(f"**{label}:** {counts.get(label, 0)} objek")
                st.image(annotated, caption="Gambar dengan kotak deteksi YOLO", use_container_width=True)
                out = io.BytesIO()
                Image.fromarray(annotated).save(out, format="PNG")
                st.download_button(
                    "Unduh gambar hasil deteksi",
                    data=out.getvalue(),
                    file_name="hasil_deteksi_pawscan.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except ImportError:
                st.error("Ultralytics belum terpasang. Jalankan `pip install ultralytics` lalu mulai ulang aplikasi.")
            except Exception as exc:
                st.error("Deteksi YOLO gagal dijalankan.")
                with st.expander("Detail kesalahan"):
                    st.code(str(exc))

st.markdown(
    '<div class="footer">PawScan · Klasifikasi gambar dan deteksi objek YOLO<br>'
    'Hasil dipengaruhi oleh model, label kelas, preprocessing, dan kualitas gambar.</div>',
    unsafe_allow_html=True
)
