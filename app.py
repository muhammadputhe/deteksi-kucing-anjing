import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(
    page_title="PawScan | Deteksi Kucing & Anjing",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model/Haikal_Laporan 4.pt"
LABELS = {"cats": "Kucing", "dogs": "Anjing"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #172033;
    --muted: #667085;
    --blue: #4263EB;
    --blue-dark: #2947C7;
    --line: #E7EAF2;
    --surface: #FFFFFF;
    --soft: #F5F7FC;
}
html, body, .stApp, .stMarkdown, .stText, p, h1, h2, h3, h4,
button, input, textarea, label {
    font-family: 'Manrope', sans-serif;
}
/* Keep Streamlit's own icon font intact (upload, sidebar controls, etc.). */
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] span,
.material-symbols-rounded, .material-icons {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
.stApp {
    background:
        radial-gradient(ellipse at 8% 8%, rgba(206, 222, 255, .52) 0, rgba(206, 222, 255, 0) 34%),
        radial-gradient(ellipse at 94% 36%, rgba(224, 235, 255, .72) 0, rgba(224, 235, 255, 0) 30%),
        linear-gradient(145deg, #F8FAFF 0%, #F0F5FF 48%, #F8FAFF 100%);
    color: var(--ink);
    position: relative;
}
/* Decorative paw prints stay subtle and never block controls. */
.stApp::before, .stApp::after {
    position: fixed;
    z-index: 0;
    pointer-events: none;
    user-select: none;
    color: #4263EB;
    opacity: .055;
    line-height: 1;
}
.stApp::before {
    content: "🐾";
    font-size: 150px;
    top: 16%;
    right: 2.5%;
    transform: rotate(-18deg);
}
.stApp::after {
    content: "🐾";
    font-size: 112px;
    bottom: 7%;
    left: 19%;
    transform: rotate(20deg);
}
.block-container, [data-testid="stSidebar"] {
    position: relative;
    z-index: 1;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: rgba(246,248,252,0.85); }
.block-container { max-width: 1320px; padding-top: 1.5rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] {
    background: #10182B;
    border-right: 1px solid #1E2A43;
}
[data-testid="stSidebar"] * { color: #E8EEFF; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #AEBAD4; }
.sidebar-brand {
    display:flex; align-items:center; gap:12px; padding: 8px 0 26px;
}
.brand-icon {
    width:46px; height:46px; border-radius:15px; display:flex;
    align-items:center; justify-content:center; background:#4263EB;
    font-size:25px; box-shadow:0 8px 22px rgba(66,99,235,.28);
}
.brand-name { font-size:20px; font-weight:800; color:#FFFFFF; letter-spacing:-.5px; }
.brand-sub { font-size:11px; color:#AEBAD4; margin-top:2px; }
.sidebar-label {
    color:#7887A8; font-size:10px; font-weight:800; letter-spacing:1.6px;
    text-transform:uppercase; margin: 16px 0 8px;
}
.sidebar-note {
    margin-top:28px; padding:14px; border:1px solid #293653; border-radius:14px;
    background:#17223A; font-size:12px; line-height:1.6; color:#AEBAD4;
}
.sidebar-note strong { color:#FFFFFF; }
.hero {
    padding: 28px 30px; border-radius: 24px;
    background: linear-gradient(120deg, #172554 0%, #2947C7 62%, #4263EB 100%);
    color: white; position:relative; overflow:hidden;
    box-shadow: 0 16px 40px rgba(41,71,199,.16);
}
.hero:after {
    content:"🐾"; position:absolute; right:32px; top:14px;
    font-size:92px; opacity:.12; transform:rotate(-18deg);
}
.hero-kicker {
    display:inline-block; padding:6px 10px; border:1px solid rgba(255,255,255,.25);
    border-radius:999px; background:rgba(255,255,255,.10);
    color:#E4EAFF; font-size:11px; font-weight:700; letter-spacing:.8px;
}
.hero h1 {
    color:#FFFFFF; font-size:clamp(27px,3vw,39px); line-height:1.18;
    letter-spacing:-1.3px; margin:16px 0 10px; font-weight:800;
}
.hero p { color:#DCE5FF; font-size:14px; max-width:650px; line-height:1.8; margin:0; }
.section-heading { font-size:18px; font-weight:800; letter-spacing:-.4px; color:#172033; margin: 0 0 4px; }
.section-sub { color:#667085; font-size:13px; margin:0 0 16px; }
.metric-card {
    background:#FFFFFF; border:1px solid #E7EAF2; border-radius:17px;
    padding:17px 18px; min-height:105px; box-shadow:0 4px 15px rgba(24,39,75,.025);
}
.metric-label { color:#667085; font-size:12px; font-weight:600; margin-bottom:9px; }
.metric-value { color:#172033; font-size:22px; font-weight:800; letter-spacing:-.6px; }
.metric-hint { color:#8A94A8; font-size:11px; margin-top:5px; }
.panel {
    background:#FFFFFF; border:1px solid #E7EAF2; border-radius:20px;
    padding:22px; box-shadow:0 4px 18px rgba(24,39,75,.025);
}
.panel-title { color:#172033; font-size:17px; font-weight:800; margin:0 0 5px; }
.panel-subtitle { color:#7A8498; font-size:12px; margin:0 0 18px; line-height:1.6; }
.drop-hint {
    background:#F5F7FF; border:1px dashed #B9C7FF; border-radius:15px;
    padding:16px; color:#586783; text-align:center; font-size:12px; line-height:1.7;
}
.result-banner {
    padding:17px 18px; border-radius:15px; background:#EEF2FF;
    border:1px solid #DDE5FF; margin-bottom:16px;
}
.result-label { color:#5265A8; font-size:11px; font-weight:800; letter-spacing:.8px; text-transform:uppercase; }
.result-answer { color:#203A9B; font-size:27px; font-weight:800; letter-spacing:-.7px; margin-top:3px; }
.conf-row { display:flex; align-items:center; gap:12px; margin:14px 0; }
.conf-name { width:76px; font-size:13px; font-weight:700; color:#344054; }
.conf-track { flex:1; height:8px; background:#EDF0F6; border-radius:99px; overflow:hidden; }
.conf-fill { height:100%; background:linear-gradient(90deg,#4263EB,#6C8CFF); border-radius:99px; }
.conf-pct { width:44px; text-align:right; font-size:12px; font-weight:800; color:#344054; }
.status-pill {
    display:inline-block; padding:5px 9px; border-radius:999px;
    background:#E8F8EF; color:#16804A; font-size:11px; font-weight:800;
}
.empty-state {
    padding:22px; border:1px dashed #D5DBE8; border-radius:16px;
    text-align:center; color:#667085; background:rgba(255,255,255,.78); line-height:1.7;
}
.footer {
    text-align:center; color:#98A2B3; font-size:11px; padding-top:28px;
}
[data-testid="stFileUploaderDropzone"] {
    background:rgba(255,255,255,.82); border:1.5px dashed #C8D2EA; border-radius:15px;
    padding:18px; transition:all .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color:#4263EB; background:#F5F7FF; }
[data-testid="stFileUploaderDropzone"] button {
    border-radius:9px; border:1px solid #D9E0F0; background:#FFFFFF;
    color:#344054; font-weight:700;
}
[data-testid="stImage"] img { border-radius:14px; }
[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
    align-items: stretch;
}
@media (max-width: 600px) {
    [data-testid="stHorizontalBlock"] { gap: .8rem; }
}
[data-testid="stExpander"] {
    border:1px solid #E7EAF2; border-radius:12px; background:#FFFFFF;
}
.stButton button, .stDownloadButton button {
    border-radius:11px; font-weight:700; min-height:42px;
}
div[data-testid="stProgress"] > div > div { background-color:#4263EB; }
@media (max-width: 900px) {
    .block-container { padding: 1rem 1rem 2rem; }
    .hero { padding: 24px; border-radius: 19px; }
    .hero h1 { font-size: 30px; }
    .hero:after { right: 10px; top: 20px; font-size: 62px; }
    .panel { padding: 17px; }
    .metric-card { padding: 14px; }
}
@media (max-width: 600px) {
    .block-container { padding: .75rem .75rem 1.5rem; }
    .hero { padding: 20px 18px; border-radius: 16px; }
    .hero h1 { font-size: 26px; letter-spacing: -.7px; }
    .hero p { font-size: 13px; max-width: 100%; }
    .hero:after { right: 7px; top: 12px; font-size: 46px; }
    .hero-kicker { font-size: 9px; letter-spacing: .4px; }
    .panel { padding: 15px; border-radius: 16px; }
    .metric-card { min-height: 88px; padding: 12px; }
    .metric-value { font-size: 18px; }
    .metric-label { font-size: 11px; }
    .sidebar-brand { padding-bottom: 16px; }
    [data-testid="stFileUploaderDropzone"] { padding: 12px; }
    .footer { line-height: 1.6; padding: 20px 8px 0; }
    .stApp::before { font-size: 82px; top: 12%; right: -8px; }
    .stApp::after { font-size: 64px; bottom: 4%; left: 2%; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="brand-icon">🐾</div>
          <div>
            <div class="brand-name">PawScan</div>
            <div class="brand-sub">AI Animal Recognition</div>
          </div>
        </div>
        <div class="sidebar-label">Navigasi</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigasi aplikasi",
        ["Dashboard", "Tentang Model"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="sidebar-label">Pengaturan prediksi</div>', unsafe_allow_html=True)
    conf = st.slider("Ambang keyakinan", 0.05, 0.95, 0.25, 0.05)
    st.caption("Nilai lebih rendah dapat membantu mendeteksi objek yang kurang jelas.")
    st.markdown(
        """
        <div class="sidebar-note">
          <strong>Tips foto</strong><br>
          Gunakan foto yang jelas, pencahayaan cukup, dan pastikan hewan terlihat di dalam gambar.
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

def render_confidence_rows(items):
    rows = []
    for name, confidence in items:
        percent = confidence * 100
        rows.append(
            f'<div class="conf-row"><span class="conf-name">{name}</span>'
            f'<span class="conf-track"><span class="conf-fill" '
            f'style="width:{percent:.1f}%;display:block"></span></span>'
            f'<span class="conf-pct">{percent:.1f}%</span></div>'
        )
    return "".join(rows)

if page == "Tentang Model":
    st.markdown(
        """
        <div class="hero">
          <span class="hero-kicker">INFORMASI APLIKASI</span>
          <h1>Mengenal model PawScan</h1>
          <p>Aplikasi ini menggunakan model YOLO untuk mendeteksi kucing dan anjing dari foto yang diunggah.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown(
            '<div class="panel"><div class="panel-title">Cara kerja</div>'
            '<div class="panel-subtitle">Alur prediksi gambar</div>'
            '<ol><li>Unggah foto JPG, JPEG, atau PNG.</li>'
            '<li>Model memproses gambar dan mencari objek.</li>'
            '<li>Hasil menampilkan objek terdeteksi dan tingkat keyakinannya.</li></ol></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            '<div class="panel"><div class="panel-title">Informasi teknis</div>'
            '<div class="panel-subtitle">Komponen yang digunakan</div>'
            '<p><b>Framework antarmuka:</b> Streamlit</p>'
            '<p><b>Model deteksi:</b> YOLO (Ultralytics)</p>'
            '<p><b>Label yang ditampilkan:</b> Kucing dan Anjing</p>'
            '<p><b>Format input:</b> JPG, JPEG, PNG</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="footer">PawScan · AI Animal Recognition</div>', unsafe_allow_html=True)
    st.stop()

st.markdown(
    """
    <div class="hero">
      <span class="hero-kicker">DETEKSI HEWAN</span>
      <h1>Kenali hewan dalam<br>sekali unggah.</h1>
      <p>Unggah foto kucing atau anjing untuk mendapatkan hasil deteksi, kotak objek, dan tingkat keyakinan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

try:
    model = load_model()
    class_ids = [i for i, name in model.names.items() if str(name).lower() in LABELS]
    model_status = "Siap digunakan"
except Exception as exc:
    model = None
    class_ids = []
    model_status = "Model belum tersedia"
    model_error = str(exc)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Status model</div>'
        f'<div class="metric-value" style="font-size:18px">{model_status}</div>'
        '<div class="metric-hint">Model YOLO lokal</div></div>',
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
        '<div class="metric-card"><div class="metric-label">Ambang keyakinan</div>'
        f'<div class="metric-value">{conf:.0%}</div>'
        '<div class="metric-hint">Dapat diatur dari sidebar</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
left, right = st.columns([0.95, 1.15], gap="large")

with left:
    st.markdown(
        '<div class="panel"><div class="panel-title">Unggah gambar</div>'
        '<div class="panel-subtitle">Pilih foto yang ingin dianalisis oleh model.</div></div>',
        unsafe_allow_html=True,
    )
    file = st.file_uploader(
        "Pilih gambar",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="drop-hint">Format file: JPG, JPEG, atau PNG<br>Gunakan foto yang jelas agar hasil deteksi lebih baik.</div>',
        unsafe_allow_html=True,
    )
    if file is not None:
        img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

with right:
    st.markdown(
        '<div class="panel"><div class="panel-title">Hasil analisis</div>'
        '<div class="panel-subtitle">Hasil prediksi akan muncul di area ini.</div></div>',
        unsafe_allow_html=True,
    )
    if model is None:
        st.error("Model belum berhasil dimuat. Pastikan file model tersedia di lokasi `model/Haikal_Laporan_4.pt`.")
        with st.expander("Detail teknis"):
            st.code(model_error)
    elif file is None:
        st.markdown(
            '<div class="empty-state"><div style="font-size:34px">🔎</div>'
            '<b>Menunggu gambar</b><br>Unggah foto di panel sebelah kiri untuk memulai prediksi.</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Model sedang menganalisis gambar..."):
            result = model.predict(
                img,
                imgsz=640,
                conf=conf,
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
            best[name] = max(best.get(name, 0.0), float(box.conf))
            counts[name] = counts.get(name, 0) + 1

        if not best:
            st.markdown(
                '<div class="empty-state"><div style="font-size:34px">🧐</div>'
                '<b>Belum ada hewan terdeteksi</b><br>'
                'Coba foto lain dengan objek lebih jelas atau turunkan ambang keyakinan di sidebar.</div>',
                unsafe_allow_html=True,
            )
            st.image(annotated, caption="Hasil pemrosesan model", use_container_width=True)
        else:
            ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
            top_name = " dan ".join(name for name, _ in ranked)
            total_objects = sum(counts.values())
            st.markdown(
                f'<div class="result-banner"><div class="result-label">Deteksi berhasil</div>'
                f'<div class="result-answer">{top_name}</div>'
                f'<div style="color:#5265A8;font-size:12px;margin-top:5px">'
                f'{total_objects} objek terdeteksi dalam gambar</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(render_confidence_rows(ranked), unsafe_allow_html=True)
            st.image(annotated, caption="Gambar dengan kotak deteksi", use_container_width=True)
            with st.expander("Rincian jumlah objek"):
                for name in LABELS.values():
                    st.write(f"**{name}:** {counts.get(name, 0)} objek")
            image_bytes = Image.fromarray(annotated).convert("RGB")
            import io
            buffer = io.BytesIO()
            image_bytes.save(buffer, format="PNG")
            st.download_button(
                "Unduh gambar hasil deteksi",
                data=buffer.getvalue(),
                file_name="hasil_deteksi_pawscan.png",
                mime="image/png",
                use_container_width=True,
            )

st.markdown(
    '<div class="footer">PawScan · Dibangun dengan Streamlit dan YOLO · Hasil bergantung pada model dan kualitas gambar.</div>',
    unsafe_allow_html=True,
)
