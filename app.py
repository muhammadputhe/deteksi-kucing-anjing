
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO
import io

st.set_page_config(
    page_title="PawScan | Deteksi Hewan",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model/Haikal_Laporan_4.pt"
LABELS = {"cats": "Kucing", "dogs": "Anjing"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

/* Jangan gunakan selector font [class*="st-"],
   karena dapat merusak ikon bawaan Streamlit. */
html, body, .stApp, p, h1, h2, h3, label,
button, input, textarea {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 5%,
            rgba(210,225,255,.65), transparent 35%),
        radial-gradient(ellipse at 95% 35%,
            rgba(225,235,255,.75), transparent 32%),
        linear-gradient(145deg,#F8FAFF,#EEF4FF,#FAFBFF);
    color: #172033;
}

/* Pastikan semua lapisan aplikasi tetap terlihat */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: transparent !important;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1320px;
    padding: 1.5rem 2rem 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #10182B;
    border-right: 1px solid #26334D;
}

[data-testid="stSidebar"] * {
    color: #E8EEFF;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #B7C3DD;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 25px;
}

.brand-icon {
    background: #4263EB;
    border-radius: 14px;
    padding: 10px;
    font-size: 25px;
}

.brand-name {
    color: white;
    font-size: 21px;
    font-weight: 800;
}

.brand-sub {
    color: #B7C3DD;
    font-size: 11px;
}

.sidebar-label {
    color: #8998B8;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin: 15px 0 8px;
}

/* Banner utama */
.hero {
    position: relative;
    overflow: hidden;
    padding: 30px;
    border-radius: 22px;
    color: white;
    background: linear-gradient(
        120deg, #172554 0%, #2947C7 58%, #4263EB 100%
    );
    box-shadow: 0 15px 38px rgba(41,71,199,.15);
}

.hero h1 {
    color: white;
    font-size: clamp(27px, 3.2vw, 40px);
    line-height: 1.2;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 12px 0;
}

.hero p {
    color: #E0E8FF;
    font-size: 14px;
    line-height: 1.8;
    max-width: 680px;
}

.hero-decoration {
    position: absolute;
    right: 25px;
    top: 20px;
    font-size: 75px;
    opacity: .16;
    pointer-events: none;
}

/* Kartu informasi */
.metric-card {
    background: rgba(255,255,255,.92);
    border: 1px solid #E3E9F5;
    border-radius: 17px;
    padding: 19px;
    min-height: 105px;
    box-shadow: 0 5px 18px rgba(24,39,75,.035);
}

.metric-label {
    color: #758096;
    font-size: 12px;
    margin-bottom: 8px;
}

.metric-value {
    color: #172033;
    font-size: 21px;
    font-weight: 800;
}

.metric-hint {
    color: #8994A8;
    font-size: 11px;
    margin-top: 6px;
}

/* Panel */
.panel {
    background: rgba(255,255,255,.94);
    border: 1px solid #E3E9F5;
    border-radius: 19px;
    padding: 21px;
    min-height: 108px;
    box-shadow: 0 5px 18px rgba(24,39,75,.03);
}

.panel-title {
    font-size: 17px;
    font-weight: 800;
    color: #172033;
    margin-bottom: 5px;
}

.panel-subtitle {
    font-size: 12px;
    color: #758096;
    line-height: 1.7;
}

/* Upload */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.88);
    border: 1.5px dashed #B9C9F5;
    border-radius: 15px;
    padding: 16px;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #4263EB;
    background: #F5F7FF;
}

[data-testid="stFileUploaderDropzone"] button {
    background: white;
    color: #2947C7;
    border: 1px solid #DCE3F2;
    border-radius: 9px;
    font-weight: 700;
}

[data-testid="stImage"] img {
    border-radius: 14px;
}

/* Hasil prediksi */
.result-banner {
    padding: 18px;
    border-radius: 15px;
    background: #EDF2FF;
    border: 1px solid #D8E2FF;
    margin: 12px 0 18px;
}

.result-label {
    color: #5265A8;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.result-answer {
    color: #203A9B;
    font-size: 28px;
    font-weight: 800;
    margin-top: 4px;
}

.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 15px 0;
}

.conf-name {
    width: 68px;
    color: #344054;
    font-size: 13px;
    font-weight: 700;
}

.conf-track {
    flex: 1;
    height: 8px;
    border-radius: 99px;
    background: #E7EBF4;
    overflow: hidden;
}

.conf-fill {
    height: 100%;
    background: linear-gradient(90deg,#4263EB,#7192FF);
    border-radius: 99px;
}

.conf-pct {
    width: 48px;
    text-align: right;
    color: #344054;
    font-size: 12px;
    font-weight: 800;
}

.empty-state {
    text-align: center;
    padding: 25px 15px;
    border: 1px dashed #CFD8EB;
    border-radius: 16px;
    background: rgba(255,255,255,.8);
    color: #667085;
    line-height: 1.8;
}

.tip {
    border: 1px solid #2A3957;
    border-radius: 13px;
    background: #17223A;
    color: #B7C3DD;
    padding: 14px;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 24px;
}

.tip strong {
    color: white;
}

.stButton button,
.stDownloadButton button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}

.footer {
    text-align: center;
    color: #8994A8;
    font-size: 11px;
    padding-top: 25px;
}

@media (max-width: 900px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }
    .hero {
        padding: 24px;
    }
    .panel {
        padding: 17px;
    }
}

@media (max-width: 600px) {
    .block-container {
        padding: .7rem .7rem 1.5rem;
    }
    .hero {
        padding: 20px 17px;
        border-radius: 16px;
    }
    .hero h1 {
        font-size: 27px;
        letter-spacing: -.6px;
    }
    .hero p {
        font-size: 13px;
    }
    .hero-decoration {
        font-size: 42px;
        right: 8px;
        top: 8px;
    }
    .metric-card {
        padding: 12px;
        min-height: 88px;
    }
    .metric-value {
        font-size: 17px;
    }
    .metric-label {
        font-size: 11px;
    }
    .panel {
        padding: 15px;
    }
    [data-testid="stFileUploaderDropzone"] {
        padding: 10px;
    }
    .footer {
        line-height: 1.7;
    }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def confidence_rows(items):
    html = ""

    for name, confidence in items:
        pct = confidence * 100

        html += (
            f'<div class="conf-row">'
            f'<span class="conf-name">{name}</span>'
            f'<span class="conf-track">'
            f'<span class="conf-fill" '
            f'style="width:{pct:.1f}%;display:block"></span>'
            f'</span>'
            f'<span class="conf-pct">{pct:.1f}%</span>'
            f'</div>'
        )

    return html


# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🐾</div>
            <div>
                <div class="brand-name">PawScan</div>
                <div class="brand-sub">Animal Recognition</div>
            </div>
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
        "Ambang keyakinan",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )

    st.caption(
        "Turunkan nilai ini jika objek di foto belum terdeteksi."
    )

    st.markdown(
        """
        <div class="tip">
            <strong>Tips foto</strong><br>
            Gunakan foto yang jelas dengan pencahayaan cukup.
            Pastikan hewan terlihat di dalam gambar.
        </div>
        """,
        unsafe_allow_html=True,
    )


if page == "Tentang Model":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-decoration">🐾</div>
            <h1>Tentang PawScan</h1>
            <p>
                Aplikasi untuk mendeteksi kucing dan anjing
                dari gambar menggunakan model YOLO.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Cara penggunaan</div>
                <div class="panel-subtitle">
                    1. Unggah gambar kucing atau anjing.<br>
                    2. Model menganalisis gambar.<br>
                    3. Lihat hasil deteksi dan keyakinannya.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Informasi teknis</div>
                <div class="panel-subtitle">
                    Antarmuka: Streamlit<br>
                    Model: YOLO Ultralytics<br>
                    Kelas: Kucing dan Anjing<br>
                    Format: JPG, JPEG, PNG
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footer">PawScan · Deteksi Kucing dan Anjing</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# Dashboard utama
st.markdown(
    """
    <div class="hero">
        <div class="hero-decoration">🐾</div>
        <h1>Kenali hewan dalam<br>sekali unggah.</h1>
        <p>
            Unggah foto kucing atau anjing untuk melihat hasil deteksi,
            kotak objek, dan tingkat keyakinan model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# Memuat model
model = None
model_error = None

try:
    model = load_model()
    class_ids = [
        i for i, name in model.names.items()
        if str(name).lower() in LABELS
    ]
except Exception as exc:
    model_error = str(exc)
    class_ids = []

# Kartu informasi
m1, m2, m3 = st.columns(3)

with m1:
    status = "Siap digunakan" if model else "Belum tersedia"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Status model</div>
            <div class="metric-value" style="font-size:18px">
                {status}
            </div>
            <div class="metric-hint">Model YOLO lokal</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Kategori</div>
            <div class="metric-value">2 kelas</div>
            <div class="metric-hint">Kucing dan anjing</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Ambang keyakinan</div>
            <div class="metric-value">{conf:.0%}</div>
            <div class="metric-hint">Dapat diatur dari sidebar</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

left, right = st.columns([0.95, 1.05], gap="large")

with left:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Unggah gambar</div>
            <div class="panel-subtitle">
                Pilih foto yang ingin dianalisis oleh model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Pilih foto",
        type=["jpg", "jpeg", "png"],
        help="Format yang didukung: JPG, JPEG, PNG.",
    )

    if uploaded_file is not None:
        try:
            image = ImageOps.exif_transpose(
                Image.open(uploaded_file)
            ).convert("RGB")

            st.image(
                image,
                caption="Foto yang diunggah",
                use_container_width=True,
            )
        except Exception:
            image = None
            st.error("Foto tidak dapat dibuka. Coba gunakan gambar lain.")
    else:
        image = None
        st.markdown(
            """
            <div class="empty-state">
                <div style="font-size:32px">📷</div>
                <strong>Belum ada gambar</strong><br>
                Pilih foto untuk memulai prediksi.
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Hasil analisis</div>
            <div class="panel-subtitle">
                Hasil prediksi akan ditampilkan di area ini.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "Model belum berhasil dimuat. Pastikan file "
            "`model/Haikal_Laporan_4.pt` tersedia."
        )

        with st.expander("Detail kesalahan"):
            st.code(model_error or "Kesalahan tidak diketahui.")

    elif image is None:
        st.markdown(
            """
            <div class="empty-state">
                <div style="font-size:32px">🔎</div>
                <strong>Menunggu gambar</strong><br>
                Unggah foto di panel sebelah kiri untuk memulai prediksi.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        with st.spinner("Sedang menganalisis gambar..."):
            result = model.predict(
                image,
                imgsz=640,
                conf=conf,
                classes=class_ids if class_ids else None,
                verbose=False,
            )[0]

        annotated = result.plot(line_width=3)[:, :, ::-1]

        best = {}
        counts = {}

        for box in result.boxes:
            raw_name = str(
                result.names[int(box.cls)]
            ).lower()

            if raw_name not in LABELS:
                continue

            name = LABELS[raw_name]
            score = float(box.conf)

            best[name] = max(best.get(name, 0.0), score)
            counts[name] = counts.get(name, 0) + 1

        if not best:
            st.markdown(
                """
                <div class="empty-state">
                    <div style="font-size:32px">🔍</div>
                    <strong>Hewan belum terdeteksi</strong><br>
                    Coba foto lain atau turunkan ambang keyakinan.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.image(
                annotated,
                caption="Hasil pemrosesan gambar",
                use_container_width=True,
            )

        else:
            ranked = sorted(
                best.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            top_name = " dan ".join(
                name for name, _ in ranked
            )
            total_objects = sum(counts.values())

            st.markdown(
                f"""
                <div class="result-banner">
                    <div class="result-label">Deteksi berhasil</div>
                    <div class="result-answer">{top_name}</div>
                    <div style="color:#5265A8;font-size:12px;margin-top:5px">
                        {total_objects} objek terdeteksi
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                confidence_rows(ranked),
                unsafe_allow_html=True,
            )

            st.image(
                annotated,
                caption="Gambar dengan kotak deteksi",
                use_container_width=True,
            )

            with st.expander("Rincian jumlah objek"):
                for name in LABELS.values():
                    st.write(
                        f"**{name}:** {counts.get(name, 0)} objek"
                    )

            buffer = io.BytesIO()
            Image.fromarray(annotated).save(
                buffer,
                format="PNG",
            )

            st.download_button(
                "Unduh gambar hasil deteksi",
                data=buffer.getvalue(),
                file_name="hasil_deteksi_pawscan.png",
                mime="image/png",
                use_container_width=True,
            )

st.markdown(
    """
    <div class="footer">
        PawScan · Deteksi Kucing dan Anjing<br>
        Hasil bergantung pada model dan kualitas gambar.
    </div>
    """,
    unsafe_allow_html=True,
)
