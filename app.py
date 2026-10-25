import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

st.set_page_config(page_title="Kucing atau Anjing", page_icon="🐾", layout="centered")

MODEL_PATH = "model/Haikal_Laporan_4.pt"
LABELS = {"cats": "Kucing", "dogs": "Anjing"}  # kelas "Air" sengaja diabaikan

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;800&display=swap');

html, body, [class*="st-"], button, input {
    font-family: 'Manrope', sans-serif;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.block-container { max-width: 640px; padding-top: 4.5rem; padding-bottom: 4rem; }

.title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em;
    line-height: 1.1; color: #1F2430; margin: 0;
}
.subtitle {
    font-size: 1rem; color: #6B7280; margin: 0.75rem 0 2.25rem 0; line-height: 1.6;
}

/* Uploader */
[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF; border: 1.5px dashed #D1D5DB; border-radius: 16px;
    padding: 2rem 1.5rem; transition: border-color .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #3B5BDB; }
[data-testid="stFileUploaderDropzone"] button {
    border-radius: 10px; border: 1px solid #E5E7EB; background: #FFFFFF;
    color: #1F2430; font-weight: 600;
}

/* Gambar */
[data-testid="stImage"] img { border-radius: 14px; }

/* Hasil */
.result { margin: 2rem 0 0.5rem 0; padding-top: 1.5rem; border-top: 1px solid #E5E7EB; }
.result .label { font-size: 0.9rem; color: #6B7280; margin-bottom: 0.25rem; }
.result .answer {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; color: #1F2430;
}
.row { display: flex; align-items: center; gap: 0.9rem; margin-top: 0.9rem; }
.row .name { width: 5.5rem; font-weight: 600; color: #1F2430; }
.row .track { flex: 1; height: 6px; background: #E5E7EB; border-radius: 99px; overflow: hidden; }
.row .fill { height: 100%; background: #3B5BDB; border-radius: 99px; }
.row .pct { width: 3.5rem; text-align: right; color: #6B7280; font-variant-numeric: tabular-nums; }

.empty { margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #E5E7EB; color: #6B7280; line-height: 1.6; }
.empty strong { color: #1F2430; }

[data-testid="stExpander"] { border: none; background: transparent; margin-top: 1.5rem; }
[data-testid="stExpander"] summary { color: #6B7280; font-size: 0.9rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()
class_ids = [i for i, n in model.names.items() if n.lower() in LABELS]

st.markdown('<p class="title">Kucing atau anjing?</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Unggah sebuah foto, model akan mendeteksi hewan di dalamnya.</p>',
    unsafe_allow_html=True,
)

file = st.file_uploader(
    "Unggah foto", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

with st.expander("Pengaturan"):
    conf = st.slider("Ambang keyakinan", 0.05, 0.95, 0.25, 0.05)
    st.caption("Turunkan nilai ini jika hewan di foto tidak terdeteksi.")

if file is not None:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")

    with st.spinner("Mendeteksi..."):
        result = model.predict(img, imgsz=640, conf=conf, classes=class_ids)[0]

    annotated = result.plot(line_width=3)[:, :, ::-1]  # BGR -> RGB
    st.image(annotated, use_container_width=True)

    # Ambil keyakinan tertinggi untuk tiap kelas
    best = {}
    for b in result.boxes:
        name = LABELS[result.names[int(b.cls)].lower()]
        best[name] = max(best.get(name, 0.0), float(b.conf))

    if not best:
        st.markdown(
            '<div class="empty"><strong>Tidak ada kucing atau anjing yang terdeteksi.</strong><br>'
            "Coba foto lain dengan hewan yang lebih jelas, atau turunkan ambang keyakinan di Pengaturan.</div>",
            unsafe_allow_html=True,
        )
    else:
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        top_name = " dan ".join(n for n, _ in ranked) if len(ranked) > 1 else ranked[0][0]

        rows = "".join(
            f'<div class="row"><span class="name">{n}</span>'
            f'<span class="track"><span class="fill" style="width:{c*100:.0f}%;display:block"></span></span>'
            f'<span class="pct">{c*100:.0f}%</span></div>'
            for n, c in ranked
        )
        st.markdown(
            f'<div class="result"><div class="label">Terdeteksi</div>'
            f'<div class="answer">{top_name}</div>{rows}</div>',
            unsafe_allow_html=True,
        )