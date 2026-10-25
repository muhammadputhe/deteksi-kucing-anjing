import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(
    page_title="PawScan | Animal Recognition",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
YOLO_PATH = ROOT / "model" / "Haikal_Laporan 4.pt"
CLASSIFIER_PATH = ROOT / "best_model.h5"
DESKTOP_BG_PATH = ROOT / "background_pawscan.png"
MOBILE_BG_PATH = ROOT / "background_pawscan_mobile.png"
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
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,.stApp,p,h1,h2,h3,h4,button,input,textarea,label{{font-family:'Manrope',sans-serif;}}
.stApp{{{bg_css}color:#172033;}}
[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"]{{background:transparent!important;}}
#MainMenu,footer{{visibility:hidden;}} header[data-testid="stHeader"]{{background:transparent;}}
.block-container{{max-width:1220px;padding:1.5rem 2rem 2rem;}}
.brand{{display:flex;align-items:center;gap:12px;margin:0 0 18px;}}
.brand-icon{{background:#4263EB;border-radius:14px;padding:10px 12px;font-size:25px;line-height:1.2;box-shadow:0 8px 18px #4263eb35;}}
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
.panel{{background:rgba(255,255,255,.94);border:1px solid #E3E9F5;border-radius:20px;padding:22px;box-shadow:0 8px 26px #18274b0a;}}
.panel-title{{font-size:18px;font-weight:800;color:#172554;margin-bottom:6px;}}
.panel-subtitle{{font-size:12px;color:#758096;line-height:1.75;}}
.metric-card{{background:rgba(255,255,255,.94);border:1px solid #E3E9F5;border-radius:17px;padding:18px;min-height:100px;}}
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
    allowed = {"cats": "Kucing", "cat": "Kucing", "dogs": "Anjing", "dog": "Anjing"}
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

def footer():
    st.markdown('<div class="footer">created by <b>Muhammad Haikal Asy’ary · 2408108010065</b><br>Proyek UTS Pemrograman Big Data | Universitas Syiah Kuala</div>', unsafe_allow_html=True)

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

# Brand + four-stage progress navigation.
st.markdown('<div class="brand"><div class="brand-icon">🐾</div><div><div class="brand-name">PawScan</div><div class="brand-sub">ANIMAL RECOGNITION · CNN + YOLO</div></div></div>', unsafe_allow_html=True)
step = st.session_state.step
labels = ["Intro", "Registrasi", "Analisis", "Hasil"]
progress = "".join(f'<div class="progress-step {"active" if i <= step else ""}"></div>' for i in range(1, 5))
st.markdown(f'<div class="progress-wrap">{progress}</div><div class="step-label">Sesi {step} dari 4 · {labels[step-1]}</div>', unsafe_allow_html=True)

if step == 1:
    st.markdown("""
    <div class="hero"><div class="hero-decoration">🐾</div>
    <span class="hero-kicker">SMART PET RECOGNITION</span>
    <h1>Kenali Anjing & Kucing<br>dengan <span style="color:#AFC2FF">Kekuatan AI</span></h1>
    <p>Selamat datang di PawScan! Unggah foto hewan untuk menganalisis gambar menggunakan CNN atau mendeteksi objek dengan YOLO. Ikuti empat langkah sederhana untuk melihat hasilnya.</p>
    </div>""", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3, gap="medium")
    for col, num, title, desc in [
        (c1, "01", "Isi registrasi", "Masukkan nama dan kampus."),
        (c2, "02", "Unggah gambar", "Pilih foto anjing atau kucing."),
        (c3, "03", "Lihat hasil", "Baca prediksi dan ringkasan AI."),
    ]:
        with col:
            st.markdown(f'<div class="panel"><div class="pill">LANGKAH {num}</div><div class="panel-title" style="margin-top:13px">{title}</div><div class="panel-subtitle">{desc}</div></div>', unsafe_allow_html=True)
    st.write("")
    if st.button("🐶 Mulai Sekarang", type="primary", use_container_width=True):
        go(2)

elif step == 2:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown('<div class="hero"><div class="hero-decoration">🐱</div><span class="hero-kicker">PAWSCAN MEMBER</span><h1>Halo,<br>Teman Baru!</h1><p>Lengkapi identitas singkat sebelum memulai analisis gambar.</p></div>', unsafe_allow_html=True)
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
    st.markdown('<div class="panel"><div class="panel-title">📷 Unggah & Analisis Gambar</div><div class="panel-subtitle">Pilih metode, lalu unggah gambar anjing atau kucing yang jelas. CNN memprediksi satu kelas utama; YOLO dapat mendeteksi beberapa objek.</div></div>', unsafe_allow_html=True)
    st.write("")
    mode = st.radio("Pilih metode analisis", ["Klasifikasi CNN", "Deteksi Objek YOLO"], horizontal=True,
                    index=0 if st.session_state.analysis_mode == "Klasifikasi CNN" else 1)
    st.session_state.analysis_mode = mode
    if mode == "Klasifikasi CNN":
        st.caption("CNN · Model H5 · 224 × 224 RGB · kelas: Anjing dan Kucing")
    else:
        st.caption("YOLO · Model .pt · kotak deteksi dan confidence")
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
            st.markdown('<div class="empty-state"><div style="font-size:36px">📸</div><b>Gambar belum dipilih</b><br>Unggah foto untuk memulai analisis.</div>', unsafe_allow_html=True)
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
                        with st.spinner("CNN sedang menganalisis gambar..."):
                            model = load_classifier(str(CLASSIFIER_PATH))
                            predictions = classifier_predict(image, model)
                        st.session_state.analysis_result = {"mode": mode, "predictions": predictions, "image": image.copy()}
                        go(4)
                else:
                    if not YOLO_PATH.exists():
                        st.error(f"Model YOLO tidak ditemukan: {YOLO_PATH}. Letakkan Haikal_Laporan_4.pt di folder model/.")
                    else:
                        with st.spinner("YOLO sedang mendeteksi objek..."):
                            model = load_yolo(str(YOLO_PATH))
                            annotated, counts, best = yolo_predict(image, model, yolo_conf)
                        st.session_state.analysis_result = {"mode": mode, "annotated": annotated, "counts": counts, "best": best, "image": image.copy()}
                        go(4)
            except ImportError as exc:
                st.error("Dependensi model belum terpasang. CNN memerlukan TensorFlow; YOLO memerlukan ultralytics.")
                with st.expander("Detail teknis"):
                    st.code(str(exc))
            except Exception as exc:
                st.error("Analisis gagal dijalankan. Periksa file model dan dependensi.")
                with st.expander("Detail kesalahan"):
                    st.code(str(exc))
    st.write("")
    if st.button("← Kembali ke Registrasi"):
        go(2)

elif step == 4:
    result = st.session_state.analysis_result
    if not result:
        st.markdown('<div class="empty-state"><div style="font-size:36px">🔎</div><b>Belum ada hasil analisis</b><br>Kembali ke sesi analisis untuk mengunggah gambar.</div>', unsafe_allow_html=True)
        if st.button("Ke sesi analisis", type="primary"):
            go(3)
    else:
        st.markdown(f'<div class="hero"><div class="hero-decoration">✨</div><span class="hero-kicker">ANALISIS SELESAI</span><h1>Hasil PawScan</h1><p>Terima kasih, {st.session_state.full_name}! Berikut ringkasan hasil analisis gambar Anda.</p></div>', unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="metric-card"><div class="metric-label">Pengguna</div><div class="metric-value">{st.session_state.full_name}</div><div class="metric-hint">{st.session_state.campus} · Metode: {result["mode"]}</div></div>', unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([1, 1], gap="large")
        with left:
            if result["mode"] == "Klasifikasi CNN":
                st.image(result["image"], caption="Gambar yang dianalisis", use_container_width=True)
            else:
                st.image(result["annotated"], caption="Gambar dengan kotak deteksi YOLO", use_container_width=True)
                out = io.BytesIO()
                Image.fromarray(result["annotated"]).save(out, format="PNG")
                st.download_button("⬇️ Unduh gambar hasil YOLO", data=out.getvalue(), file_name="hasil_pawscan.png", mime="image/png", use_container_width=True)
        with right:
            if result["mode"] == "Klasifikasi CNN":
                predictions = result["predictions"]
                top_label, top_score = predictions[0]
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
                go(3)
        with c2:
            if st.button("⌂ Kembali ke Beranda", use_container_width=True):
                st.session_state.analysis_result = None
                go(1)

footer()
