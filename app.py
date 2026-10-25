import base64
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="PawScan | Animal Recognition", page_icon="🐾", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).resolve().parent
# Letakkan model H5 di folder yang sama dengan app.py.
CLASSIFIER_PATH = ROOT / "best_model.h5"
# Letakkan model YOLO di folder model/ dengan nama Haikal_Laporan_4.pt.
YOLO_PATH = ROOT / "model" / "Haikal_Laporan 4.pt"
IMAGE_SIZE = (224, 224)

# Label CNN dibuat tetap (tanpa input nama kelas). Urutannya diasumsikan mengikuti
# urutan kelas training yang umum: Kucing, lalu Anjing. Tukar urutan di bawah jika
# folder training model Anda menggunakan urutan berbeda.
CNN_CLASS_NAMES = ["Kucing", "Anjing"]

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family:'Manrope',sans-serif; }
.stApp { background:radial-gradient(circle at 8% 10%,#f8d5f2 0,transparent 28%),radial-gradient(circle at 92% 8%,#d9e4ff 0,transparent 30%),linear-gradient(135deg,#f5e2ff,#e6edff 55%,#f8e6fb); color:#20243a; }
[data-testid="stHeader"] {background:transparent;} #MainMenu, footer {visibility:hidden;}
.block-container {max-width:1240px;padding:1.5rem 2rem 2.5rem;}
[data-testid="stSidebar"] {background:#10182b;} [data-testid="stSidebar"] * {color:#e8eeff;}
.brand {display:flex;align-items:center;gap:12px;margin:5px 0 20px;}
.brand-icon {background:linear-gradient(135deg,#f15ec9,#9875ff);border-radius:15px;padding:10px;font-size:25px;}
.brand-name {font-size:23px;font-weight:800;color:#35245f;letter-spacing:-.6px;}
.brand-sub {font-size:11px;color:#826aa7;}
.hero {position:relative;overflow:hidden;padding:35px;border:1px solid rgba(255,255,255,.75);border-radius:26px;background:linear-gradient(120deg,rgba(255,255,255,.78),rgba(255,255,255,.45));box-shadow:0 16px 45px rgba(105,76,164,.10);}
.hero h1 {font-size:clamp(30px,4vw,48px);line-height:1.14;font-weight:800;letter-spacing:-1.5px;color:#432768;margin:14px 0 12px;}
.hero p {font-size:14px;line-height:1.9;color:#78628f;max-width:760px;}
.kicker {display:inline-block;border:1px solid #f4b6e8;background:#fff8ff;color:#8654aa;border-radius:99px;padding:7px 12px;font-size:10px;font-weight:800;letter-spacing:1px;}
.hero-deco {position:absolute;right:28px;top:15px;font-size:75px;opacity:.22;}
.panel {background:rgba(255,255,255,.76);border:1px solid rgba(255,255,255,.95);border-radius:21px;padding:22px;box-shadow:0 10px 28px rgba(92,70,140,.07);}
.panel-title {font-size:18px;font-weight:800;color:#432768;margin-bottom:5px;}
.panel-sub {font-size:12px;line-height:1.8;color:#866e9e;}
.metric {background:rgba(255,255,255,.7);border:1px solid rgba(255,255,255,.9);border-radius:17px;padding:17px;min-height:96px;}
.metric-label {font-size:11px;color:#927aa8;margin-bottom:8px;}.metric-value {font-size:19px;font-weight:800;color:#432768;}
.step-pill {background:rgba(255,255,255,.65);border:1px solid rgba(255,255,255,.95);padding:12px 15px;border-radius:15px;text-align:center;color:#856b9e;font-size:12px;font-weight:800;}
.step-active {background:linear-gradient(110deg,#f263cb,#a477ff);color:white;box-shadow:0 8px 20px rgba(192,93,211,.22);border-color:transparent;}
.stButton>button,.stDownloadButton>button {border:0;border-radius:13px;min-height:45px;font-weight:800;color:white;background:linear-gradient(105deg,#f263c7,#a477ff);box-shadow:0 7px 18px rgba(189,92,208,.19);}
.stButton>button:hover,.stDownloadButton>button:hover {border:0;color:white;filter:brightness(1.04);}
.stTextInput input {border:1px solid #eadcf6;border-radius:12px;background:rgba(255,255,255,.9);}
[data-testid="stFileUploaderDropzone"] {background:rgba(255,255,255,.78);border:1.5px dashed #d6b9ed;border-radius:16px;}
[data-testid="stImage"] img {border-radius:16px;}
.result-banner {padding:20px;border-radius:18px;background:linear-gradient(120deg,#fff5fc,#f0edff);border:1px solid #efd7fa;margin:12px 0 18px;}
.result-label {font-size:10px;font-weight:800;letter-spacing:1.3px;color:#a15ab3;}.result-answer {font-size:30px;font-weight:800;color:#482b70;margin-top:5px;}
.footer {text-align:center;color:#9874b2;font-size:11px;line-height:1.9;padding:28px 0 8px;}
.note {border-radius:14px;padding:14px 16px;background:rgba(255,255,255,.6);border:1px solid rgba(255,255,255,.9);font-size:12px;color:#806a96;line-height:1.8;}
div[data-testid="stRadio"] > label {font-weight:800;color:#4d326d;}
@media(max-width:700px){.block-container{padding:.8rem .8rem 1.5rem}.hero{padding:23px 18px}.hero h1{font-size:30px}.hero-deco{font-size:42px;right:12px}.panel{padding:16px}}
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {"step": 1, "full_name": "", "campus": "", "mode": "CNN — Klasifikasi", "upload_bytes": None, "upload_name": None, "analysis": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

@st.cache_resource
def load_classifier(path):
    from tensorflow.keras.models import load_model
    return load_model(path, compile=False)

@st.cache_resource
def load_yolo(path):
    from ultralytics import YOLO
    return YOLO(path)


def image_to_rgb(source):
    return ImageOps.exif_transpose(Image.open(source)).convert("RGB")


def run_cnn(image):
    model = load_classifier(str(CLASSIFIER_PATH))
    arr = np.asarray(image.resize(IMAGE_SIZE), dtype=np.float32) / 255.0
    pred = np.asarray(model.predict(np.expand_dims(arr, axis=0), verbose=0)).reshape(-1)
    if pred.size != 2:
        raise ValueError(f"Model H5 menghasilkan {pred.size} nilai keluaran, padahal aplikasi mengharapkan 2 kelas (kucing dan anjing).")
    # Softmax output biasanya berjumlah 1; sigmoid tunggal tidak didukung model dua-output ini.
    if np.any(pred < 0) or np.any(pred > 1) or not np.isclose(float(pred.sum()), 1.0, atol=0.08):
        exp = np.exp(pred - np.max(pred))
        pred = exp / exp.sum()
    ranking = sorted([(CNN_CLASS_NAMES[i], float(pred[i])) for i in range(2)], key=lambda x: x[1], reverse=True)
    return {"method": "CNN", "label": ranking[0][0], "confidence": ranking[0][1], "scores": ranking, "summary": f"Model CNN paling kuat memprediksi gambar ini sebagai {ranking[0][0]} dengan skor {ranking[0][1]:.1%}."}


def run_yolo(image):
    model = load_yolo(str(YOLO_PATH))
    result = model.predict(image, imgsz=640, conf=st.session_state.get("yolo_conf", 0.25), verbose=False)[0]
    names = result.names
    aliases = {"cat": "Kucing", "cats": "Kucing", "dog": "Anjing", "dogs": "Anjing", "kucing": "Kucing", "anjing": "Anjing"}
    counts, best, detections = {}, {}, []
    for box in result.boxes:
        raw = str(names[int(box.cls)]).strip().lower()
        label = aliases.get(raw, str(names[int(box.cls)]))
        conf = float(box.conf)
        counts[label] = counts.get(label, 0) + 1
        best[label] = max(best.get(label, 0.0), conf)
        detections.append({"label": label, "confidence": conf})
    annotated = result.plot(line_width=3)[:, :, ::-1]
    total = len(detections)
    if total:
        detail = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
        summary = f"YOLO mendeteksi {total} objek. Rinciannya: {detail}."
    else:
        summary = "Belum ada objek yang terdeteksi. Coba foto yang lebih jelas atau sesuaikan ambang keyakinan."
    return {"method": "YOLO", "counts": counts, "best": best, "detections": detections, "annotated": annotated, "summary": summary}


def go(step):
    st.session_state.step = step
    st.rerun()


def footer():
    st.markdown('<div class="footer">🐾 PawScan · Animal Recognition<br>created by <b>Muhammad Haikal Asy\'ary | 2408108010065</b><br>Proyek Deteksi dan Klasifikasi Hewan dengan AI</div>', unsafe_allow_html=True)


def step_header():
    labels = [(1, "01 · Intro"), (2, "02 · Registrasi"), (3, "03 · Analisis"), (4, "04 · Hasil")]
    cols = st.columns(4)
    for col, (number, label) in zip(cols, labels):
        with col:
            active = st.session_state.step == number
            st.markdown(f'<div class="step-pill {"step-active" if active else ""}">{label}</div>', unsafe_allow_html=True)
    st.write("")


with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">🐾</div><div><div style="font-size:22px;font-weight:800;color:white">PawScan</div><div style="font-size:11px;color:#b7c3dd">Animal Recognition</div></div></div>', unsafe_allow_html=True)
    st.caption("ALUR APLIKASI")
    for n, title in [(1,"Intro"),(2,"Registrasi"),(3,"Analisis gambar"),(4,"Hasil & ringkasan")]:
        if st.button(f"{n:02d}  {title}", key=f"nav_{n}", use_container_width=True):
            if n <= 2 or (n >= 3 and st.session_state.full_name.strip()):
                go(n)
            else:
                st.warning("Isi registrasi terlebih dahulu.")
    st.divider()
    st.caption("STATUS MODEL")
    st.write(f"CNN H5: {'✅ tersedia' if CLASSIFIER_PATH.exists() else '⚠️ belum ditemukan'}")
    st.write(f"YOLO PT: {'✅ tersedia' if YOLO_PATH.exists() else '⚠️ belum ditemukan'}")
    st.caption("CNN memakai label tetap: Kucing dan Anjing.")
    st.slider("Ambang keyakinan YOLO", 0.05, 0.95, 0.25, 0.05, key="yolo_conf")

step_header()
step = st.session_state.step

if step == 1:
    st.markdown('''<div class="hero"><div class="hero-deco">🐾</div><span class="kicker">PAWSCAN · ANIMAL RECOGNITION</span><h1>Kenali Anjing & Kucing<br>dengan <span style="background:linear-gradient(90deg,#f263c7,#9875ff);-webkit-background-clip:text;color:transparent">Kekuatan AI</span></h1><p>Selamat datang di PawScan! Unggah foto hewan peliharaan, pilih metode CNN atau YOLO, lalu lihat hasil analisis beserta ringkasannya dalam beberapa langkah mudah.</p></div>''', unsafe_allow_html=True)
    st.write("")
    a,b,c = st.columns(3)
    for col, emoji, title, desc in [(a,"🐱","CNN · Klasifikasi","Memprediksi kelas utama: kucing atau anjing."),(b,"🎯","YOLO · Deteksi","Menandai lokasi hewan dan menghitung objek."),(c,"📊","Ringkasan hasil","Lihat skor, jumlah deteksi, dan rangkuman analisis.")]:
        with col:
            st.markdown(f'<div class="panel" style="min-height:165px"><div style="font-size:30px">{emoji}</div><div class="panel-title">{title}</div><div class="panel-sub">{desc}</div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="note">✨ <b>Cara kerja:</b> perkenalan → isi identitas → unggah gambar dan pilih metode → lihat hasil. CNN langsung menggunakan label Kucing dan Anjing, tanpa pengaturan nama kelas manual.</div>', unsafe_allow_html=True)
    st.write("")
    if st.button("🐾 Mulai Sekarang", use_container_width=True): go(2)

elif step == 2:
    st.markdown('<div class="hero"><div class="hero-deco">👋</div><span class="kicker">SESI 02 · REGISTRASI</span><h1>Halo, Teman PawScan!</h1><p>Kenalan dulu sebelum mulai menganalisis foto. Informasi ini digunakan untuk menyapa Anda di halaman hasil.</p></div>', unsafe_allow_html=True)
    st.write("")
    left,right = st.columns([1,1], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">📝 Data pengguna</div><div class="panel-sub">Isi nama dan institusi Anda untuk melanjutkan.</div>', unsafe_allow_html=True)
        name = st.text_input("Nama lengkap", value=st.session_state.full_name, placeholder="Contoh: Muhammad Haikal Asy'ary", key="registration_name")
        campus = st.text_input("Nama kampus", value=st.session_state.campus, placeholder="Nama universitas atau institusi", key="registration_campus")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div style="font-size:36px">🐶 🐱</div><div class="panel-title">Siap memulai?</div><div class="panel-sub">Di sesi berikutnya, pilih CNN untuk klasifikasi kucing/anjing atau YOLO untuk deteksi objek. Pastikan foto cukup terang dan objek terlihat jelas.</div></div>', unsafe_allow_html=True)
    st.write("")
    prev,nxt = st.columns(2)
    with prev:
        if st.button("← Kembali ke Intro", use_container_width=True): go(1)
    with nxt:
        if st.button("Simpan & Lanjutkan →", use_container_width=True):
            if not name.strip() or not campus.strip(): st.warning("Nama lengkap dan nama kampus wajib diisi.")
            else:
                st.session_state.full_name = name.strip(); st.session_state.campus = campus.strip(); go(3)

elif step == 3:
    st.markdown('<div class="hero"><div class="hero-deco">🔎</div><span class="kicker">SESI 03 · ANALISIS GAMBAR</span><h1>Unggah foto hewan</h1><p>Pilih metode analisis yang sesuai, lalu unggah gambar JPG, JPEG, atau PNG. Hasil akan ditampilkan pada sesi terakhir.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<div class="note">👤 Pengguna: <b>{st.session_state.full_name}</b> · {st.session_state.campus}</div>', unsafe_allow_html=True)
    st.write("")
    mode = st.radio("Pilih metode analisis", ["CNN — Klasifikasi", "YOLO — Deteksi objek"], horizontal=True, index=0 if st.session_state.mode.startswith("CNN") else 1)
    st.session_state.mode = mode
    if mode.startswith("CNN"):
        st.markdown('<div class="panel"><div class="panel-title">🧠 CNN · Klasifikasi gambar</div><div class="panel-sub">Model H5 memprediksi satu kelas utama: <b>Kucing</b> atau <b>Anjing</b>. Tidak perlu mengisi nama kelas secara manual.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel"><div class="panel-title">🎯 YOLO · Deteksi objek</div><div class="panel-sub">Model YOLO menandai objek pada gambar dengan bounding box dan skor confidence. File model .pt harus tersedia di folder model/.</div></div>', unsafe_allow_html=True)
    st.write("")
    uploaded = st.file_uploader("Pilih foto hewan", type=["jpg","jpeg","png"], key="analysis_uploader")
    if uploaded is not None:
        try:
            preview = image_to_rgb(uploaded)
            st.image(preview, caption="Pratinjau gambar", use_container_width=True)
            st.session_state.upload_bytes = uploaded.getvalue()
            st.session_state.upload_name = uploaded.name
        except Exception:
            st.error("File tidak dapat dibaca sebagai gambar. Silakan pilih file lain.")
    elif st.session_state.upload_bytes:
        st.caption(f"Gambar tersimpan: {st.session_state.upload_name}")
    st.write("")
    back, analyze = st.columns(2)
    with back:
        if st.button("← Kembali ke Registrasi", use_container_width=True): go(2)
    with analyze:
        if st.button("✨ Analisis Gambar", use_container_width=True):
            image_bytes = st.session_state.upload_bytes
            if uploaded is not None: image_bytes = uploaded.getvalue()
            if not image_bytes:
                st.warning("Unggah gambar terlebih dahulu.")
            elif mode.startswith("CNN") and not CLASSIFIER_PATH.exists():
                st.error(f"Model CNN tidak ditemukan: {CLASSIFIER_PATH.name}. Letakkan file H5 di folder yang sama dengan app.py.")
            elif mode.startswith("YOLO") and not YOLO_PATH.exists():
                st.error(f"Model YOLO tidak ditemukan: {YOLO_PATH.relative_to(ROOT)}. Letakkan file .pt pada lokasi tersebut.")
            else:
                try:
                    with st.spinner("AI sedang menganalisis gambar..."):
                        source = Image.open(io.BytesIO(image_bytes))
                        image = image_to_rgb(io.BytesIO(image_bytes))
                        if mode.startswith("CNN"):
                            result = run_cnn(image)
                            result["original"] = image_bytes
                        else:
                            result = run_yolo(image)
                            result["original"] = image_bytes
                        result["filename"] = st.session_state.upload_name or "gambar"
                        st.session_state.analysis = result
                    go(4)
                except ImportError as exc:
                    if mode.startswith("CNN"): st.error("TensorFlow belum terpasang. Jalankan: pip install tensorflow")
                    else: st.error("Ultralytics belum terpasang. Jalankan: pip install ultralytics")
                except Exception as exc:
                    st.error("Analisis gagal dijalankan. Periksa file model dan dependensinya.")
                    with st.expander("Detail teknis"): st.code(str(exc))

elif step == 4:
    st.markdown('<div class="hero"><div class="hero-deco">✨</div><span class="kicker">SESI 04 · HASIL & RINGKASAN</span><h1>Hasil analisis PawScan</h1><p>Berikut ringkasan dari model AI berdasarkan gambar yang Anda unggah.</p></div>', unsafe_allow_html=True)
    st.write("")
    result = st.session_state.analysis
    if not result:
        st.markdown('<div class="note">Belum ada hasil analisis. Silakan kembali ke sesi analisis dan jalankan model terlebih dahulu.</div>', unsafe_allow_html=True)
        if st.button("Ke Sesi Analisis", use_container_width=True): go(3)
    else:
        st.markdown(f'<div class="note">👋 Hasil untuk <b>{st.session_state.full_name or "Pengguna PawScan"}</b> · Metode: <b>{result["method"]}</b> · File: <b>{result.get("filename","gambar")}</b></div>', unsafe_allow_html=True)
        st.write("")
        left,right = st.columns([1,1], gap="large")
        with left:
            if result["method"] == "CNN":
                image = image_to_rgb(io.BytesIO(result["original"]))
                st.image(image, caption="Gambar yang diklasifikasikan", use_container_width=True)
            else:
                st.image(result["annotated"], caption="Gambar dengan penanda deteksi YOLO", use_container_width=True)
                output = io.BytesIO(); Image.fromarray(result["annotated"]).save(output, format="PNG")
                st.download_button("⬇️ Unduh gambar hasil deteksi", data=output.getvalue(), file_name="hasil_deteksi_pawscan.png", mime="image/png", use_container_width=True)
        with right:
            if result["method"] == "CNN":
                st.markdown(f'<div class="result-banner"><div class="result-label">PREDIKSI UTAMA</div><div class="result-answer">{result["label"]}</div><div style="color:#8c67a5;font-size:13px;margin-top:5px">Skor model: {result["confidence"]:.1%}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">📊 Skor tiap kelas</div>', unsafe_allow_html=True)
                for label, score in result["scores"]:
                    st.write(f"**{label}** · {score:.1%}")
                    st.progress(max(0.0, min(1.0, score)))
                st.markdown('<div class="panel-sub">Skor menunjukkan keluaran model, bukan jaminan bahwa prediksi selalu benar.</div>', unsafe_allow_html=True)
            else:
                counts = result["counts"]; total = len(result["detections"])
                st.markdown(f'<div class="result-banner"><div class="result-label">TOTAL OBJEK TERDETEKSI</div><div class="result-answer">{total} objek</div><div style="color:#8c67a5;font-size:13px;margin-top:5px">Kategori: {", ".join(counts.keys()) if counts else "Belum ada"}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">📊 Rincian objek</div>', unsafe_allow_html=True)
                for label in ["Kucing", "Anjing"]:
                    st.write(f"**{label}:** {counts.get(label, 0)} objek")
                if result["best"]:
                    st.markdown("**Confidence tertinggi per kategori**")
                    for label, score in sorted(result["best"].items()):
                        st.write(f"{label} · {score:.1%}")
                        st.progress(max(0.0, min(1.0, score)))
        st.write("")
        st.markdown('<div class="panel"><div class="panel-title">📝 Ringkasan hasil</div><div class="panel-sub">' + result["summary"] + '</div></div>', unsafe_allow_html=True)
        st.write("")
        a,b = st.columns(2)
        with a:
            if st.button("🔄 Analisis Gambar Lain", use_container_width=True):
                st.session_state.analysis = None; st.session_state.upload_bytes = None; st.session_state.upload_name = None; go(3)
        with b:
            if st.button("🏠 Kembali ke Beranda", use_container_width=True):
                st.session_state.analysis = None; go(1)

footer()
