import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="centered"
)

# Nama berkas untuk dataset dan penyimpanan model
DATASET_FILE = 'SMSSPamCollection'
MODEL_FILE = 'model.pkl'
VECTORIZER_FILE = 'vectorizer.pkl'

# --- TAHAP 1: PELATIHAN MODEL (MANUAL DARI FILE LOKAL) ---
def latih_dan_simpan_model(force_retrain=False):
    # Memastikan file dataset asli diletakkan secara manual di folder
    if not os.path.exists(DATASET_FILE):
        st.error(f"❌ Berkas dataset '{DATASET_FILE}' tidak ditemukan di folder lokal Anda!")
        st.info("👉 Silakan letakkan file **'SMSSPamCollection'** di folder yang sama dengan file **'spam.py'** Anda.")
        st.stop()  # Menghentikan eksekusi Streamlit agar tidak error lebih lanjut

    with st.spinner('🔄 Sedang melatih model menggunakan dataset asli (5.500+ data)...'):
        # Membaca dataset asli (TSV - Tab Separated Values)
        df = pd.read_csv(DATASET_FILE, sep='\t', names=['label', 'message'])

        # Ekstraksi fitur teks menggunakan TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df['message'])
        y = df['label']

        # Klasifikasi menggunakan Naive Bayes (MultinomialNB) sesuai modul kuliah
        model = MultinomialNB()
        model.fit(X, y)

        # Menyimpan model dan vectorizer sebagai berkas pickle (.pkl)
        joblib.dump(model, MODEL_FILE)
        joblib.dump(vectorizer, VECTORIZER_FILE)
        
        if force_retrain:
            st.toast("🎉 Model berhasil dilatih ulang menggunakan dataset asli!", icon="🔥")
        else:
            st.success("🎉 Model berhasil dilatih secara lokal menggunakan dataset asli!")

# Lakukan proses pelatihan jika berkas model (.pkl) belum terbentuk
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    latih_dan_simpan_model()

# Memuat kembali model dan vectorizer yang telah tersimpan
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)


# --- TAHAP 2: TAMPILAN SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/561/561127.png", width=80)
    st.title("SpamGuard AI")
    st.write("Aplikasi pintar pendeteksi spam berbasis Machine Learning.")
    st.divider()
    st.markdown("### ⚙️ Spesifikasi Sistem:")
    st.write("🤖 **Algoritma:** Naive Bayes")
    st.write("📊 **Dataset:** SMS Spam Collection")
    
    # Tombol untuk paksa latih ulang jika ada perubahan dataset
    st.divider()
    if st.button("🔄 Paksa Latih Ulang Model", use_container_width=True):
        latih_dan_simpan_model(force_retrain=True)
        st.rerun()
        
    st.divider()
    st.caption("Tugas Praktikum Deployment • Sesi 14")


# --- TAHAP 3: ANTARMUKA UTAMA (STREAMLIT UI PREMIUM) ---
st.markdown("""
    <div style="background: linear-gradient(to right, #1E3A8A, #9333EA);
                padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px;">
        <h1 style="color: white; margin: 0;">🛡️ SpamGuard AI</h1>
        <p style="color: #FACC15; font-size:18px; margin: 5px 0 0 0;">Proteksi pesan cerdas berbasis AI</p>
    </div>
""", unsafe_allow_html=True)

# Area Input Pesan
st.subheader("📝 Masukkan Pesan SMS")
input_sms = st.text_area(
    label="Pesan SMS yang akan dianalisis:",
    placeholder="Ketik atau tempelkan pesan SMS di sini...",
    height=150,
    label_visibility="collapsed"  # Menyembunyikan label duplikat agar rapi
)

# Tombol interaksi analisis
st.write("")
predict_btn = st.button("🔍 Jalankan Analisis", type="primary", use_container_width=True)

# --- RIWAYAT (SESSION STATE) ---
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- PROSES DETEKSI & ANIMASI ---
if predict_btn:
    if input_sms.strip() == "":
        st.warning("⚠️ Silakan ketikkan pesan SMS terlebih dahulu sebelum mendeteksi!")
    else:
        # Efek loading mulus
        with st.spinner('Menganalisis pola kata pada teks...'):
            time.sleep(0.6)  # Jeda loading singkat
            
            # Ekstraksi dan prediksi
            transformed = vectorizer.transform([input_sms])
            prediction = model.predict(transformed)[0]
            prob = model.predict_proba(transformed)[0]
            confidence = prob[1] if prediction == 'spam' else prob[0]

            st.write("")
            st.markdown("### 📊 Hasil Analisis AI:")
            
            # Logika visualisasi hasil berdasarkan label prediksi
            if prediction == 'spam':
                # Efek visual bahaya merah
                st.error("### 🚫 TERDETEKSI SPAM")
                st.warning("⚠️ **Saran Keamanan:** Jangan pernah mengklik tautan mencurigakan atau memberikan data pribadi Anda kepada pengirim ini.")
            else:
                # Efek visual sukses hijau
                st.success("### ✅ PESAN AMAN (HAM)")
                st.info("💡 **Informasi:** Pesan ini terlihat seperti obrolan normal sehari-hari.")
                # Efek animasi selebrasi balon terbang
                st.balloons()
            
            # Tampilan Meter Keyakinan AI
            st.write(f"**Tingkat Keyakinan Klasifikasi:** `{confidence * 100:.2f}%`")
            st.progress(confidence)
            
            # Simpan ke riwayat
            st.session_state["history"].append((input_sms, prediction, confidence))

# --- RIWAYAT ANALISIS ---
if st.session_state["history"]:
    st.write("")
    st.markdown("### 🕒 Riwayat Analisis")
    # Tampilkan maksimal 5 riwayat terakhir
    for msg, pred, conf in reversed(st.session_state["history"][-5:]):
        status_label = "🚫 SPAM" if pred == 'spam' else "✅ AMAN"
        st.write(f"- **{msg[:50]}...** → {status_label} (`{conf*100:.1f}%` yakin)")

# --- UPLOAD FILE UNTUK BATCH TESTING ---
st.divider()
st.subheader("📂 Analisis Massal (Batch Upload)")
uploaded_file = st.file_uploader("Upload file SMS dalam format CSV atau TSV", type=["csv","tsv"])
if uploaded_file:
    # Membaca data yang diupload
    df_upload = pd.read_csv(uploaded_file, sep='\t', names=['label','message'])
    st.write("### 📊 Pratinjau Data Unggahan:")
    st.dataframe(df_upload.head())

# Footer Aplikasi
st.markdown("<br><br><hr><p style='text-align: center; color: #475569; font-size: 11px;'>Tugas Mandiri 14 • Teknik Informatika Universitas Nusa Putra</p>", unsafe_allow_html=True)