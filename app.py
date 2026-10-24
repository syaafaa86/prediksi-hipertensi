import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ======================
# Pengaturan Halaman (harus paling awal)
# ======================
st.set_page_config(
    page_title="Prediksi Hipertensi",
    page_icon="🩺",
    layout="centered"
)

# ======================
# Load Model & Scaler
# ======================
@st.cache_resource
def load_model():
    model = joblib.load('svm_hipertensi.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ======================
# Header
# ======================
st.title("🩺 Prediksi Hipertensi")
st.markdown("**Pendekatan Prediksi Menggunakan model SVM**")
st.warning(
    "Hasil ini hanyalah perkiraan dari model machine learning dan **bukan diagnosis medis**. "
    "Untuk pemeriksaan yang akurat, konsultasikan dengan dokter.",
    icon="⚠️"
)
st.divider()

# ======================
# Form Input
# ======================
st.subheader("📋 Masukkan Data Pasien")

with st.form("form_pasien"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (Umur)", min_value=1, max_value=120, value=40,
            help="Usia pasien dalam tahun."
        )
        salt_intake = st.number_input(
            "Salt Intake (Konsumsi Garam)", min_value=0.0, max_value=20.0,
            value=5.0, step=0.1,
            help="Konsumsi garam harian dalam gram."
        )
        stress_score = st.number_input(
            "Stress Score (Tingkat Stres)", min_value=0, max_value=10, value=5,
            help="Skala 0 (tidak stres) sampai 10 (sangat stres)."
        )
        bp_history = st.selectbox(
            "BP History (Riwayat Tekanan Darah)",
            ["Normal", "Prehypertension", "Hypertension"],
            help="Kategori tekanan darah pasien sebelumnya."
        )
        sleep_duration = st.number_input(
            "Sleep Duration (Durasi Tidur)", min_value=0.0, max_value=15.0,
            value=7.0, step=0.5,
            help="Rata-rata jam tidur per hari."
        )

    with col2:
        bmi = st.number_input(
            "BMI (Indeks Massa Tubuh)", min_value=10.0, max_value=50.0,
            value=24.0, step=0.1,
            help="Berat badan (kg) dibagi tinggi badan (m) kuadrat."
        )
        medication = st.selectbox(
            "Medication (Riwayat Konsumsi Obat)",
            ["None", "ACE Inhibitor", "Beta Blocker", "Diuretic", "Other"],
            help="Jenis obat tekanan darah yang pernah dikonsumsi."
        )
        family_history = st.selectbox(
            "Family History (Riwayat Keluarga)", ["No", "Yes"],
            help="Apakah ada anggota keluarga dengan hipertensi?"
        )
        exercise_level = st.selectbox(
            "Exercise Level (Aktivitas Fisik)", ["Low", "Moderate", "High"],
            help="Tingkat aktivitas fisik sehari-hari."
        )
        smoking_status = st.selectbox(
            "Smoking Status (Status Merokok)", ["Non-smoker", "Smoker"]
        )

    submitted = st.form_submit_button(
        "🔍 Prediksi Sekarang", type="primary", use_container_width=True
    )

# ======================
# Encoding
# ======================
bp_mapping = {"Normal": 0, "Prehypertension": 1, "Hypertension": 2}
med_mapping = {
    "ACE Inhibitor": 0,
    "Beta Blocker": 1,
    "Diuretic": 2,
    "None": 3,
    "Other": 4
}
exercise_map = {"Low": 0, "Moderate": 1, "High": 2}

# ======================
# Prediksi
# ======================
if submitted:
    # Urutan harus sama dengan saat training
    data = np.array([[
        age,
        salt_intake,
        stress_score,
        bp_mapping[bp_history],
        sleep_duration,
        bmi,
        med_mapping[medication],
        1 if family_history == "Yes" else 0,
        exercise_map[exercise_level],
        1 if smoking_status == "Smoker" else 0
    ]])

    data_scaled = scaler.transform(data)
    prediksi = model.predict(data_scaled)[0]

    # predict_proba hanya ada jika model dilatih dengan probability=True
    try:
        proba = model.predict_proba(data_scaled)[0]
    except (AttributeError, NotImplementedError):
        proba = None

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    if prediksi == 1:
        st.error("### ⚠️ BERISIKO HIPERTENSI")
        st.write(
            "Disarankan untuk memeriksakan tekanan darah ke dokter atau fasilitas kesehatan, "
            "serta mulai mengurangi garam, mengelola stres, dan rutin berolahraga."
        )
    else:
        st.success("### ✅ TIDAK BERISIKO HIPERTENSI")
        st.write(
            "Pertahankan gaya hidup sehat: batasi garam, tidur cukup, "
            "aktif bergerak, dan cek tekanan darah secara berkala."
        )

    if proba is not None:
        st.write("**Tingkat keyakinan model:**")
        c1, c2 = st.columns(2)
        c1.metric("Tidak Berisiko", f"{proba[0]*100:.1f}%")
        c2.metric("Berisiko", f"{proba[1]*100:.1f}%")
        st.progress(float(proba[1]), text=f"Probabilitas hipertensi: {proba[1]*100:.1f}%")

    # Ringkasan input dengan label asli (bukan angka hasil encoding)
    with st.expander("Lihat data yang diprediksi"):
        ringkasan = pd.DataFrame({
            "Fitur": [
                "Umur", "Konsumsi Garam (g/hari)", "Tingkat Stres (0-10)",
                "Riwayat Tekanan Darah", "Durasi Tidur (jam)", "BMI",
                "Riwayat Obat", "Riwayat Keluarga", "Aktivitas Fisik",
                "Status Merokok"
            ],
            "Nilai": [
                age, salt_intake, stress_score, bp_history, sleep_duration,
                bmi, medication, family_history, exercise_level, smoking_status
            ]
        }).astype({"Nilai": str})
        st.dataframe(ringkasan, use_container_width=True, hide_index=True)

# ======================
# Sidebar
# ======================
with st.sidebar:
    st.header("ℹ️ Informasi")
    st.write("Aplikasi ini menggunakan model **Support Vector Machine (SVM) kernel RBF**.")
    st.write("**Keterangan:**")
    st.markdown("""
    - **Age**: Usia pasien (tahun)
    - **Salt Intake**: Konsumsi garam harian (gram)
    - **Stress Score**: Skala 0–10 tingkat stres psikologis
    - **BP History**: Riwayat tekanan darah: Normal, Prehypertension, Hypertension
    - **Sleep Duration**: Rata-rata jam tidur per hari
    - **BMI**: Indeks Massa Tubuh (berat/tinggi badan)
    - **Medication**: Obat sebelumnya: None, Beta Blocker, Diuretic, ACE Inhibitor, Other
    - **Family History**: Riwayat keluarga hipertensi: Yes/No
    - **Exercise Level**: Aktivitas fisik: Low, Moderate, High
    - **Smoking Status**: Smoker / Non-smoker
    """)