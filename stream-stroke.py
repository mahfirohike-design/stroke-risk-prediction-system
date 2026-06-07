import streamlit as st
import pandas as pd
import pickle


# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Risiko Stroke",
    layout="wide"
)

# STYLE TAMBAHAN
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f4fd8;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
.section {
    background-color: #f9fafc;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 0 10px rgba(0,0,0,0.05);
}
.result-card {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)


# LOAD MODEL
model = pickle.load(open('model_knn_stroke.pkl', 'rb'))
scaler = pickle.load(open('scaler_stroke.pkl', 'rb'))


# HEADER
st.markdown('<div class="main-title"> Prediksi Risiko Stroke</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Sistem Prediksi Risiko Stroke Dengan '
    '<b>K-Nearest Neighbor (KNN)</b> Berdasarkan Data Rekam Medis Pasien.</div>',
    unsafe_allow_html=True
)

st.divider()


# SIDEBAR INPUT
st.sidebar.header(" Input Data Pasien")

gender = st.sidebar.selectbox("Jenis Kelamin", ("Laki-Laki", "Perempuan"))
age = st.sidebar.slider("Usia")

hypertension = st.sidebar.selectbox("Hipertensi", ("Tidak", "Ya"))
heart_disease = st.sidebar.selectbox("Riwayat Penyakit Jantung", ("Tidak", "Ya"))

ever_married = st.sidebar.selectbox("Status Pernikahan", ("Belum Menikah", "Menikah"))

work_type = st.sidebar.selectbox(
    "Jenis Pekerjaan",
    ("Private", "Self-employed", "Children", "Never worked"),
)

Residence_type = st.sidebar.selectbox("Tempat Tinggal", ("Urban", "Rural"))

avg_glucose_level = st.sidebar.number_input(
    "Rata-rata Glukosa Darah")

bmi = st.sidebar.number_input("BMI")

smoking_status = st.sidebar.selectbox(
    "Status Merokok",
    ("Pernah", "Tidak Pernah", "Merokok", "Tidak Diketahui"),
)

# KONVERSI
hypertension = 1 if hypertension == "Ya" else 0
heart_disease = 1 if heart_disease == "Ya" else 0


# FEATURE MODEL
feature_names = [
    'id','age','hypertension','heart_disease',
    'avg_glucose_level','bmi',
    'gender_Male','gender_Other',
    'ever_married_Yes',
    'work_type_Never_worked','work_type_Private',
    'work_type_Self-employed','work_type_children',
    'Residence_type_Urban',
    'smoking_status_formerly smoked',
    'smoking_status_never smoked',
    'smoking_status_smokes'
]

input_df = pd.DataFrame(0, index=[0], columns=feature_names)

input_df["id"] = 0
input_df["age"] = age
input_df["hypertension"] = hypertension
input_df["heart_disease"] = heart_disease
input_df["avg_glucose_level"] = avg_glucose_level
input_df["bmi"] = bmi

# gender
if gender == "Laki-Laki":
    input_df["gender_Male"] = 1
else:
    input_df["gender_Other"] = 1

# married
input_df["ever_married_Yes"] = 1 if ever_married == "Menikah" else 0

# residence
input_df["Residence_type_Urban"] = 1 if Residence_type == "Urban" else 0

# work
if work_type == "Private":
    input_df["work_type_Private"] = 1
elif work_type == "Self-employed":
    input_df["work_type_Self-employed"] = 1
elif work_type == "Children":
    input_df["work_type_children"] = 1
elif work_type == "Never worked":
    input_df["work_type_Never_worked"] = 1

# smoking
if smoking_status == "Pernah":
    input_df["smoking_status_formerly smoked"] = 1
elif smoking_status == "Tidak Pernah":
    input_df["smoking_status_never smoked"] = 1
elif smoking_status == "Merokok":
    input_df["smoking_status_smokes"] = 1
elif smoking_status == "Tidak Diketahui":
    input_df["smoking_status_smokes"] = 1


# SCALING
input_scaled = scaler.transform(input_df)


# AREA HASIL
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("###  Ringkasan Input")
    st.dataframe(input_df)

with col2:
    st.markdown("###  Prediksi")

    if st.button(" Jalankan Prediksi", use_container_width=True):

        pred = model.predict(input_scaled)

        if pred[0] == 1:
            st.markdown(
                '<div class="result-card" style="background:#ffe1e1;color:#b30000;">'
                ' <b>BERISIKO STROKE</b><br>Segera konsultasi ke dokter.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-card" style="background:#e6ffef;color:#008a3a;">'
                ' <b>TIDAK BERISIKO STROKE</b>'
                '</div>',
                unsafe_allow_html=True
            )

st.divider()

