import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sistem Prediksi Risiko Penyakit Stroke", layout="wide")

#===PATH SETUP===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_stroke_knn.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "fitur_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data_pasien.csv")

#===SESSION STATE===
if "login_status" not in st.session_state:
    st.session_state.login_status = False

if "data_prediksi" not in st.session_state:
    st.session_state.data_prediksi = None

#===STYLE===
st.markdown("""
<style>
.title { font-size: 26px; font-weight: bold; color: #1F3A8A; }
.subtitle { font-size: 16px; color: #4B5563; margin-bottom: 10px; }
.section { font-size:16px; font-weight:600; margin-top:15px; margin-bottom:5px; }
.block-container { padding-top: 2.5 rem; }
hr { border: 1px solid #E5E7EB; }
</style>
""", unsafe_allow_html=True)

#===LOGIN CONFIG===
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "12345"

#===LOGIN PAGE===
def login_page():
    st.markdown('<div class="title">Login Administrator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Silakan Masuk Untuk Mengelola Data Pasien</div>', unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Masuk"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.login_status = True
            st.rerun()
        else:
            st.error("Username atau Password Salah!")

#===ADMIN DASHBOARD===
def admin_page():
    st.markdown('<div class="title">Dashboard Administrator</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Monitoring Data Pasien dan Hasil Prediksi</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("Logout"):
        st.session_state.login_status = False
        st.session_state.data_prediksi = None
        st.rerun()

    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
        st.error("Model atau file fitur belum tersedia.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)

    #====UPLOAD===
    st.markdown('<div class="section">Upload Data Rekam Medis Pasien</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is None and st.session_state.data_prediksi is None:
        st.info("Belum ada data yang diunggah. Silakan upload file CSV terlebih dahulu.")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            data.columns = data.columns.str.strip()

            if "ID Pasien" not in data.columns:
                st.error("File harus memiliki kolom 'ID Pasien'.")
                st.stop()

            id_data = data["ID Pasien"]

            data = data.drop(columns=["ID Pasien"], errors="ignore")
            data = data.drop(columns=["Stroke"], errors="ignore")

            missing_features = [f for f in feature_names if f not in data.columns]
            if missing_features:
                st.error(f"Kolom berikut tidak ditemukan: {missing_features}")
                st.stop()

            X = data[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
            predictions = model.predict(X)

            hasil_df = pd.DataFrame({
                "ID Pasien": id_data,
                "Hasil Prediksi": predictions
            })

            st.session_state.data_prediksi = hasil_df
            hasil_df.to_csv(DATA_PATH, index=False)

            st.success("Data berhasil diproses dan disimpan.")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.stop()

    #===TAMPILAN HASIL====
    if st.session_state.data_prediksi is not None:
        data = st.session_state.data_prediksi

        total = len(data)
        risiko = len(data[data["Hasil Prediksi"] == 1])
        tidak_risiko = len(data[data["Hasil Prediksi"] == 0])

        # ====Statistik + Grafik====
        col1, col2 = st.columns([1,2])

        with col1:
            st.markdown('<div class="section">Statistik Prediksi</div>', unsafe_allow_html=True)
            st.metric("Total Pasien", total)
            st.metric("Berisiko Stroke", risiko)
            st.metric("Tidak Berisiko", tidak_risiko)

        with col2:
            st.markdown('<div class="section">Distribusi Hasil Prediksi</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,3))
            ax.bar(["Tidak Berisiko", "Berisiko"], [tidak_risiko, risiko])
            ax.set_ylabel("Jumlah")
            st.pyplot(fig)

        # ====Tabel tersembunyi====
        st.markdown('<div class="section">Data Hasil Prediksi</div>', unsafe_allow_html=True)
        with st.expander("Tampilkan Tabel Data"):
            st.dataframe(data)

#===CEK HASIL PASIEN===
def cek_hasil_page():
    st.markdown('<div class="title">Cek Hasil Prediksi Pasien</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Masukkan ID Rekam Medis untuk Melihat Hasil</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        st.warning("Belum Ada Data Hasil Prediksi.")
        return

    data = pd.read_csv(DATA_PATH)

    id_input = st.text_input("Masukkan ID Pasien")

    if st.button("Cek Hasil"):
        hasil = data[data["ID Pasien"].astype(str) == id_input]

        if not hasil.empty:
            prediksi = hasil["Hasil Prediksi"].values[0]

            if prediksi == 1:
                st.error("Pasien Termasuk Dalam Kategori Berisiko Stroke.")
                st.warning("Ayo Segera Berkonsultasi Dengan Dokter.")
            else:
                st.success("Pasien Termasuk Dalam Kategori Tidak Berisiko Stroke.")
                st.info("Tetap Jaga Pola Hidup Sehat dan Lakukan Pemerikasaan Rutin.")
        else:
            st.warning("ID tidak ditemukan.")

#===NAVIGASI===
st.sidebar.title("Navigasi Sistem")

menu = st.sidebar.radio("Pilih Menu", ["Admin", "Cek Hasil Pasien"])

if menu == "Admin":
    if st.session_state.login_status:
        admin_page()
    else:
        login_page()
else:
    cek_hasil_page()