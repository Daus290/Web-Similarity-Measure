import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- Config ----------------
st.set_page_config(page_title="Similarity Measure - Otsu & Angle Distance Signature", layout="wide")

st.title("🔍 Similarity Measure pada Citra Digital")
st.caption("Implementasi Metode Otsu Thresholding dan Angle Distance Signature menggunakan Python & Streamlit")

# Pilihan mode
mode = st.sidebar.selectbox(
    "Pilih Mode Analisis:",
    ("Analisis 1 Gambar", "Perbandingan 2 Gambar")
)

st.divider()

# ---------------- Function ----------------
def process_image(uploaded_file):
    """Proses utama: Otsu + kontur + centroid + signature"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(main_contour)
    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

    angles, distances = [], []
    for point in main_contour:
        x, y = point[0]
        dx, dy = x - cx, y - cy
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        distance = np.sqrt(dx**2 + dy**2)
        angles.append(angle)
        distances.append(distance)

    sorted_pairs = sorted(zip(angles, distances))
    sorted_angles, sorted_distances = zip(*sorted_pairs)
    sorted_angles = np.array(sorted_angles)
    sorted_distances = np.array(sorted_distances)
    normalized = sorted_distances / sorted_distances.max()

    return image, gray, binary, sorted_angles, sorted_distances, normalized, (cx, cy), main_contour


# ==============================================================
# Mode 1 — Analisis 1 Gambar
# ==============================================================
if mode == "Analisis 1 Gambar":
    st.header("🖼️ Analisis Citra Tunggal")
    uploaded_file = st.file_uploader("📁 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image, gray, binary, angles, distances, normalized, (cx, cy), contour = process_image(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Citra Asli")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.subheader("Citra Binerisasi (Otsu)")
            st.image(binary, use_container_width=True)

        with col2:
            st.subheader("Kontur dan Centroid")
            contour_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_img, [contour], -1, (0, 0, 255), 1)
            cv2.circle(contour_img, (cx, cy), 5, (0, 255, 0), -1)
            st.image(contour_img, use_container_width=True)

            st.subheader("Grafik Angle Distance Signature")
            fig, ax = plt.subplots()
            ax.plot(angles, normalized, color='blue')
            ax.set_xlabel("Sudut (derajat)")
            ax.set_ylabel("Jarak Normalisasi")
            ax.grid(True)
            st.pyplot(fig)

        st.divider()
        st.subheader("📊 Tabel Nilai Sudut dan Jarak (10 Baris Pertama)")
        df = pd.DataFrame({
            "Sudut (derajat)": angles,
            "Jarak dari Centroid": distances,
            "Jarak Normalisasi": normalized
        })
        st.dataframe(df.head(10), use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Unduh Hasil sebagai CSV",
            data=csv,
            file_name="hasil_signature.csv",
            mime="text/csv"
        )

        st.success("✅ Analisis selesai! Grafik dan tabel berhasil dibuat.")
    else:
        st.info("Silakan unggah satu gambar untuk memulai analisis.")


# ==============================================================
# Mode 2 — Perbandingan 2 Gambar
# ==============================================================
elif mode == "Perbandingan 2 Gambar":
    st.header("⚖️ Pengukuran Kemiripan Dua Gambar")

    col_upload = st.columns(2)
    with col_upload[0]:
        file1 = st.file_uploader("📁 Unggah Gambar 1", type=["jpg", "jpeg", "png"], key="img1")
    with col_upload[1]:
        file2 = st.file_uploader("📁 Unggah Gambar 2", type=["jpg", "jpeg", "png"], key="img2")

    if file1 and file2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Citra 1")
            img1, gray1, bin1, ang1, dist1, norm1, c1, contour1 = process_image(file1)
            st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.image(bin1, caption="Hasil Otsu (Gambar 1)", use_container_width=True)

        with col2:
            st.subheader("🖼️ Citra 2")
            img2, gray2, bin2, ang2, dist2, norm2, c2, contour2 = process_image(file2)
            st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.image(bin2, caption="Hasil Otsu (Gambar 2)", use_container_width=True)

        st.divider()

        st.subheader("📈 Perbandingan Grafik Angle Distance Signature")
        fig, ax = plt.subplots()
        ax.plot(ang1, norm1, label="Gambar 1", color='blue')
        ax.plot(ang2, norm2, label="Gambar 2", color='red', alpha=0.7)
        ax.set_xlabel("Sudut (derajat)")
        ax.set_ylabel("Jarak Normalisasi")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Samakan panjang signature
        min_len = min(len(norm1), len(norm2))
        sig1 = np.interp(np.linspace(0, 1, 360), np.linspace(0, 1, min_len), norm1[:min_len])
        sig2 = np.interp(np.linspace(0, 1, 360), np.linspace(0, 1, min_len), norm2[:min_len])

        euclidean_distance = np.linalg.norm(sig1 - sig2)
        similarity_score = max(0, 100 - (euclidean_distance * 100))

        st.divider()
        st.subheader("📊 Hasil Perbandingan")
        st.metric(label="Jarak Euclidean", value=f"{euclidean_distance:.4f}")
        st.metric(label="Persentase Kemiripan", value=f"{similarity_score:.2f}%")

        df = pd.DataFrame({
            "Sudut (derajat)": ang1[:10],
            "Jarak Normalisasi Gambar 1": norm1[:10],
            "Jarak Normalisasi Gambar 2": norm2[:10]
        })
        st.dataframe(df, use_container_width=True)

        # Analisis otomatis sederhana
        st.divider()
        if similarity_score > 80:
            st.success("✅ Kedua gambar memiliki bentuk yang sangat mirip.")
        elif similarity_score > 50:
            st.warning("⚠️ Kedua gambar memiliki kemiripan sedang.")
        else:
            st.error("❌ Kedua gambar memiliki bentuk yang berbeda.")

    else:
        st.info("Silakan unggah dua gambar untuk menghitung tingkat kemiripan.")
