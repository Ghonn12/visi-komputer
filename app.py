import os
import cv2
import base64
import numpy as np
import pandas as pd
import time
from flask import Flask, render_template, request, jsonify, send_from_directory

# --- Impor Model ---
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Impor GLCM (Minggu 1) ---
try:
    from modules.glcm_module import process_image_file
except ImportError:
    print("Peringatan: 'modules/glcm_module.py' tidak ditemukan. Rute Minggu 1 mungkin gagal.")
    # Buat fungsi dummy jika tidak ada
    def process_image_file(file, angle):
        return {'status': 'error', 'message': 'Modul GLCM tidak ditemukan'}

# --- Konfigurasi Aplikasi ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DATA_PATH = os.path.join("data", "Skin_NonSkin.txt")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- CACHE MODEL ---
# Kunci: "model_type_k_colorspace" (misal, "MLP_0_BGR")
# Nilai: Dict berisi {'model', 'scaler', 'fpr', 'tpr', 'roc_auc'}
model_cache = {}


# ===================== LOGIKA INTI MODEL =====================

def train_model(k_value=3, color_space="RGB", model_type="KNN"):
    """
    Melatih model klasifikasi warna kulit.
    Fungsi ini sekarang dipanggil HANYA JIKA model belum ada di cache.
    """
    # (Argumen color_space diabaikan, kita selalu melatih pada BGR sesuai jurnal)
    print(f"--- MEMULAI PELATIHAN: {model_type}, K={k_value} ---")
    start_time = time.time()
    
    try:
        df = pd.read_csv(DATA_PATH, sep="\t", names=["B", "G", "R", "Class"])
        df.dropna(inplace=True)
    except FileNotFoundError:
        print(f"KESALAHAN FATAL: File dataset tidak ditemukan di {DATA_PATH}")
        return None

    # Fitur (X) adalah B,G,R. Label (y) adalah 'Class'
    pixels = df[["B", "G", "R"]].values
    y = df["Class"].values # Label 1 = Kulit, Label 2 = Bukan Kulit
    print(f"Menggunakan {len(pixels)} piksel BGR sebagai fitur.")
    
    # Normalisasi
    print("Menyesuaikan scaler...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(pixels)

    print("Membagi data train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Pilih model
    print(f"Melatih model {model_type}...")
    if model_type == "MLP":
        model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=300, random_state=42)
    elif model_type == "NB":
        model = GaussianNB()
    else:  # Default: KNN
        model = KNeighborsClassifier(n_neighbors=k_value, n_jobs=-1) # n_jobs=-1 untuk mempercepat

    model.fit(X_train, y_train)
    
    print("Mengevaluasi model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Ambil probabilitas untuk kelas '1' (kulit)
    class_index = 0
    if len(model.classes_) > 1 and model.classes_[1] == 1:
        class_index = 1
    y_prob_skin = y_prob[:, class_index]

    acc = accuracy_score(y_test, y_pred)
    
    # Hitung ROC dari probabilitas (lebih akurat)
    fpr, tpr, _ = roc_curve(y_test, y_prob_skin, pos_label=1) 
    roc_auc = auc(fpr, tpr)

    end_time = time.time()
    print(f"--- PELATIHAN SELESAI ({end_time - start_time:.2f} detik) ---")

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": acc,
        "fpr": fpr.tolist(), # Konversi ke list untuk JSON
        "tpr": tpr.tolist(), # Konversi ke list untuk JSON
        "roc_auc": roc_auc,
    }

def get_model_from_cache(k_value, color_space, model_type):
    """
    Mengambil model dari cache. Jika tidak ada, latih model baru dan simpan.
    """
    color_space_key = "BGR" # Kita selalu pakai BGR
    
    if model_type != "KNN":
        k_value = 0 # K tidak relevan jika bukan KNN
        
    key = f"{model_type}_{k_value}_{color_space_key}"
    
    if key not in model_cache:
        print(f"Model {key} tidak ditemukan di cache. Melatih model baru...")
        model_data = train_model(k_value, color_space, model_type) 
        if model_data is None:
            raise Exception(f"Gagal melatih model. Apakah '{DATA_PATH}' ada?")
        model_cache[key] = model_data
    else:
        print(f"Menggunakan model {key} dari cache.")
        
    return model_cache[key]


# ===================== MINGGU 1: GLCM =====================
@app.route('/')
def index():
    """Halaman utama (GLCM)."""
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve gambar yang diunggah."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/process', methods=['POST'])
def process_image_route():
    """Proses gambar dan hitung fitur GLCM."""
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'Tidak ada file gambar yang diunggah.'}), 400

    file = request.files['image']
    angle = int(request.form.get('angle', 0))
    result = process_image_file(file, angle)
    return jsonify(result)


# ===================== MINGGU 2: DETEKSI WARNA KULIT =====================
@app.route('/tugas2')
def tugas2():
    """Halaman Deteksi Warna Kulit."""
    return render_template('tugas2.html')


@app.route('/color_distribution')
def color_distribution_route():
    """Ambil data distribusi warna kulit vs bukan kulit."""
    try:
        df = pd.read_csv(DATA_PATH, sep="\t", names=["B", "G", "R", "Class"])
        count_skin = (df["Class"] == 1).sum()
        count_non = (df["Class"] == 2).sum()
        return jsonify({'skin': int(count_skin), 'non_skin': int(count_non)})
    except Exception as e:
        print(f"Error di /color_distribution: {e}")
        return jsonify({'skin': 0, 'non_skin': 0})


@app.route('/skin_process', methods=['POST'])
def skin_process_route():
    """Proses deteksi kulit dari gambar yang diunggah (upload biasa)."""
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'Tidak ada file gambar.'}), 400

    try:
        file = request.files['image']
        k = int(request.form.get('k_value', 3))
        color_space = request.form.get('color_space', 'RGB') # Diabaikan, tapi kita ambil
        model_type = request.form.get('algorithm', 'knn').upper() 

        # 1. Ambil model (latih jika belum ada)
        model_data = get_model_from_cache(k, color_space, model_type)
        model = model_data['model']
        scaler = model_data['scaler']

        # 2. Baca gambar yang diunggah
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # 3. Ubah gambar menjadi daftar piksel & Normalisasi
        h, w, _ = img_bgr.shape
        pixels_bgr = img_bgr.reshape(-1, 3) # Selalu gunakan BGR
        pixels_scaled = scaler.transform(pixels_bgr)

        # 4. Prediksi
        print(f"Memprediksi {len(pixels_scaled)} piksel dengan {model_type}...")
        predictions = model.predict(pixels_scaled)

        # 5. Buat gambar hasil (Masking)
        mask = (predictions == 1).reshape(h, w).astype(np.uint8)
        segmented_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

        # 6. Simpan file
        ts = int(time.time())
        original_filename = f"original_{ts}.jpg"
        segmented_filename = f"segmented_{ts}.jpg"
        
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, original_filename), img_bgr)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, segmented_filename), segmented_img)

        # 7. Kirim kembali hasil
        result = {
            'status': 'success',
            'original_image': f'/uploads/{original_filename}',
            'segmented_image': f'/uploads/{segmented_filename}',
            **model_data # Tambahkan 'accuracy', 'fpr', 'tpr', 'roc_auc'
        }
        # Hapus model dan scaler dari dict sebelum dikirim sebagai JSON
        del result['model']
        del result['scaler']

        return jsonify(result)

    except Exception as e:
        print(f"Error di rute /skin_process: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/skin_camera_process', methods=['POST'])
def skin_camera_process_route():
    """Terima frame kamera (base64) dan proses deteksi warna kulit."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'message': 'Gambar tidak ditemukan.'}), 400

        k = int(data.get('k_value', 3))
        color_space = data.get('color_space', 'RGB') # Diabaikan
        model_type = data.get('algorithm', 'knn').upper()

        # 1. Ambil model (latih jika belum ada)
        model_data = get_model_from_cache(k, color_space, model_type)
        model = model_data['model']
        scaler = model_data['scaler']

        # 2. Decode gambar base64
        data_url = data['image'].split(',')[1]
        img_data = base64.b64decode(data_url)
        npimg = np.frombuffer(img_data, np.uint8)
        img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # 3. Prediksi
        h, w, _ = img_bgr.shape
        pixels_bgr = img_bgr.reshape(-1, 3) # Selalu BGR
        pixels_scaled = scaler.transform(pixels_bgr)
        predictions = model.predict(pixels_scaled)

        # 4. Buat gambar hasil (Masking)
        mask = (predictions == 1).reshape(h, w).astype(np.uint8)
        segmented_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        
        # 5. Encode kembali ke base64
        _, buffer = cv2.imencode('.jpg', segmented_img)
        img_base64_result = base64.b64encode(buffer).decode('utf-8')

        # --- PERBAIKAN DIMULAI ---
        # Kirim semua data model (fpr, tpr, roc_auc)
        # BUKAN HANYA roc_auc
        result = {
            'status': 'success',
            'image': f'data:image/jpeg;base64,{img_base64_result}',
            **model_data 
        }
        del result['model']
        del result['scaler']
        return jsonify(result)
        # --- PERBAIKAN SELESAI ---
        
    except Exception as e:
        print(f"Error di rute /skin_camera_process: {e}")
        return jsonify({'status': 'error', 'message': f'Kesalahan saat memproses kamera: {str(e)}'}), 500

# ===================== MAIN EXECUTION =====================
if __name__ == '__main__':
    # Pastikan file dataset ada
    if not os.path.exists(DATA_PATH):
        print("="*50)
        print(f"PERINGATAN: File dataset '{DATA_PATH}' tidak ditemukan.")
        print("Aplikasi akan berjalan, tetapi akan gagal saat pemrosesan gambar.")
        print("Silakan unduh 'Skin_NonSkin.txt' dari UCI dan letakkan di folder 'data'.")
        print("="*50)
        
    app.run(debug=True)