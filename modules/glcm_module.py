import os
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def calculate_glcm_features(image_path, angle_deg):
    """
    Membuka gambar (yang sudah grayscale) dan menghitung fitur tekstur GLCM.
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        if img_array.max() > 255:
            img_array = (img_array / img_array.max() * 255).astype(np.uint8)

        levels = 256
        angle_rad = np.deg2rad(angle_deg)
        glcm = graycomatrix(
            img_array,
            distances=[1],
            angles=[angle_rad],
            levels=levels,
            symmetric=True,
            normed=True
        )

        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        asm = graycoprops(glcm, 'ASM')[0, 0]

        return {
            'status': 'success',
            'contrast': round(contrast, 4),
            'energy': round(energy, 4),
            'homogeneity': round(homogeneity, 4),
            'asm': round(asm, 4)
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def process_image_file(file, angle):
    """
    Simpan file upload, ubah ke grayscale, dan hitung fitur GLCM.
    """
    if file.filename == '':
        return {'status': 'error', 'message': 'Nama file kosong.'}

    original_filename = file.filename
    original_filepath = os.path.join(UPLOAD_FOLDER, original_filename)
    file.save(original_filepath)

    try:
        img = Image.open(original_filepath)
        grayscale_img = img.convert('L')
        base, ext = os.path.splitext(original_filename)
        grayscale_filename = f"{base}_gray{ext}"
        grayscale_filepath = os.path.join(UPLOAD_FOLDER, grayscale_filename)
        grayscale_img.save(grayscale_filepath)
    except Exception as e:
        return {'status': 'error', 'message': f'Gagal konversi ke grayscale: {str(e)}'}

    # Hitung fitur
    features = calculate_glcm_features(grayscale_filepath, angle)
    features['original_image_url'] = f'/uploads/{original_filename}'
    features['grayscale_image_url'] = f'/uploads/{grayscale_filename}'
    return features
