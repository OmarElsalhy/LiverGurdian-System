import os
import json
import numpy as np
import nibabel as nib
from flask import Flask, request, jsonify
import tensorflow as tf
from skimage.transform import resize
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# ---------------------------
# 1. تعريف الـ Custom Objects (لمنع خطأ التحميل)
# ---------------------------
def combined_loss(y_true, y_pred): return 0
def dice_coef(y_true, y_pred): return 0
def dice_coefficient(y_true, y_pred): return 0

# ---------------------------
# 2. تحميل الموديل عند تشغيل السيرفر
# ---------------------------
def load_full_model():
    try:
        config_path = 'config.json'
        weights_path = 'model.weights.h5'
        
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            print("❌ Missing model files!")
            return None
            
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        custom_objs = {
            'combined_loss': combined_loss,
            'dice_coef': dice_coef,
            'dice_coefficient': dice_coefficient,
            'function': dice_coefficient,
            'loss': combined_loss
        } 
        
        print("--- Building model architecture ---")
        model = tf.keras.models.model_from_json(json.dumps(model_config), custom_objects=custom_objs)
        
        print("--- Loading weights ---")
        model.load_weights(weights_path)
        
        print("✅✅✅✅ MODEL LOADED SUCCESSFULLY ✅✅✅✅")
        return model
    except Exception as e:
        print(f"❌❌ CRITICAL LOAD ERROR: {str(e)}")
        return None

model_3d = load_full_model()

# ---------------------------
# 3. دالات المعالجة
# ---------------------------
def read_nii(nii_path):
    try:
        # بنستخدم mmap=False عشان نضمن إن الداتا كلها في الميموري
        img = nib.load(nii_path)
        data = img.get_fdata()
        return data
    except Exception as e:
        print(f"Nibabel error: {str(e)}")
        return None

def normalize_ct_scan(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)

def create_full_visualization(ct, pred_mask, status, ratio, confidence):
    n_slices = 5
    slice_indices = np.linspace(0, ct.shape[2]-1, n_slices, dtype=int)
    
    fig, axes = plt.subplots(2, n_slices, figsize=(20, 8))
    for idx, slice_idx in enumerate(slice_indices):
        axes[0, idx].imshow(ct[:, :, slice_idx], cmap='gray')
        axes[0, idx].set_title(f'Slice {slice_idx}')
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(ct[:, :, slice_idx], cmap='gray')
        axes[1, idx].imshow(pred_mask[0, :, :, slice_idx, 0], cmap='hot', alpha=0.4)
        axes[1, idx].set_title('Prediction')
        axes[1, idx].axis('off')
    
    plt.suptitle(f'{status} | Ratio: {ratio:.6f} | Conf: {confidence:.2%}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

# ---------------------------
# 4. الـ Endpoints
# ---------------------------
@app.route('/')
def home():
    status = "READY" if model_3d is not None else "NOT_LOADED"
    return jsonify({"project": "LiverGuardian 3D", "model_status": status})

@app.route('/predict_ct', methods=['POST'])
def predict_ct():
    if model_3d is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    
    file = request.files['file']
    temp_path = os.path.abspath("temp_inference.nii")

    try:
        # حفظ الملف "بإحكام"
        file.save(temp_path)
        file.close() 

        # قراءة ومعالجة
        ct = read_nii(temp_path)
        if ct is None:
            return jsonify({"success": False, "error": "Invalid NIfTI file"}), 400

        target_shape = (96, 96, 48)
        ct_resized = resize(ct, target_shape, preserve_range=True, anti_aliasing=True)
        ct_norm = normalize_ct_scan(ct_resized)
        ct_input = ct_norm[np.newaxis, ..., np.newaxis]

        # التوقع
        pred = model_3d.predict(ct_input, verbose=0)
        pred_mask = (pred > 0.5).astype(np.uint8)

        # النتائج
        ratio = float(np.sum(pred_mask) / np.prod(pred_mask.shape))
        confidence = float(np.mean(np.abs(pred - 0.5)) * 2)
        status_text = "CANCER DETECTED" if ratio >= 0.003 else "NO CANCER"

        viz_base64 = create_full_visualization(ct_norm, pred_mask, status_text, ratio, confidence)

        return jsonify({
            "success": True,
            "status": status_text,
            "tumor_ratio": f"{ratio:.6f}",
            "confidence": f"{confidence:.2%}",
            "visualization": viz_base64
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)