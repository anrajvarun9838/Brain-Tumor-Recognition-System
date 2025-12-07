from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from AI import ai_bp

app = Flask(__name__, template_folder='frontend')
app = Flask(__name__, template_folder='frontend', static_folder='frontend')

@app.route('/about')
def about():
    return render_template('about.html')

app.register_blueprint(ai_bp)

model =load_model('models/brain_tumor_vgg16.h5')

class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

UPLOAD_FOLDER = './Testing'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor Detected", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    file_path = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            result, confidence = predict_tumor(file_path)

            confidence = f"{confidence * 100:.2f}%"

            return render_template(
                'index.html',
                result=result,
                confidence=confidence,
                file_path=f'/Testing/{file.filename}'
            )

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        file_path=file_path
    )

@app.route('/Testing/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    app.run(debug=True, host='0.0.0.0', port=5000)