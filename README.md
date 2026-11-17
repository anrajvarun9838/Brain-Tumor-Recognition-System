# Brain Tumor Recognition System

An AI-powered brain tumor recognition system that uses **Deep Learning and Transfer Learning (VGG16)** to classify brain MRI images into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

The project combines a trained CNN-based image classification model with a Flask web application and a Groq-powered AI Information Assistant to provide an interactive experience for brain MRI image analysis.

---

## Project Overview

Brain tumors are serious medical conditions that require accurate examination and diagnosis. This project explores the application of Deep Learning in medical image classification by analyzing MRI brain scans.

The system uses a VGG16-based transfer learning approach to extract image features and classify MRI images into four categories.

Users can upload an MRI image through the web interface, receive a predicted category with a confidence score, and explore additional educational information through an integrated AI assistant.

## ✨ Key Features

### 1. Brain Tumor Classification

- Upload brain MRI images through the web interface.
- Analyze uploaded images using a trained VGG16 model.
- Classify images into four categories:
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor

### 2. Prediction Confidence

- Displays the predicted tumor category.
- Provides the model's prediction confidence score.
- Presents classification results through an interactive interface.

### 3. AI Information Assistant

Integrated with the Groq API to provide an AI-powered information assistant.

- Provides educational information about brain tumor categories.
- Explains general characteristics of different tumor types.
- Answers questions about symptoms and prevention.
- Helps users understand the classification results.

### 4. Web Application

- Flask-based backend.
- MRI image upload functionality.
- Prediction result display.
- Interactive user interface.
- Integrated AI information assistant.

---

## Deep Learning Model

The core classification model is built using **VGG16 with Transfer Learning**.

The model is developed and trained in Google Colab using TensorFlow/Keras.

Training notebook:

[`models/Brain_Tumor_Training.ipynb`](models/Brain_Tumor_Training.ipynb)


## Dataset and Preprocessing

The model is trained and evaluated using brain MRI images organized into separate training and testing directories.

### Dataset Structure

```text
MRI Images/
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

##  Model Performance

The model was trained for five epochs using the configured VGG16 transfer learning architecture.

### Training Results

| Epoch | Training Accuracy | Training Loss |
|---|---:|---:|
| 1 | 74.46% | 0.6687 |
| 2 | 90.11% | 0.2710 |
| 3 | 93.30% | 0.1837 |
| 4 | 94.69% | 0.1540 |
| 5 | 96.30% | 0.0947 |

### Testing Results

The model was evaluated on 1,311 testing images.

| Metric | Result |
|---|---:|
| Test Accuracy | 95% |
| Macro Precision | 95% |
| Macro Recall | 94% |
| Macro F1-Score | 94% |
| Weighted F1-Score | 95% |

### Classification Report

| Class Index | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.90 | 0.97 | 0.94 | 300 |
| 1 | 0.93 | 0.87 | 0.90 | 306 |
| 2 | 0.96 | 0.99 | 0.97 | 405 |
| 3 | 0.99 | 0.95 | 0.97 | 300 |

### Evaluation Techniques

The training notebook includes:

- Training accuracy and loss visualization.
- Classification report.
- Confusion matrix.
- One-vs-rest ROC curves.
- AUC calculation for individual classes.

*These metrics represent the results recorded in the training notebook and are not evidence of clinical validation.*

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| CNN Architecture | VGG16 |
| Transfer Learning | ImageNet |
| Data Processing | NumPy, Pillow |
| Data Visualization | Matplotlib, Seaborn |
| Model Evaluation | Scikit-learn |
| Model Training | Google Colab |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| AI Assistant | Groq API |

---

## 📂 Project Structure

```text
Brain-Tumor-Recognition-System/
│
├── main.py
├── AI.py
├── requirements.txt
│
├── models/
│   ├── Brain_Tumor_Training.ipynb
│   └── MRI.h5
│
├── Testing/
│
├── frontend/
│
└── README.md
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/anrajvarun9838/Brain-Tumor-Recognition-System.git
```

### 2. Navigate to the Project

```bash
cd Brain-Tumor-Recognition-System
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Groq API

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from [Groq Console](https://console.groq.com/).

Never commit your API key or `.env` file to GitHub.

### 6. Run the Application

```bash
python main.py
```

Open your browser:

```text
http://127.0.0.1:5000/
```


## ⚠️ Medical Disclaimer

This project is developed for educational and research purposes only.

It is not intended to replace professional medical diagnosis, medical advice, or treatment.

The model may produce incorrect predictions. Always consult a qualified healthcare professional for medical evaluation.

---
