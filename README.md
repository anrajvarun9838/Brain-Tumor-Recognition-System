#  Brain Tumor Recognition System  
A web-based intelligent system for brain tumor classification using deep learning and MRI images.  
This project uses a custom-trained CNN model (or transfer learning model vgg16) along with a Groq-powered AI explanation engine.

---

## Features  
✔ Upload MRI brain images
✔ Detect tumor type using trained AI model  
✔ Get instant prediction result  
✔ Clean and responsive frontend (HTML, CSS, JS)  
✔ Optional AI Explanation using Groq LLM 
✔ Fully modular Flask backend  

---

##  Project Structure

Brain-Tumor-Recognition/
│
├── main.py                     # Main Flask app
├── AI.py                      
├── requirements.txt           
├── README.md                  
├── .env                       
│
├── model/
│   └── model.h5               # Trained Brain Tumor Classification Model
│
│
├── frontend/
│   ├── css/css/
│   │   └── style.css          # Styling for frontend
│   │
│   ├── index.html
│   │
│   └── about.html
