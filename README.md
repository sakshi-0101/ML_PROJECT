# 🎓 Student Performance Prediction Web App

An end-to-end Machine Learning project that predicts student performance using a trained pipeline and serves predictions through a Flask web application.

---

## 🚀 Overview

This project implements a complete ML pipeline and deploys it using Flask.
Users can input student details through a web interface and get instant predictions.

---

## 🧠 Problem Statement

Predict a student's performance based on the following features:

- Gender  
- Race/Ethnicity  
- Parental Level of Education  
- Lunch Type  
- Test Preparation Course  
- Reading Score  
- Writing Score  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- CatBoost  
- Flask  
- HTML/CSS  

---

## 📂 Project Structure

MLPROJECT/
│
├── artifacts/                  
├── catboost_info/              
├── logs/                       
├── notebook/
│   └── data/
│       ├── EDA STUDENT PERFORMANCE.ipynb
│       └── stud.csv
│
├── src/
│   ├── components/             
│   ├── pipeline/               
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   ├── index.html              
│   └── home.html               
│
├── app.py                      
├── application.py              
├── requirements.txt
├── setup.py
├── README.md
└── stud.csv

---

## ⚙️ Installation & Setup

### Clone the Repository
git clone https://github.com/sakshi-0101/student-performance-ml.git  
cd student-performance-ml 

### Create Virtual Environment
conda create -p venv python=3.8 -y  
conda activate venv/  

### Install Dependencies
pip install -r requirements.txt  

---

## ▶️ Run the Application

python app.py  

---

## 🌐 Access the App

home:http://127.0.0.1:5000
index:http://127.0.0.1:5000/predictdata
---

## 🔄 How It Works

1. User inputs data via web form  
2. Flask collects data using request.form  
3. Data is converted into a custom object  
4. Transformed into DataFrame  
5. Passed into prediction pipeline  
6. Model generates prediction  
7. Result displayed on UI  

---

## 📊 Example Input

- Gender: Female  
- Race/Ethnicity: Group D  
- Parental Education: Associate's Degree  
- Lunch: Standard  
- Test Prep Course: None  
- Reading Score: 34  
- Writing Score: 56  

---

## ✨ Features

- End-to-end ML pipeline  
- Modular structure  
- Real-time predictions  
- Flask web app  

---

