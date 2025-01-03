# 🚀 Satellite Imagery System 🌍  
**Welcome to the Satellite Imagery System repository!**  

---

## 📑 Project Overview  
Forests are vital for ecological balance, but deforestation, climate change, and urbanization have increased the need for efficient monitoring methods. This project leverages 🌌 satellite imagery and 🧠 artificial intelligence to detect and monitor 🌲 trees, providing an accurate, scalable, and efficient alternative to traditional methods.

---

## 🛠️ Objectives  
- 🌳 **Detect and monitor trees** using satellite imagery and AI.  
- 📏 **Efficiently assess tree cover and forest density.**  
- 📡 **Provide real-time data visualization and monitoring** via a web application.

---

## ✨ Key Features  

### 🌐 Web Application:  
- 🚀 **Real-time Data Monitoring**: Access live data on forest coverage.  
- 📊 **Dashboard Interface**: User-friendly dashboards for analysis.  
- 🗺️ **Map Integration**: Navigation and visualization of geographical data.  
- 📉 **Data Visualization**: Comprehensive visual insights from satellite imagery.  

### 🧠 AI Model:  
- 🏆 Utilizes advanced deep learning algorithms to process satellite imagery (YOLOv8).  
- 🌳 Provides accurate outputs of tree cover and density.  

---

## 🛠️ Tools & Technologies  

### ⚙️ Deep Learning Model:  
- 🐍 **Python**
- 🌟 **YOLOv8**
- 🖼️ **OpenCV**  
- 📊 **Matplotlib**


### 💻 Frontend:  
- 🖌️ Web technologies for an intuitive user interface (HTML, CSS).  

### 🛠️ Backend:  
- **🌐 Flask Framework**:  
  - ✨ Lightweight and modular for flexibility.  
  - 🔗 Built-in support for RESTful APIs.  
  - 🖼️ Template integration with Jinja2 for dynamic HTML rendering.  

---

## 📚 How to Use  

### Prerequisites  
Before you get started, ensure you have the following:  
- 🐍 Python 3.8 or later installed.  
- 📦 Required Python libraries (listed in `requirements.txt`).  
- 🔧 Access to a dataset of satellite imagery (optional for testing).  
- 🌐 An active internet connection for accessing the web application.  

### 🛠️ Setup and Installation  

1. Clone the Repository:  
   ```bash
   git clone https://github.com/AhmedKamel200058/Satelite_Imagery_System.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download AI Model Weights:
   
   Place the pre-trained YOLOv8 model weights in the directory named best.pt.
   
4. Run the application:
   ```bash
   python app.py
   ```
5. Access the Application:
   ```bash
   Open your browser and go to http://127.0.0.1:5000/ to view the web interface.
   ```
---
## 🗺️ Implementation Plan  

1. 📥 **Data Collection**: Gather satellite imagery data for forests.  
2. 🧠 **AI Model Training**: Develop a machine learning model for tree detection.  
3. 🌐 **Web Application Development**: Integrate the model into a Flask-based web app with real-time monitoring and dashboard features.  
4. 🌍 **Deployment**: Host the application for users to access.

---
## 🖼️ Visual Model Summary  

Here is an example of the AI model detecting trees from satellite imagery:  

### 🖼️ Image Example  
![Tree Detection Example](Images/Output_model_image/tree_detection_example.png)  

---

### 🎥 Demo Video  

[📹 Watch the video](https://drive.google.com/file/d/1gBsj8EGIw8kcnAXdcxiSJk2Hm7WgZHAN/view?usp=drive_link)
