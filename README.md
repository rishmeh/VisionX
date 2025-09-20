# VisionX – Assistive Smart Glasses for the Visually Impaired  

VisionX is an affordable, offline, and AI-powered solution designed to help blind and visually impaired individuals navigate the world with greater independence and safety. It integrates **obstacle detection, text reading, QR code scanning, face recognition, and currency validation** into a single system that works in real-time without requiring internet connectivity.  

---

## 🚀 Features
- **Real-Time Obstacle Detection**: Detects harmful objects and estimates their distance.  
- **Face Recognition & Anti-Impersonation**: Identifies known individuals and alerts against potential fraud.  
- **Currency Recognition & Fraud Prevention**: Reads denominations and detects counterfeit notes.  
- **Text & QR Code Reading**: Handles printed text and QR codes on the go.  
- **Offline Functionality**: Works without internet for reliability anytime, anywhere.  
- **User-Friendly Output**: Provides instant audio feedback via text-to-speech.  

---

## 📂 Project Structure
```

VisionX/
│── espFile.ino        # ESP32-CAM firmware (set SSID & password before uploading)
│── script.py          # Runs VisionX with ESP32-CAM video feed
│── onDevice.py        # Runs VisionX using computer webcam
│── requirements.txt   # Python dependencies
│── data/              # Registered user images, calibration files, temp storage

````

---

## ⚙️ Setup Instructions

### 1. ESP32-CAM Integration
1. Open `espFile.ino` in Arduino IDE.  
2. Update your **WiFi SSID** and **password** in the file.  
3. Upload the code to your ESP32-CAM module.  
4. Note the **IP address** shown in the Serial Monitor (e.g., `http://192.168.x.xxx:81/stream`).  

⚠️ Ensure your **computer and ESP32-CAM are connected to the same hotspot/network**.  

---

### 2. Computer Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/rishmeh/VisionX.git
   cd VisionX
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. To run with **ESP32-CAM stream**, edit `script.py` and set your ESP32-CAM IP:

   ```python
   stream_url = "http://<ESP32-CAM-IP>:81/stream"
   ```

   Then run:

   ```bash
   python script.py
   ```
4. To run with your **computer webcam**, use:

   ```bash
   python onDevice.py
   ```

---

## 🛠 Tech Stack

* **Hardware**: ESP32-CAM, Raspberry Pi (optional), Cameras, Audio Output
* **Software**: Python, Embedded C/C++
* **Computer Vision & AI**: YOLOv10, OpenCV, TensorFlow Lite, EasyOCR, SFace for face recognition
* **Feedback**: Text-to-Speech (pyttsx3), Audio/Haptic Alerts

---

## 📌 Roadmap

* [x] ESP32-CAM streaming integration
* [x] Object detection with YOLOv10
* [x] Face recognition with anti-impersonation
* [x] Text & QR code recognition
* [x] Currency recognition
* [ ] GPS + offline mapping integration
* [ ] Prototype into wearable smart glasses

---

## 👥 Team

* **Rishi Mehrotra** – Team Lead
* **Samarth Rana** – Hardware
* **Palak Bansal** – AI/ML
* **Shubhrika** – Edge Device
* **Priyanshi Jain** – Research

---

Would you like me to also include **screenshots and usage examples** (like terminal outputs and ESP32-CAM stream snapshots) in the README for better presentation on GitHub?
```
