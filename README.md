# 🚦 Buad | بُعد  
### Smart Road Accident & Surface Hazard Detection System  

**Buad** is an AI-powered system for detecting, analyzing, and logging road accidents and surface hazards (like cracks and potholes).  
It uses YOLOv11 models for real-time detection, a Vision-Language Model (VLM) for generating structured reports, and a Retrieval-Augmented Generation (RAG) chatbot for intelligent analytics.  

This system automates live road monitoring and contributes to building **safer and smarter cities**.  

> *Capstone Project – Data Science & Machine Learning Bootcamp, Tuwaiq Academy.*

---

##  Objectives  

Buad aims to create an intelligent road-safety platform that:  

- Detects **accidents** and **potholes** from **live camera feeds**.  
- Generates **structured VLM reports** describing severity and conditions.  
- Stores each event in a central **database**.  
- Allows **AI-powered retrieval** of insights via a RAG chatbot.  
- Visualizes all events on an interactive **dashboard**.  

> It transforms raw road footage into **real-time actionable intelligence** that supports faster response and safer roads.

---

## ⚙️ Features  

-  Real-time YOLO detection (accidents & potholes)  
-  Smart confirmation using voting & cooldown filtering  
-  Scene understanding via Vision-Language Model (VLM)  
-  Dashboard visualization with event metadata  
-  RAG chatbot for intelligent Q&A (text + voice)  
-  Audio-based interaction using GPT-4o-audio-preview  

---

## 🧩 Tech Stack  

| Layer | Technology |
|-------|-------------|
| **Backend** | FastAPI |
| **Frontend** | HTML/CSS, JavaScript (Web app) |
| **Computer Vision Modeling** | YOLOv11 (Accidents & Potholes) |
| **VLM** | Ollama(Accidents Scene analysis) |
| **RAG** | Agno, PostgreSQL, pgVector (DB), OpenAI (Embbedings, AudioGen) |
| **Storage** | Supabase Storage (media hosting) |
| **Python Libraries** | NumPy, Pandas, Matplotlib, OpenCV |

---



##  System Pipeline  

```text
Camera Feed 
   ↓
YOLO Detection 
   ↓
Event Confirmation (Voting + Cooldown + IoU)
   ↓
VLM Structured Report Generation
   ↓
Database Insertion & Supabase Upload
   ↓
Dashboard Visualization
   ↓
Vector DB Indexing → RAG Chatbot Retrieval
````

---

##  Project Structure

```text
project/
│
├── backend/
│   ├── app.py                  # Main FastAPI backend
│   ├── accident_detector.py    # YOLO accident model handler
│   ├── pothole_detector.py     # YOLO pothole model handler
│   ├── vlm_agent_wrapper.py    # GPT-4.1 Vision report generator
│   ├── db_insertion.py         # Database insertion logic
│   └── ...
│
├── templates/
│   ├── home.html               # Landing page
│   ├── dashboard.html          # Dashboard visualization
│   ├── chat.html               # RAG chatbot interface
│   ├── event_details.html      # Accident details view
│   ├── pothole_details.html    # Pothole details view
│   └── ...
│
├── static/
│   ├── style.css
│   └── assets/
│
├── models/                     # YOLO models
└── analysis/                   # EDA datasets
```

---

##  Datasets

Two dataset types were used in **Buad**:

1. **Exploratory Data Analytics (EDA)** datasets
   → Used for identifying key factors influencing accidents.
2. **Modeling datasets**
   → Used for training and validating YOLO models.

 [View Datasets](https://github.com/SarahAlshaikhmohamed/Road-Accidents-Detection-System/tree/main/datasets)

---

##  VLM Report Structure

Each confirmed accident image is processed by the Vision-Language Model (GPT-4.1 Vision) to produce a **structured JSON report** like this:

```json
{
  "Severity": "High",
  "Contact_level": "Severe Impact",
  "Weather": "Rainy",
  "Environment": "Highway",
  "Vehicles_count": 2,
  "Evidence": ["Damaged vehicles", "Smoke"],
  "Confidence": {"Accident": 0.92}
}
```

> This report is automatically stored in the database and shown in the dashboard’s “Accident/Pothole Details” view.

---

## 🧩 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SarahAlshaikhmohamed/Road-Accidents-Detection-System
cd Road-Accidents-Detection-System
```

###  2️⃣ Backend Setup

```bash
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac

pip install -r requirements.txt
```

###  3️⃣ Environment Variables (.env)

```bash
# Model Paths
MODEL_PATH=models/accident-yolov11n.pt
POTHOLE_MODEL_PATH=models/pothole-yolo.pt

# Database
DATABASE_URL=postgresql://user:password@host:5432/dbname

# Supabase Storage
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=YOUR_SERVICE_KEY
SUPABASE_BUCKET=Capstone

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
```

###  4️⃣ Run the Backend Server

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Access the web app:
 [http://localhost:8000/home](http://localhost:8000/home)

###  5️⃣ Raspberry Pi Live Detection

```bash
source venv/bin/activate
python3 test.py
```

Live Stream:
 [http://localhost:9000/stream](http://localhost:9000/stream)

---

## 🌐 API Endpoints

| Method | Endpoint                    | Description                                        |
| ------ | --------------------------- | -------------------------------------------------- |
| GET    | `/health`                   | Check system & models status                       |
| GET    | `/stream`                   | Stream real-time detections (accidents & potholes) |
| POST   | `/ingest/accident`          | Submit confirmed accident image & metadata         |
| POST   | `/ingest/pothole`           | Submit confirmed pothole image & metadata          |
| GET    | `/events`                   | List recent accidents                              |
| GET    | `/events/{event_id}`        | Retrieve metadata for one accident                 |
| GET    | `/events/{event_id}/report` | Get structured VLM report                          |
| GET    | `/potholes`                 | List recent pothole events                         |
| GET    | `/potholes/{event_id}`      | Retrieve one pothole’s metadata                    |
| POST   | `/chat`                     | Text-based chatbot query                           |
| POST   | `/audio-chat`               | Voice-based chatbot query                          |

---

## 📊 Results & Insights

Our BUAD | بُعد system successfully detects road accidents and potholes in real time using Raspberry Pi cameras integrated with AI models. 

-   *🚗 Accident Detection*
  - High detection accuracy on live video streams
  - False alarms reduced using voting logic + VLM validation
  - Events automatically stored in the database with image + location

-   *🕳️ Pothole Detection*
  - Potholes classified into 3 sizes:
    - Small
    - Medium
    - Large
  - Helps in prioritizing road repairs based on severity

-   *📍 GPS Integration*
  - Each event is tagged with *latitude and longitude*

-   *🖥️ Dashboard Insights*
  - Real time monitoring for all detected events
  - Searchable event history with images and metadata
  - Instant access to detailed AI generated reports

-   *🤖 RAG Chatbot Reporting*
  - Vector Database stores accidents insights
  - VLM generates a descriptive summary for every accident
  - Chatbot can answer queries related to event & pothole incidents

---

## 🔮 Recommendations & Future Work

* Deploy in real urban environments for continuous monitoring.
* Expand RAG database with more historical data.
* Retrain pothole model on broader datasets.
* Integrate automatic alert system for emergency response.
* Move toward full smart-city analytics dashboard.

---

## 👩‍💻 Contributors

- Sarah Alshaikhmohamed
- Lama Almoutiry
- Yahya Alhabboub
- Rawan Alsaffar
- Khalid Alzahrani

---

## 🎥 Presentation

[Project Demo](https://drive.google.com/drive/folders/1rGE39RDCr-Sov52OHaQ41kee8__eGXeD?usp=drive_link)

---

## Report

[Project Report](https://drive.google.com/file/d/1I_Fewk9PA1ZV3-SfbzoSNpM-LaOH1E6p/view?usp=sharing)

---

## License

This is licensed under the MIT License.
```
