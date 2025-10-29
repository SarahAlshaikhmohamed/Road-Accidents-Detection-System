# 🚦 Buad | بُعد

### Smart Roud Accidents Detection System

This project is building an AI-powered system for detecting, analyzing, and logging road accidents and hazards. Using YOLOv11 models. Detected images are analyzed through a Vision-Language Model (VLM), vectorized, and integrated into a Retrieval-Augmented Generation (RAG) agent for intelligent analytics. This system automates real-time road monitoring and forms a foundation for future smart city platforms.

*A Capstone Project built as part of the Data Science & Machine Learning Bootcamp at Tuwaiq Academy.*

---

## Objectives

**Buad** is an intelligent road safety monitoring platform that:

* Detects **accidents** and **potholes** from **live camera feeds**.
* Generates **structured reports** using a Vision-Language Model (VLM).
* Stores incidents in a database.
* Enables retrieval of insights through a **RAG chatbot**.
* Visualizes real-time and historical event data via a **dashboard**.

The system transforms raw road footage into **real time actionable intelligence**, supporting safer transportation infrastructure and faster emergency response.

---

## Features

*  Accident & pothole detection in real time.
*  Smart confirmation to avoid repeated false events.
*  Scene understanding using VLM.
*  Dashboard with metadata view.
*  RAG chatbot using embedded incident records.
*  Audio query support.

---

## Tech Stack

* Python: NumPy, Pandas, Matplotlib, Seaborn.
* Frontend: HTML/CSS, JavaScript (Web app).
* Backend: FastAPI (for API endpoints).
* Computer Vision Modeling: YOLO (Accidents & Potholes)
* Vision Language Modeling: Ollama(Accidents Scene analysis)
* RAG System: Agno, PostgreSQL, pgVector (DB), OpenAI (Embbedings, AudioGen).
* Storage: Supabase Storage (Media hosting).


---

## System Pipeline

```
Camera Feed → YOLO Detection → Event Confirmation → VLM Structured Report
            → Database Insert → Supabase Thumbnail Upload → Dashboard Display
            → Indexed in Vector DB → Chatbot Retrieval
```

---

## Project Structure

```
project/
│
├── backend/ # FastAPI Backend
│   ├── app.py # app functionality & routes 
│   ├── accident_detector.py # YOLOV11 accident detection handler 
│   ├── pothole_detector.py # YOLOV11 pothole detection handler 
│   ├── vlm_agent_wrapper.py # vlm reporting
│   ├── db_insertion.py # database insertion handler 
│   └── ...
│
├── templates/
│   ├── home.html # the landing page
│   ├── chat.html # RAG Chatbot 
│   ├── dashboard.html # The Accidents & Pothole Dashboard
│   ├── pothole_details.html # pothole details page
│   ├── event_details.html # accidents details page
│   └── ...
│
├── static/
│   ├── style.css # the css styling file
│   └── assets/ 
│
├── models/            # Model
└── analysis/          # EDA datasets
```

---

## Datasets

In our Buad Project flow, we used two different type of dataset:
- **Exploratory Data Analytics (EDA) Datasets**
- **Modeling Datasets**

For more details, **[Click Here](https://github.com/SarahAlshaikhmohamed/Road-Accidents-Detection-System/tree/main/datasets)** 

---

## Installation

1. Clone the Repository
git clone github.com/SarahAlshaikhmohamed/Road-Accidents-Detection-System

2. Backend Setup
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate  # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt

3- Setup Environment Variables
# Create .env file in project root

- Model Paths
echo "MODEL_PATH=models/accident-yolov11n.pt" >> .env
echo "POTHOLE_MODEL_PATH=models/pothole-yolo.pt" >> .env

-Database Setup
echo "DATABASE_URL=postgresql://user:password@host:5432/dbname" >> .env

- Supabase Storage
echo "SUPABASE_URL=https://xxxx.supabase.co" >> .env
echo "SUPABASE_KEY=YOUR_SERVICE_KEY" >> .env
echo "SUPABASE_BUCKET=Name" >> .env

- API Keys
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

4- Run the Backend Server
uvicorn backend.app:app --host 0.0.0.0 --port 8000

📍 API Available at:

http://localhost:8000/home

5- Run Raspberry Pi Camera Detection
source venv/bin/activate   # activate same env
python3 pi-camera.py

Live stream preview:

http://localhost:9000/stream

Detections are automatically submitted to the system and stored.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Checks if the system and models are running correctly |
| GET | /stream | Streams live video with real-time detection (accidents & potholes) |
| POST | /ingest/accident | Receives a confirmed accident image + detection details for storing |
| POST | /ingest/pothole | Receives a confirmed pothole image + detection details for storing |
| GET | /events | Lists the most recent accident events |
| GET | /events/{event_id} | Retrieves metadata about a specific accident |
| GET | /events/{event_id}/report | Returns the generated incident report |
| GET | /potholes | Lists the most recent pothole events |
| GET | /potholes/{event_id} | Retrieves metadata about a specific pothole |
| POST | /chat | Text-based chatbot for querying stored information |
| POST | /audio-chat | Voice-based chatbot that replies with audio |
---

## Results & Insights

Our BUAD | بُعد system successfully detects road accidents and potholes in real time using Raspberry Pi cameras integrated with AI models. 

###  Key Results

-   *Accident Detection*
  - High detection accuracy on live video streams
  - False alarms reduced using voting logic + VLM validation
  - Events automatically stored in the database with image + location

-   *Pothole Detection*
  - Potholes classified into 3 sizes:
    - Small
    - Medium
    - Large
  - Helps in prioritizing road repairs based on severity

-   *GPS Integration*
  - Each event is tagged with *latitude and longitude*

-   *Dashboard Insights*
  - Real time monitoring for all detected events
  - Searchable event history with images and metadata
  - Instant access to detailed AI generated reports

-   *RAG Chatbot Reporting*
  - Vector Database stores accidents insights
  - VLM generates a descriptive summary for every accident
  - Chatbot can answer queries related to event & pothole incidents

---

## Recommendations & Future Work
- Deploy the system in real environments.
- Expand data and analytics features.
- Move toward an integrated, smart city management platform.
---

## Contributors

- Sarah Alshaikhmohamed
- Lama Almoutiry
- Yahya Alhabboub
- Rawan Alsaffar
- Khalid Alzahrani

---

## Presentation

[Project Demo]()

---

## License

This is licensed under the MIT License.


