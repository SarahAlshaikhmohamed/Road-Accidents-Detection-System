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
├── backend/
│   ├── app.py
│   ├── accident_detector.py
│   ├── pothole_detector.py
│   ├── vlm_agent_wrapper.py
│   ├── db_insertion.py
│   └── ...
│
├── templates/
│   ├── home.html
│   ├── live.html
│   ├── dashboard.html
│   ├── chat.html
│   ├── event_details.html
│   └── ...
│
├── static/
│   ├── style.css
│   └── assets/
│
├── models/            # Model
└── analysis/          # EDA datasets
```

---

## Datasets

---

## Installation

---

## API Endpoints

---

## Results & Insights

---

## Recommendations & Future Work

---

## Contributors

- Sarah Alshaikhmohamed
- Lama Almoutiry
- Yahya Alhabboub
- Rawan Alsaffar
- Khalid Alzahrani
- 
---

## Presentation

[Project Demo]()

---

## License

This is licensed under the MIT License.


