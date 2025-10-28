# 🚦 Buad | بُعد

### Smart Road Monitoring System

*A Capstone Project built as part of the Data Science & Machine Learning Bootcamp at Tuwaiq Academy*

---

## Introduction

**Buad** is an intelligent road-safety monitoring platform that:

* Detects **accidents** and **potholes** from **live camera feeds**
* Generates **structured reports** using a Vision-Language Model (VLM)
* Stores incidents in a database
* Enables retrieval of insights through a **RAG chatbot**
* Visualizes real-time and historical event data via a **dashboard**

The system transforms raw road footage into **real time actionable intelligence**, supporting safer transportation infrastructure and faster emergency response.

---

## Project Overview

Buad integrates multiple AI components into one seamless platform:

| Feature              | Description                                                     |
| -------------------- | --------------------------------------------------------------- |
| Accident Detection   | YOLO-based detection with multi-frame event confirmation        |
| Pothole Detection    | Hazard detection with size classification                       |
| VLM Report Generator | Generates JSON reports: severity, vehicles count, weather, etc..|
| Knowledge Retrieval  | Chatbot answering  queries using vector search                  |
| Live Monitoring      | Camera streaming with start/stop control                        |
| Event History        | Dashboard listing archived incidents with thumbnails            |
| Supabase Storage     | Hosting thumbnails with signed URLs                             |

---

## Tech Stack

### Backend

* FastAPI 
* YOLO (Accidents & Potholes)
* Vision-Language Model (Accidents Scene analysis)
* PostgreSQL + pgVector (DB)
* Supabase Storage (Media hosting)

### Frontend

* HTML, CSS, JavaScript

### AI / NLP

* OpenAI embeddings + hybrid vector search
* RAG agents for text & audio queries

---

## System Pipeline

```
Camera Feed → YOLO Detection → Event Confirmation 
→ VLM Structured Report → Database Insert
→ Supabase Thumbnail Upload → Dashboard Display
→ Indexed in Vector DB → Chatbot Retrieval
```

---

## Repository Structure

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

## Core Features

*  Accident & pothole detection in real time
*  Smart confirmation to avoid repeated false events
*  Scene understanding using VLM
*  Dashboard with metadata view
*  RAG chatbot using embedded incident records
*  Audio query support


## Contributors
- Sarah Alshaikhmohamed
- Lama Almoutiry
- Yahya Alhabboub
- Rawan Alsaffar
- Khalid Alzahrani
---

## Acknowledgments

Developed as part of the
**Tuwaiq Academy Data Science & Machine Learning Bootcamp**


