#  The Kairos Engine

> **AI-Powered Cognitive Autopilot for BVLOS Medical Drone Operations in High-Altitude Himalayan Terrain**

[![Model](https://img.shields.io/badge/LLM-Gemma--4--E2B--GGUF-blue.svg)](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
[![Risk Predictor](https://img.shields.io/badge/ML-XGBoost--Classifier-green.svg)](https://xgboost.readthedocs.io/)
[![Accuracy](https://img.shields.io/badge/Model_Accuracy-97.40%25-brightgreen.svg)]()
[![SLA](https://img.shields.io/badge/Emergency_Re--planning-sub--2s-red.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)]()

---

##  Executive Summary

Deliveries of life-critical medical supplies (such as **Oxytocin**, blood plasma, and antivenoms) to remote health clinics in the Nepalese Himalayas face severe threats: sudden mountain storms, extreme headwinds, and rapid high-altitude battery drain.

**The Kairos Engine** is an autonomous cognitive autopilot system that combines **large language model reasoning (Gemma 4 E2B GGUF)** with **physics-based machine learning (XGBoost)** to make real-time, life-critical flight routing and dynamic emergency decisions in sub-2 seconds.

---

##  Key Highlights & Statistics

| Metric | Benchmark Value | Description |
|---|---|---|
|  **Primary Reasoning Core** | **Gemma 4 E2B (4-bit Q4_K_M GGUF)** | Powered by `llama-cpp-python` C++ runtime (~2.89 GB) |
|  **Risk Prediction Accuracy** | **97.40% Test Accuracy** | Trained on 5,000 physics-informed Himalayan flight logs |
|  **Emergency Decision SLA** | **< 2.0 Seconds** | Meets strict real-time aviation safety constraints |
|  **GIS Visualization** | **Interactive Folium HTML** | Generates real-time 3D flight maps (`flight_plan_map.html`) |
|  **Max Operational Ceiling** | **5,500 Meters AMSL** | Optimized for Thorong La Pass & Annapurna Circuit corridors |

---

##  System Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                         THE KAIROS ENGINE CLI                          │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                             [ main.py Entry ]
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                     KAIROS MASTER ORCHESTRATOR                         │
 └───────────────┬───────────────────┬───────────────────┬────────────────┘
                 │                   │                   │
                 ▼                   ▼                   ▼
 ┌───────────────────────┐ ┌───────────────────┐ ┌───────────────────────┐
 │   Gemma 4 GGUF Core   │ │   XGBoost Risk    │ │   Weighted Dijkstra   │
 │ (llama-cpp C++ Engine)│ │    Classifier     │ │   Energy Pathfinder   │
 └───────────────┬───────┘ └─────────┬─────────┘ └───────────┬───────────┘
                 │                   │                       │
                 ▼                   ▼                       ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                       TOOL EXECUTOR & GIS MAP                          │
 │       (SRTM DEM Elevation, CAAN Airspace, Folium Map Renderer)         │
 └────────────────────────────────────────────────────────────────────────┘
```

---

##  Core Features

### 1.  Gemma 4 Native Function Calling DSL
The engine uses Google DeepMind's **Gemma 4 E2B** model with native tool-calling syntax (`<|tool_call|func_name{...}<tool_call|>`) to autonomously select and orchestrate aviation tools.

### 2.  XGBoost Crash Risk Predictor (`assess_risk`)
Predicts crash probability based on 4 telemetry variables:
- **Rule 1 (Mountain RTL):** Low battery + high altitude climbing over ridges.
- **Rule 2 (Headwind Exhaustion):** Severe headwinds (>15 m/s) with low battery.
- **Rule 3 (Safe Divert):** High-probability safe contingency landings at pre-mapped LZs.

### 3.  Interactive Folium GIS Map Generator
Dynamically renders a high-contrast OpenStreetMap visualizer (`flight_plan_map.html`) showing:
- **Green Marker:** Pokhara Base Origin (`28.2096, 83.9856`)
- **Red Marker:** Muktinath Clinic Destination (`28.8167, 83.8667`)
- **Orange Markers:** Contingency Landing Zones (`LZ_01`, `LZ_02`, `LZ_03`)
- **Red Corridor Line:** Energy-optimized BVLOS flight route

---

##  Quickstart Guide

### 1. Installation
```powershell
# Run automated setup
.\setup_env.ps1
```

### 2. Train XGBoost Model
```powershell
python build_crash_predictor.py
```

### 3. Run Kairos Engine Mission Demo
```powershell
python main.py
```

### 4. Run Automated Test Suite
```powershell
python run_tests.py
```

---

## Hackathon Presentation Summary for Judges

- **What problem does it solve?** Prevents high-altitude BVLOS drone crashes during life-critical medical deliveries in Nepal.
- **How does LLM + ML work together?** Gemma 4 acts as the high-level cognitive brain (reasoning & tool selection), while XGBoost acts as the fast, deterministic flight safety verifier.
- **Is it fast?** Yes! Decision latency is sub-2 seconds for mid-flight anomalies.

---

*Built with ❤️ for Himalayan Medical BVLOS Logistics.*
