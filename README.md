# Grade Change Intelligence

> **AI-Powered Decision Support System for Industrial Grade Transition Optimization**

## Overview

Grade Change Intelligence is an AI-driven decision support system designed to optimize **grade transition operations** in continuous manufacturing industries such as paper, steel, plastics, chemicals, and textiles.

The system predicts the likelihood of **off-spec production** during grade changes, recommends optimal machine operating conditions, validates recommendations using a Digital Twin simulation, and explains every prediction using Explainable AI (SHAP).

By leveraging historical production data and machine learning, the system enables industries to minimize waste, reduce transition time, improve product quality, and support operators with intelligent recommendations.

---

## Problem Statement

During grade transitions, manufacturing plants often experience:

- High material wastage
- Off-spec production
- Increased production cost
- Longer transition time
- Manual operator intervention
- Inconsistent process quality

Traditional approaches rely heavily on operator experience rather than predictive analytics, leading to inefficiencies and production losses.

---

## Proposed Solution

Grade Change Intelligence provides an integrated AI-based solution that:

- Predicts off-spec production before it occurs
- Recommends optimized machine setpoints
- Simulates recommendations using a Digital Twin
- Explains predictions using SHAP Explainable AI
- Displays results through an interactive dashboard
- Logs recommendations for future analysis

---

## Features

- Historical production data analysis
- Machine Learning-based prediction
- Intelligent recommendation engine
- Digital Twin simulation
- Explainable AI (SHAP)
- Interactive Streamlit dashboard
- Recommendation logging
- Production performance visualization

---

## Project Architecture

```
Historical Production Data
            │
            ▼
     Data Preprocessing
            │
            ▼
    Feature Engineering
            │
            ▼
 Machine Learning Model
        (XGBoost)
            │
            ▼
 Off-Spec Prediction
            │
            ▼
 Recommendation Engine
            │
            ▼
 Digital Twin Simulation
            │
            ▼
 SHAP Explainability
            │
            ▼
 Streamlit Dashboard
```

---

## Technology Stack

### Programming Language

- Python

### Machine Learning

- XGBoost
- Scikit-Learn

### Data Processing

- Pandas
- NumPy

### Explainable AI

- SHAP

### Visualization

- Matplotlib

### Dashboard

- Streamlit

### Database

- SQLite

### Model Storage

- Joblib

---

## Project Structure

```
Grade-Change-Intelligence/
│
├── data/
│   ├── historical_data.csv
│   ├── grades.csv
│   └── recommendations_log.db
│
├── dashboard/
│   └── app.py
│
├── explainability/
│   └── shap_explainer.py
│
├── models/
│   ├── train.py
│   ├── predict.py
│   ├── recommender.py
│   └── model.pkl
│
├── config.py
├── benchmark_optimizer.py
├── main.py
├── requirements.txt
└── HANDOVER.md
```

---

## Workflow

1. Load historical production data.
2. Perform preprocessing and feature engineering.
3. Train the machine learning model.
4. Predict off-spec production risk.
5. Generate optimized machine recommendations.
6. Validate recommendations using Digital Twin simulation.
7. Explain predictions using SHAP.
8. Display insights through the Streamlit dashboard.
9. Store recommendations in the SQLite database.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Grade-Change-Intelligence.git

cd Grade-Change-Intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the Model

```bash
python models/train.py
```

### Run Prediction

```bash
python models/predict.py
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Machine Learning Pipeline

```
Historical Data
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
Model Training
       │
       ▼
Prediction
       │
       ▼
Recommendation Generation
       │
       ▼
Digital Twin Validation
       │
       ▼
SHAP Explainability
```

---

## Expected Outcomes

- Reduced production waste
- Lower operational costs
- Faster grade transitions
- Improved product quality
- Data-driven decision making
- Increased manufacturing efficiency

---

## Future Enhancements

- Deep Learning-based prediction models
- Reinforcement Learning for autonomous optimization
- IoT sensor integration
- Real-time PLC/SCADA connectivity
- Cloud deployment
- Multi-factory optimization
- Predictive maintenance integration
- Mobile dashboard application

---

## Applications

- Paper Manufacturing
- Steel Industry
- Plastic Manufacturing
- Textile Industry
- Chemical Processing
- Food Processing
- Pharmaceutical Manufacturing

---

## Contributors

- Project Team
- Faculty Guide
- Institution Name

---

## License

This project is intended for academic and research purposes.

---

## Acknowledgements

Special thanks to the open-source community and the developers of:

- XGBoost
- Scikit-Learn
- SHAP
- Streamlit
- Pandas
- NumPy
- Matplotlib

whose tools made this project possible.
