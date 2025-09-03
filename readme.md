# 🌌 Aurora Forecast • Rovaniemi

**Aurora Borealis visibility prediction demo** using space weather data, cloud forecasts, and a trained Random Forest machine learning model.  
This project was built as part of a **Bachelor's thesis in Information and Communication Engineering (Lapland UAS)**.

---

## 📖 Overview

The application predicts the **probability of aurora visibility in Rovaniemi**.  

- ✅ Uses **NOAA & FMI space weather features** (e.g. `bz_gsm`, `kp_index`, rolling averages, lags).  
- ✅ Trained with a **Random Forest Classifier** and evaluated with calibration + metrics.  
- ✅ Considers **cloud forecasts** for realistic visibility.  
- ✅ Interactive **Streamlit web app** with map, site selection, and prediction UI.  
- ✅ Clean styling with **custom CSS** and background visuals.  

---

## 📂 Project Structure

.
├── app.py # Main Streamlit app
├── assets/
│ └── styles.css # Custom styles (UI improvements)
├── aurora_rf_model.pkl # Trained Random Forest model + features
├── live_spaceweather.csv # Current space weather features (hourly, UTC)
├── cloud_forecast.csv # Cloud cover forecast (per site, hourly)
├── latest_spaceweather.json # Raw snapshot (fallback if CSV missing)
├── model_report.json # Model performance report (ROC AUC, F1, feature importance)
├── requirements.txt # Python dependencies
└── data/ # Historical datasets (not required to run app)
├── fmi_noaa_aurora_2023.csv
└── fmi_noaa_combined_2023.csv

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-repo>/aurora-forecast
cd aurora-forecast
```

2. Create environment & install requirements
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app
```bash
streamlit run app.py
```

4. Open in browser

Streamlit will give you a local URL, usually:
👉 http://localhost:8501

📊 Data Sources

#### Space Weather:
- NOAA SWPC & FMI open datasets.
- Features include bz_gsm, kp_index, rolling 3h averages, and lag features.

#### Cloud Forecast:
- Simplified hourly cloud coverage CSV (per location).
- Effective aurora probability = model prediction × (1 − cloud_cover).

#### Locations:
- 🌆 Arktikum Park (city lights)
- 🌃 Ounasvaara Scenic Point (semi-dark sky)
- 🌑 Vikaköngäs Rapids (darkest site, best visibility)

📈 Model Information

- Algorithm: Random Forest Classifier

- Calibration: Platt scaling (improved probability outputs)

- Metrics (from model_report.json):
    - ROC AUC: ~0.92
    - F1-score: ~0.78

Top features (feature importance):
- kp10
- bz_gsm
- kp10_3h_avg
- bz_gsm_lag1h

🎨 UI Features

- Dark sky theme with custom CSS styling
- Map with selectable observation sites
- Date/time picker aligned with prediction button
- Cloud-adjusted probabilities
- Interactive charts & feature explanations

⚙️ Files You Need to Run the Demo

Minimum required:

- app.py
- aurora_rf_model.pkl
- live_spaceweather.csv
- cloud_forecast.csv
- requirements.txt

Optional (extra info, better UX):

- assets/styles.css
- latest_spaceweather.json
- model_report.json

📌 Notes

This app is a demo for educational purposes.

Predictions are not an official forecast and should not replace FMI/NOAA services.

To extend:
- Replace live_spaceweather.csv with a real-time API fetch.
- Add more observation sites.
- Train with additional years of data for better generalization.

## 🔄 Data & Model Pipeline

![Aurora Forecast Pipeline](pipeline.png)
