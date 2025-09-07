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

## ⚙️ Data Generation & Pipeline

**Please Note:** The `.csv` files included in this repository (`live_spaceweather.csv`, `cloud_forecast.csv`) are **static examples** and will not update automatically. They are provided so that the demo can be run immediately.

A real-world, production version of this application would require an automated data pipeline to fetch fresh data. This would typically involve:

1.  **A separate Python script** (e.g., `fetch_data.py`).
2.  **API calls** to fetch the latest space weather data from [NOAA SWPC](https://www.swpc.noaa.gov/) and cloud cover forecasts from a weather provider like the [Finnish Meteorological Institute (FMI)](https://en.ilmatieteenlaitos.fi/open-data).
3.  **Data processing** to calculate the required features (rolling averages, lags, etc.).
4.  **Running this script on a schedule** (e.g., every hour) using a tool like Cron (on Linux/macOS) or Windows Task Scheduler. The script would then overwrite the `.csv` files with fresh data for the Streamlit app to consume.

---

## 🚀 Installation & Usage

It is highly recommended to run this project in a virtual environment to avoid conflicts with other Python packages.

**1. Clone the repository**
```bash
git clone [https://github.com/](https://github.com/)nehannin/auroraforecastRovaniemi
cd auroraforecastRovaniemi
```

**2. Create and activate a virtual environment**
```bash
# Create the environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
# venv\Scripts\activate
```

**3. Install the required dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**
```bash
streamlit run app.py
```

**5. Open the app in your browser**

Streamlit will provide a local URL, typically:
👉 http://localhost:8501

---

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

- Metrics (validated on a chronological test set):
    - **Accuracy:** 99.9%
    - **F1-score (aurora class):** 0.978
    - **ROC AUC:** 1.000

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

