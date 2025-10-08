# LunariSpectraPortal  
AI-Powered Exoplanet Detection and Classification Platform  

---

## Overview
LunariSpectraPortal is an artificial intelligence–based system developed to detect and classify exoplanets using open NASA mission data from **Kepler**, **K2**, and **TESS**.  
The project automates the identification of exoplanetary candidates through a machine learning pipeline that combines astrophysical feature engineering, model calibration, and a web-based interface for interactive predictions.

This repository includes the full workflow: data processing, model training, prediction, and deployment.

---

## Workflow Summary
1. **Data Preparation**  
   The harmonized dataset (`harmonized_exoplanets_clean.csv`) is located under the `data/` directory and merges Kepler, K2, and TESS catalogs with consistent formatting.

2. **Model Training**  
   Run the training pipeline located at:
   "src/nasa_main.py"

When executed, it:
- Generates feature-engineered datasets  
- Trains and calibrates LightGBM, XGBoost, or RandomForest models  
- Saves all outputs into the following:
  ```
  artifacts/
    ├── feature_importance.csv
    ├── train_config.json
    ├── label_encoder.pkl
    ├── metrics_raw.txt
    ├── metrics_calibrated.txt
  ```
- Stores the trained model (`exoplanet_lightgbm_calibrated.pkl`) inside:
  ```
  models/
  ```

3. **Backend Deployment**  
Once training is complete, start the Flask backend:
"python back.py"

The backend loads the `.pkl` model and exposes a REST API (`/predict`) for real-time classification.  
It automatically applies the same feature engineering used during training for full consistency.

4. **Frontend Interface**  
The web interface under `frontend/` communicates with the Flask API.  
Users can input parameters such as:
- `mission`
- `period_days`
- `duration_hrs`
- `depth_ppm`
- `radius_rearth`
- `st_rad`
- `impact`
- `teff`
- `logg`  

After clicking “Predict,” the system returns:
- **Predicted class** (Confirmed / Candidate / False Positive)  
- **Confidence level (% probability)**  

---

## Machine Learning Pipeline
The ML model uses a combination of astrophysical transformations and advanced algorithms to improve detection accuracy.

### Core Processing Steps
- Feature engineering:  
- `depth_norm = depth_ppm / st_rad²`  
- `depth_norm_x_logg = depth_norm × logg`  
- Logarithmic and ratio-based transformations  
- Outlier clipping and gravity filtering (`logg ≥ 3.8`)
- Model ensemble with LightGBM, XGBoost, and RandomForest
- Calibration using sigmoid scaling
- F1-based threshold tuning for CONFIRMED class

---

## Technology Stack
**Languages & Frameworks**
- Python 3.10  
- LightGBM, XGBoost, Scikit-learn, Pandas, NumPy  
- Flask (backend REST API), Flask-CORS  
- HTML, CSS, JavaScript (frontend UI)  



## How to Run
### 1. Train and Generate Artifacts
### 2. Start Backend Server

### 3. Access Web Interface
Open the `frontend/model.html` file in your browser to interact with the model through the API.

---

## Future Work
- Integrate time-series light curve data for temporal pattern detection  
- Add Explainable AI visualization for astrophysical interpretation  
- Deploy public version via Hugging Face Spaces or AWS Lambda  
- Continuous retraining for newly published exoplanet data  

---

## Author
Developed by **Muhammed Eren Kendir, Yasin Boran Ekşi, Berke Gürbüz, Berat Kırış, Furkan Kılıç** as part of the **NASA Space Apps Challenge 2025**.  
LunariSpectraPortal demonstrates how artificial intelligence can accelerate the discovery of exoplanets and make astrophysical data more accessible.

---

## License
This project is distributed under the MIT License.  
You are free to use, modify, and share it with attribution.
