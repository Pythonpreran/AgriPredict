# AgriPredict 🌾
An intelligent price forecasting solution for farmers and policymakers.

## 🚀 Elevator Pitch
Every year, over 10,000 farmers in India die by suicide, often due to unpredictable price drops, weather shocks, or disasters. AgriPredict aims to empower farmers with foresight — helping them make better financial decisions using AI-driven price forecasting.

## 📌 Problem Statement
Agriculture is one of the most critical yet riskiest occupations globally. Most farmer losses stem from a lack of reliable, region-specific price predictions. This project tackles that problem by providing hyper-local, data-driven forecasts tailored for different regions across India.

## 🧠 Key Features
- 🔄 **Customized Forecasting for 100 Regional Centers**: India’s vastness means price trends vary greatly. AgriPredict trains separate models for 100 simulated centers to reflect realistic, region-specific market behavior.
- 📉 **SARIMA for Seasonal Trends**: SARIMA (Seasonal AutoRegressive Integrated Moving Average) models help capture repeating seasonal patterns in agricultural prices and also detect sudden drops caused by shocks.
- 🔺 **Boosted Accuracy with Gradient Boosting**: XGBoost is layered on top of SARIMA predictions to refine and improve forecasting accuracy.
- 📊 **Core Prediction System Implemented**: Built using Flask, this web app allows users to choose a commodity and center, then view its upcoming price trends via dynamic graphs.
- ⚠️ **Future Expansion Plans**: Weather and disaster forecasts are planned to be integrated soon to enhance prediction accuracy.

## 🧪 Dataset
The dataset is artificially generated but mimics real-world agricultural price behavior — including seasonal patterns and abrupt fluctuations caused by simulated disasters.

## 🛠️ Tech Stack
- Python
- Flask
- Pandas, NumPy
- SARIMAX (statsmodels)
- XGBoost
- Matplotlib
- HTML/CSS (for frontend)

## 📷 Screenshots
![image](https://github.com/user-attachments/assets/17fbf12c-3508-4d66-bd11-e66fbfee756c)
![image](https://github.com/user-attachments/assets/41161c4a-31de-4f02-99d4-0fbd7e8b38ea)
![image](https://github.com/user-attachments/assets/df49342c-76a4-4aa4-b5d4-e6696f5d6bbb)
![image](https://github.com/user-attachments/assets/21adc2f0-9c4c-471e-9414-132c1f16d958)






