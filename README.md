# 🛠️ PrediMaint – Machinery Predictive Maintenance

An AI-powered solution to forecast machine failures and optimize maintenance schedules using machine learning models.

---

## 🔍 Overview

PrediMaint processes real-time or historical sensor data from industrial machines (temperature, vibration, pressure, etc.) to predict when maintenance is needed — *before breakdowns occur*. This helps reduce downtime, save costs, and improve operational efficiency.

---

## ⚙️ Tech Stack

- **Language:** Python  
- **ML Frameworks:** scikit-learn, XGBoost, TensorFlow / PyTorch  
- **Data Handling:** pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Deployment:** Flask for APIs

---

## 🌐 Live Demo

🔗 [Click to View Live Project](https://predimaint-machinery-prediction.onrender.com/)



---

## 🛠️ Installation & Setup

```bash
# 1. Clone the repo
git clone https://github.com/dileep0998666/prediMaint-Machinery-prediction-
cd prediMaint-Machinery-prediction-

# 2. Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the full pipeline
python src/train_model.py --data-path data/processed.csv
python src/predict.py --model-path models/model.pkl --input data/new_input.csv
