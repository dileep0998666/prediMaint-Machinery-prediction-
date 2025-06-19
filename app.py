from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("rf_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    # Feature engineering
    df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
    df['Temperature difference [°C]'] = df['Process temperature [°C]'] - df['Air temperature [°C]']
    df['Temperature power [°C]'] = df['Temperature difference [°C]'] / df['Power']

    input_features = [
        'Air temperature [°C]',
        'Process temperature [°C]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Power',
        'Temperature difference [°C]',
        'Temperature power [°C]',
        'Type'
    ]

    prediction = model.predict(df[input_features])
    return jsonify({
        'prediction': int(prediction[0]),
        'status': "Machine Failure" if prediction[0] == 1 else "No Failure"
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
