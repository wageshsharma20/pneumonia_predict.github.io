import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Global variables
model = None
scaler = None
train_columns = None


def train_model():
    global model, scaler, train_columns
    print("Loading data and training model...")

    df = pd.read_csv('clinical_pneumonia_dataset.csv')

    # Target
    df['target'] = (df['true_label'] == 'pneumonia').astype(int)

    # Features
    X = df[['fever', 'tachycardia', 'crackles',
            'oxygen_saturation', 'wbc_count', 'chest_xray_result']]
    y = df['target']

    # One-hot encoding
    X_encoded = pd.get_dummies(X, columns=['chest_xray_result'])
    train_columns = X_encoded.columns

    # Scaling
    scaler = StandardScaler()
    continuous_cols = ['oxygen_saturation', 'wbc_count']
    X_encoded[continuous_cols] = scaler.fit_transform(X_encoded[continuous_cols])

    # SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

    # Model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)

    print("Model trained successfully!")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if data is None:
            return jsonify({'success': False, 'error': 'No data received'})

        input_data = {
            'fever': [1 if data.get('fever') == 'Yes' else 0],
            'tachycardia': [1 if data.get('tachycardia') == 'Yes' else 0],
            'crackles': [1 if data.get('crackles') == 'Yes' else 0],
            'oxygen_saturation': [float(data.get('oxygen_saturation', 0))],
            'wbc_count': [float(data.get('wbc_count', 0))],
            'chest_xray_result': [data.get('xray_result', 'Normal')]
        }

        df_input = pd.DataFrame(input_data)

        # One-hot encode
        df_input_encoded = pd.get_dummies(df_input, columns=['chest_xray_result'])

        # Align columns
        for col in train_columns:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0

        df_input_encoded = df_input_encoded[train_columns]

        # Scale
        df_input_encoded[['oxygen_saturation', 'wbc_count']] = scaler.transform(
            df_input_encoded[['oxygen_saturation', 'wbc_count']]
        )

        # Prediction
        prob = model.predict_proba(df_input_encoded)[0, 1]
        is_high_risk = bool(prob >= 0.35)

        return jsonify({
            'success': True,
            'is_high_risk': is_high_risk,
            'prediction': 'High Risk of Pneumonia' if is_high_risk else 'Low Risk (Clear)',
            'probability': f"{prob * 100:.1f}%"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    train_model()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
