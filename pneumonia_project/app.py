import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Global variables to hold our model and preprocessing tools
model = None
scaler = None
train_columns = None

def train_model():
    """Trains the model on startup using the exact methodology from your writeup."""
    global model, scaler, train_columns
    print("Loading data and training model...")
    
    # 1. Load the dataset
    df = pd.read_csv('pneumonia_project/clinical_pneumonia_dataset.csv')
    
    # Create binary target variable (1 for pneumonia, 0 for others)
    df['target'] = (df['true_label'] == 'pneumonia').astype(int)
    
    # 2. Select Features matching the HTML form
    X = df[['fever', 'tachycardia', 'crackles', 'oxygen_saturation', 'wbc_count', 'chest_xray_result']]
    y = df['target']
    
    # 3. Preprocess Categorical Data (One-Hot Encoding)
    X_encoded = pd.get_dummies(X, columns=['chest_xray_result'])
    train_columns = X_encoded.columns # Save these to align user input later
    
    # 4. Scale Continuous Features
    scaler = StandardScaler()
    continuous_cols = ['oxygen_saturation', 'wbc_count']
    X_encoded[continuous_cols] = scaler.fit_transform(X_encoded[continuous_cols])
    
    # 5. Apply SMOTE for class imbalance (as per your writeup)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
    
    # 6. Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)
    print("Model trained successfully and is ready for predictions!")

@app.route('/')
def home():
    # Serves the HTML file from the templates folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the frontend fetch request
        data = request.json
        
        # 1. Format the incoming data into a DataFrame
        input_data = {
            'fever': [int(data['fever'])],
            'tachycardia': [int(data['tachycardia'])],
            'crackles': [int(data['crackles'])],
            'oxygen_saturation': [float(data['oxygen_saturation'])],
            'wbc_count': [float(data['wbc_count'])],
            'chest_xray_result': [data['xray_result']]
        }
        df_input = pd.DataFrame(input_data)
        
        # 2. One-hot encode the input
        df_input_encoded = pd.get_dummies(df_input, columns=['chest_xray_result'])
        
        # 3. Align input columns with training columns (fill missing categories with 0)
        for col in train_columns:
            if col not in df_input_encoded.columns:
                df_input_encoded[col] = 0
        df_input_encoded = df_input_encoded[train_columns]
        
        # 4. Scale the continuous inputs
        df_input_encoded[['oxygen_saturation', 'wbc_count']] = scaler.transform(df_input_encoded[['oxygen_saturation', 'wbc_count']])
        
        # 5. Get the probability of the Positive class (Pneumonia)
        prob = model.predict_proba(df_input_encoded)[0, 1]
        
        # 6. Apply your custom project threshold of 0.35
        is_high_risk = bool(prob >= 0.35)
        
        # Return the result back to the HTML JavaScript
        return jsonify({
            'success': True,
            'is_high_risk': is_high_risk,
            'prediction': 'High Risk of Pneumonia' if is_high_risk else 'Low Risk (Clear)',
            'probability': f"{prob * 100:.1f}%"
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Train the model before starting the server
    train_model()
    # Run the Flask app on localhost
    app.run(debug=True, port=5000)