from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model, features, and encoders
model = joblib.load('laptop_price_model.pkl')
features = joblib.load('features.pkl')
encoders = joblib.load('encoders.pkl')  # Load encoders for categorical columns

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        input_data = []

        for feature in features:
            value = request.form.get(feature)

            # Use label encoder if this feature is categorical
            if feature in encoders:
                le = encoders[feature]
                try:
                    value = le.transform([value])[0]
                except ValueError:
                    value = 0  # fallback for unknown category
            else:
                try:
                    value = int(value)
                except:
                    value = 0  # fallback for invalid numbers

            input_data.append(value)

        # Make prediction
        prediction = int(model.predict([np.array(input_data)])[0])

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5050)

