from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and features
model = joblib.load('laptop_price_model.pkl')
features = joblib.load('features.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = []

        for feature in features:
            value = request.form.get(feature)
            try:
                input_data.append(int(value))
            except:
                input_data.append(0)

        prediction = model.predict([np.array(input_data)])
        return render_template('index.html', prediction=int(prediction[0]))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


