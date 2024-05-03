from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model_path = 'restaurant_rating_prediction.joblib'  # Update with the actual path to your model file
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    rating_color = request.form['rating_color']
    has_table_booking = request.form['has_table_booking']
    has_online_delivery = request.form['has_online_delivery']
    locality = request.form['locality']
    city = request.form['city']
    
    # Convert user inputs into DataFrame for prediction
    input_data = pd.DataFrame([[rating_color, has_table_booking, has_online_delivery, locality, city]],
                              columns=['Rating color', 'Has Table booking', 'Has Online delivery', 'Locality', 'City'])
    
    # Make prediction using the model
    predicted_rating = model.predict(input_data)
    
    return render_template('result.html', predicted_rating=predicted_rating[0])

if __name__ == '__main__':
    app.run(debug=True)
