from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = joblib.load('xgb_model_final.pkl')

# Airport/landmark coordinates
JFK = (-73.7781, 40.6413)
LGA = (-73.8700, 40.7769)
EWR = (-74.1745, 40.6895)
MET = (-73.9760, 40.7831)
WTC = (-74.0132, 40.7128)

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Optional: if you send pickup_datetime instead of year/month/day/hour/weekday
    # dt = datetime.strptime(data['pickup_datetime'], "%Y-%m-%d %H:%M:%S")
    # pickup_year = dt.year
    # pickup_month = dt.month
    # pickup_day = dt.day
    # weekday = dt.weekday()
    # pickup_hour = dt.hour

    # Calculate airport/landmark distances
    jfk_dist = haversine_np(data['pickup_longitude'], data['pickup_latitude'], JFK[0], JFK[1])
    lga_dist = haversine_np(data['pickup_longitude'], data['pickup_latitude'], LGA[0], LGA[1])
    ewr_dist = haversine_np(data['pickup_longitude'], data['pickup_latitude'], EWR[0], EWR[1])
    met_dist = haversine_np(data['pickup_longitude'], data['pickup_latitude'], MET[0], MET[1])
    wtc_dist = haversine_np(data['pickup_longitude'], data['pickup_latitude'], WTC[0], WTC[1])
    
    # Prepare features in the order your model expects
    features = np.array([[
        data['pickup_longitude'],
        data['pickup_latitude'],
        data['dropoff_longitude'],
        data['dropoff_latitude'],
        data['passenger_count'],
        data['pickup_year'],
        data['pickup_month'],
        data['pickup_day'],
        data['weekday'],
        data['pickup_hour'],
        data['distance'],
        jfk_dist,
        lga_dist,
        ewr_dist,
        met_dist,
        wtc_dist
    ]])

    prediction = model.predict(features)
    return jsonify({'predicted_fare': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
