from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# Appliance power ratings (watts)
appliances = {
    "Fan": 75,
    "Air Conditioner": 1500,
    "Television": 100,
    "Refrigerator": 180,
    "Washing Machine": 500,
    "LED Bulb": 12,
    "Laptop": 65,
    "Water Heater": 2000
}

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:
            # Get user inputs
            previous = float(request.form["previous_usage"])
            appliance = request.form["appliance"]
            hours = float(request.form["usage_hours"])

            # Validate inputs
            if previous < 0 or hours <= 0 or hours > 24:
                raise ValueError("Enter valid usage values.")

            if appliance not in appliances:
                raise ValueError("Select a valid appliance.")

            # Get appliance power
            power = appliances[appliance]

            # Prepare input for prediction
            input_data = pd.DataFrame([{
                "previous_usage": previous,
                "appliance_power": power,
                "usage_hours": hours
            }])

            # Predict electricity consumption
            prediction = model.predict(input_data)[0]
            print("Raw Prediction:", prediction)

            prediction = max(0, float(prediction))
            prediction = round(prediction, 4)

        except ValueError as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)