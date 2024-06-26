import numpy as np
from flask import Flask, request, render_template
import pickle

# Load the pickled model
model = pickle.load(open("model.pkl", "rb"))

# Create Flask application
app = Flask(__name__)

# Define routes
@app.route("/")
def home():
    return render_template("index2.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Obtain input features from the form and convert to float
    BHK = float(request.form["BHK"])
    Size = float(request.form["Size"])
    Bathroom = float(request.form["Bathroom"])

    # Retrieve categorical variables from the form
    AreaType = request.form.get("AreaType", "")
    City = request.form.get("City", "")
    FurnishingStatus = request.form.get("FurnishingStatus", "")
    TenantPreferred = request.form.get("TenantPreferred", "")
    PointOfContact = request.form.get("PointOfContact", "")

    # Define a mapping of categorical feature names to their values
    categorical_features = {
        'AreaType': 'Area Type_' + AreaType,
        'City': 'City_' + City,
        'FurnishingStatus': 'Furnishing Status_' + FurnishingStatus,
        'TenantPreferred': 'Tenant Preferred_' + TenantPreferred,
        'PointOfContact': 'Point of Contact_' + PointOfContact
    }

    # Retrieve coefficients and intercept from the model
    coefficients = model.coef_
    intercept = model.intercept_

    # Calculate the predicted rent using coefficients and intercept
    Rent = intercept

    # Add contributions from numerical features (BHK, Size, Bathroom)
    Rent += coefficients[0] * BHK  # Coefficient for BHK
    Rent += coefficients[1] * Size  # Coefficient for Size
    Rent += coefficients[2] * Bathroom  # Coefficient for Bathroom

    # Add contributions from categorical features
    for feature_name, feature_value in categorical_features.items():
        if feature_value in model.feature_names_in_:
            coef_index = np.where(model.feature_names_in_ == feature_value)[0][0]
            Rent += coefficients[coef_index]

    # Pass the predicted Rent to the HTML template
    return render_template("index2.html", prediction_result=Rent)

if __name__ == "__main__":
    # Run the Flask app on localhost with port 5000
    app.run(debug=True, host="127.0.0.1", port=5000)