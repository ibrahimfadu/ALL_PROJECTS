from flask import Flask,render_template,request
import joblib
model=joblib.load("model.pkl")
app=Flask(__name__)

@app.route("/",methods=["GET","POST"])
def home():
    return render_template('index.html')

from flask import Flask, render_template, request
import joblib

# Load the trained model
model = joblib.load("model.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template('index.html')

@app.route("/price", methods=["POST"])
def price():
    # Extract form data and convert to appropriate types
    car_name = request.form.get('car_name', 'Unknown Car')
    try:
        year = int(request.form.get('year', 0))
        selling_price = float(request.form.get('selling_price', 0.0))
        kms_driven = int(request.form.get('kms_driven', 0))
        owner = int(request.form.get('owner', 0))
        
        # Map categorical values to integers
        fuel_type = request.form.get('fuel_type')
        seller_type = request.form.get('seller_type')
        transmission = request.form.get('transmission')
        
        fuel_type_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
        seller_type_mapping = {"Individual": 0, "Dealer": 1}
        transmission_mapping = {"Manual": 0, "Automatic": 1}
        
        fuel_type_encoded = fuel_type_mapping.get(fuel_type, -1)  # Default to -1 if not found
        seller_type_encoded = seller_type_mapping.get(seller_type, -1)
        transmission_encoded = transmission_mapping.get(transmission, -1)
        price_sell=selling_price
        # Prepare data for prediction
        data = [year, selling_price, kms_driven, fuel_type_encoded, seller_type_encoded, transmission_encoded, owner]
        
        print("Data collected:", data)
        
        # Make prediction
        prediction = model.predict([data])[0] # Assuming the model expects a 2D array

        # Render the output template with prediction result
        return render_template("output.html", car_name=car_name, prediction=prediction,price_sell=price_sell)
    
    except ValueError as e:
        print(f"Error processing form data: {e}")
        return "Invalid input data", 400

if __name__ == "__main__":
    app.run(debug=True)


if __name__=="__main__":
    app.run(debug=True)














