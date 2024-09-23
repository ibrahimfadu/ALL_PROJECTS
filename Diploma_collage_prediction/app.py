from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve and validate form data
        rank = int(request.form.get('rank', '0'))
        branch = request.form.get('branch')
        category = request.form.get('category')

        if not branch or not category:
            return redirect(url_for('home'))

        # Check if file exists
        file_path = 'data/comple_n1.csv'
        if not os.path.exists(file_path):
            return "Error: Data file not found.", 500

        # Load the CSV file into DataFrame
        df_copy = pd.read_csv(file_path)

        # Check if required columns are in DataFrame
        required_columns = {"BRANCH", "CATEGORIES", "RANKING", "COLLEGE"}
        if not required_columns.issubset(df_copy.columns):
            return "Error: Data file format is incorrect.", 500

        # Filter DataFrame
        x = df_copy[
            (df_copy["BRANCH"] == branch) &
            (df_copy["CATEGORIES"] == category) &
            (df_copy["RANKING"] > rank) &
            (df_copy["RANKING"] != 0)
        ][["COLLEGE", "RANKING"]]

        # Check if the result is empty
        if x.empty:
            return "No results found for the given criteria."

        # Find the college with the minimum ranking
        min_ranking_idx = x["RANKING"].idxmin()  # Find the index of the minimum ranking
        result = x.loc[min_ranking_idx, "COLLEGE"]

        return render_template("outputfile.html", Output=result)
    
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
