# app.py
from flask import Flask,render_template,request
import joblib
from PIL import Image
import numpy as np
import pandas as pd
import io
app=Flask(__name__) 

model=joblib.load("letters/model_kkn.pkl")

def process(path):
    image= Image.open(io.BytesIO(path.read())).convert("L").resize((28,28))#gray scaling using convert, resize image
    image_rev=np.invert(image)#change background white to black
    final_img=image_rev.flatten()
    return final_img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict():
    
    file = request.files['file']
    final=process(file)
    prediction=model.predict([final])
    return render_template("index.html",prediction=prediction)


if __name__=="__main__":
    app.run(debug=True)