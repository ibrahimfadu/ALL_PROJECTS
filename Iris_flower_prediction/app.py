from flask import Flask,render_template,request
import joblib


model=joblib.load('models/log_reg.pkl')

app=Flask(__name__)

@app.route("/",methods=["GET","POST"])
def home():
    if request.method=="POST":
        sl=float(request.form['Sepal_length'])
        sw=float(request.form['Sepal_width'])
        pl=float(request.form['Petal_length'])
        pw=float(request.form['Petal_width'])
        
        prediction = model.predict([[sl, sw, pl, pw]])[0]
        label = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        flower = label[prediction]

        return render_template('form.html',flower=flower)


    return render_template('form.html')

if __name__ == '__main__':
    app.run(port=5001, debug=True)  # Change 5001 to any port number you prefer
