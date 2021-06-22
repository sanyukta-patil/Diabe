import numpy as np
import flask
from flask import Flask, render_template,request,redirect
import pickle
#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET','POST'])
def index():
   return render_template("index.html")
#To use the predict button in our web-app
@app.route('/predict',methods=['GET','POST'])
def predict():
    # For rendering results on HTML GUI
        prg = request.form['prg']
        glc = request.form['gl']
        bp = request.form['bp']
        skt = request.form['sk']
        ins = request.form['ins']
        bmi = request.form['BMI']
        dpf = request.form['ped']
        age = request.form['age']

        prg = int(prg)
        glc = int(glc)
        bp = int(bp)
        skt = int(skt)
        ins = int(ins)
        bmi = float(bmi)
        dpf = float(dpf)
        age = int(age)
    #   int_features = [int(x) for x in request.form.values()]
        final_features = np.array([(prg, glc, bp, skt, ins, bmi, dpf, age)])
        sc=pickle.load(open('scaler.sav','rb'))
        final_features=sc.transform(final_features)

        prediction = model.predict(final_features)
        print(prg,glc,bp,skt,ins,bmi,dpf,age);
        #output = round(prediction[0], 2)
        prediction_text=""
        if prediction==[1]:
            return render_template("result.html", prediction_text='You have chance of diabetes')
        else:
           return render_template("result.html", prediction_text='Congratulations,You are safe')




if __name__ == "__main__":
    app.run(debug=True)
