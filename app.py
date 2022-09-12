import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json["data"]
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
#     pred=model.predict(new_data)
#     print(pred[0])
#     return jsonify(pred[0])

@app.route("/predict",methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text='The Medical Insurance Cost Prediction is: {}'.format(output))
#@app.route('/predict',methods=['POST'])
#def predict():



if __name__=='__main__':
    app.run(debug=True)

