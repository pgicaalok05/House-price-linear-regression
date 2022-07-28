import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app1 = Flask(__name__)
model = pickle.load(open('house-price.pkl','rb')) 


@app1.route('/')
def home():
  
    return render_template("index1.html")
  
@app1.route('/predict',methods=['GET'])

def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    SqFt = float(request.args.get('SqFt'))
    
    prediction = model.predict([[SqFt]])
    
        
    return render_template('index1.html', prediction_text='Regression Model  has predicted House Price for given Square Foot Area is : {}'.format(prediction))


if __name__ == "__main__":
  app1.run(debug = True)
