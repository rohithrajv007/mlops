from flask import Flask, request, jsonify
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
app = Flask(__name__)
X,y= load_iris(return_X_y=True)
clf= LogisticRegression(random_state=0, max_iter=200).fit(X, y)

@app.route('/',methods=['GET' ,'POST'])
def predict():
    if request.method == 'GET':
        return "Welcome to the Iris Prediction API. Use POST method to get predictions."
    data=request.json['features']
    prediction=clf.predict([data])
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    return jsonify({'prediction':int(prediction[0])}
                   ,{'class_name':class_names[int(prediction[0])]})

if __name__ == '__main__':
    import sys
    port = 5000  
    if len(sys.argv) > 1:
        port = int(sys.argv[1])  
    app.run(host='0.0.0.0', port=port, debug=True)
