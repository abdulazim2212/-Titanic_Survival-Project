from flask import Flask, render_template, request,url_for
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['pclass']
    data2 = request.form['sex']
    data3 = request.form['age']
    data4 = request.form['sibsp']
    data5 = request.form['parch']
    data6 = request.form['fare']
    data7 = request.form['embarked']
    data8 = request.form['tot_family_members']
    arr = np.array([data1, data2, data3, data4, data5, data6, data7, data8], dtype=float)
    arr = arr.reshape(1, -1)
    pred = model.predict(arr)
    return render_template('after.html', data=pred)
if __name__ == "__main__":
    app.run(debug=True)
