import pickle

import numpy as np
from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)
model = pickle.load(open('models/tree.joblib', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('form.html')
    if request.method == "POST":
        return predict(request.form)


def predict(form):
    clf = load('models/tree.joblib')
    data = np.zeros(24)
    data[0] = int(form['age'])
    data[1] = int(form['fare'])
    data[2] = 1 if form['class'] == '1' else 0
    data[4] = 1 if form['class'] == '3' else 0
    data[6] = 1 if form['sex'] == 'male' else 0
    data[11] = 1 if form['sibsp'] == '4' else 0
    prediction = clf.predict_proba(data.reshape(1, -1), )
    output = prediction[0][0]
    return '<h1>Survival chance: {}</h1>'.format(str(output))


if __name__ == '__main__':
    app.run(port=5000, debug=True)
