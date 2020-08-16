

import flask
import pickle 
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = flask.Flask(__name__, template_folder='templates')

with open(r'C:\Users\Syed\Desktop\Studies\Python\Data Mining\model.sav', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods= ['GET', 'POST'])
def main():

    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        age = flask.request.form.get('age', False)
        sex = flask.request.form.get('sex', False)
        cp1 = flask.request.form.get('cp1',False)
        cp2 = flask.request.form.get('cp2',False)
        cp3 = flask.request.form.get('cp3',False)
        trestbps = flask.request.form['trestbps']
        chol = flask.request.form['chol']
        fbs = flask.request.form.get('fbs', False)
        restecg = flask.request.form.get('restecg', False)
        thalach = flask.request.form.get('thalach', False)
        exang = flask.request.form.get('exang', False)
        oldpeak = flask.request.form.get('oldpeak', False)
        slope0 = flask.request.form.get('slope0', False)
        slope1 = flask.request.form.get('slope1', False)
        slope2 = flask.request.form.get('slope2', False)
        ca1 = flask.request.form.get('ca1',False)
        ca2 = flask.request.form.get('ca2',False)
        ca3 = flask.request.form.get('ca3', False)
        ca4 = flask.request.form.get('ca4', False)
        thal0 = flask.request.form.get('thal0', False)
        thal1 = flask.request.form.get('thal1', False)
        thal2 = flask.request.form.get('thal2', False)
        thal3 = flask.request.form.get('thal3', False)

        input_var = pd.DataFrame([[age,sex,cp1,cp2,cp3,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope0,slope1,slope2,ca1,ca2,ca3,ca4,thal0, thal1,thal2,thal3]], columns= ['age', 'sex','cp1','cp2','cp3','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope0','slope1','slope2','ca1','ca2','ca3','ca4','thal0','thal1','thal2','thal3'], dtype = float)
        
        
        x = StandardScaler()
        columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        hdd = input_var
        hdd[columns_to_scale] = x.fit_transform(hdd[columns_to_scale]) 
        
        prediction = int(model.predict(input_var))
       
        return flask.render_template("index.html", result = prediction)
if __name__ == "__main__":
    app.run()