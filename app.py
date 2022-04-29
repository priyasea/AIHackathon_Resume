from flask import Flask ,render_template,request
from  text_preprocess import *
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model_rf= joblib.load('C://Users//lenovo//RESUME_SCREENING_APP//joblib//modelrf.pkl')
model_cbow = joblib.load('C://Users//lenovo//RESUME_SCREENING_APP//joblib//vecmodelcbow.pkl')
print('[INFO] model loaded')


@app.route('/')
def hello_word():
    return render_template('input_test.html')

@app.route('/predict' , methods = ['post'])
def predict():
    filename = 'sample'
    text_resume = request.form.get('resume')
    label = -1
    test_resume_df = pd.DataFrame([[filename, text_resume, label]],
                                     columns=['filename', 'content', 'label'], dtype = object)
    test_resume_content = clean_sentences(test_resume_df)  
    num_features = 400
    test_vect_resume = avgFeatureVectors( test_resume_content, model_cbow, num_features )
    result_test_resume = model_rf.predict( test_vect_resume )
    if result_test_resume[0] == 1:
        ans = 'Selected'
    else:
        ans = 'Rejected'



    return render_template('predict.html' , predict = f'The Resume is {ans}' )

    
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)