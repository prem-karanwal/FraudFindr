from flask import Flask, render_template, request
import pandas as pd
from Fake_job_postings_detection import model, vectorizer

app = Flask(__name__)

model = model
vectorizer = vectorizer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        requirements = request.form['requirements']
        
        input_data = pd.DataFrame({'title': [title], 'description': [description], 'requirements': [requirements]})
        
        input_text_features = vectorizer.transform(input_data.apply(lambda x: ' '.join(x), axis=1))
        
        prediction = model.predict(input_text_features)[0]
        
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
