from flask import Flask, render_template, request
import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from Fake_job_postings_detection import model, vectorizer
import pickle

app = Flask(__name__)

# model = model
# vectorizer = vectorizer
model = pickle.load(open('model.pkl','rb'))
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

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