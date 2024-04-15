import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv('job_train.csv')

data = data.drop('location', axis=1)

features = ['title', 'description', 'requirements']
target = 'fraudulent'

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()  
    text = re.sub(r'[^\w\s]', '', text)  
    words = nltk.word_tokenize(text)  
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  
    return ' '.join(words)


for col in features:
    data[col] = data[col].apply(clean_text)

X = data[features]
y = data[target]

vectorizer = CountVectorizer(max_features=5000)

text_features = vectorizer.fit_transform(X.apply(lambda x: ' '.join(x), axis=1))


X_train, X_test, y_train, y_test = train_test_split(text_features, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = LogisticRegression(solver='lbfgs')
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)*100
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)



