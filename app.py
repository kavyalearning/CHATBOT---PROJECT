from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

df = pd.read_csv(r"C:\Users\jaswa\Documents\AI CHATBOT\dialogs.txt", sep="\t", header=None)
df.columns = ["question", "answer"]

X = df['question']
y = df['answer']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec,y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def chatbot():
    user_message = request.json["message"]
    
    user_vec = vectorizer.transform([user_message])
    response = model.predict(user_vec)[0]
    
    return jsonify({"response":response})

if __name__ == "__main__":
    app.run(debug=True)
