import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

review = input("Enter a movie review: ")

review_vector = vectorizer.transform([review])

prediction = model.predict(review_vector)

print("Prediction:", prediction[0])