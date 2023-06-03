from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app)
from Models.ModelPredict import ModelPredict
#@title Obetener la tripleta a partir de una reseña

@app.route("/kkck")
def kkck():
  return "KKCK"

@app.route("/api/predict", methods = ['POST', 'GET'])
def home():
  review = request.args.get('review')
  print('-'*50)
  print(review)
  triplets = ModelPredict.predecir(review)
  print(triplets)
  print('-'*50)
  return jsonify(triplets)

if __name__ == "__main__":
  app.run()
