from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app)
from Models.ModelPredict import ModelPredict
#@title Obetener la tripleta a partir de una reseña

@app.route("/prueba")
def prueba():
  return "OK"

@app.route("/api/predict", methods = ['POST', 'GET'])
def predict():
  review = request.args.get('review')
  triplets = ModelPredict.predecir(review)
  return jsonify(triplets)

if __name__ == "__main__":
  app.run()
