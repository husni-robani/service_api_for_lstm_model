# this is the API (flask app file)
from flask import Flask, request, jsonify
from utils.predict import Predict
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app=app)

@app.route('/api/predict', methods=['POST'])
def predicts():
    abstracts = request.json.get('abstracts')

    predict = Predict()
    results = []

    for abstract in abstracts:
        if not abstract or abstract == '':
            return jsonify({
                "message": "You must include abstract correctly"
            }), 400
        
        result = predict.make_predict(abstract)
        study_program, index, tensor = result
        results.append(study_program)

    return jsonify({
        "message": "Success",
        "results": results
    }), 200

if __name__ == '__main__':
    app.run(host="127.0.0.1", port="8080")