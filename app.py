# this is the API (flask app file)
from flask import Flask, request, jsonify
from utils.predict import Predict
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app=app)

@app.route('/api/predict', methods=['POST'])
def predicts():
    try:
        abstracts = request.json.get('abstracts')
    except Exception as e:
        app.logger.error("Error when try to get abstract json")
        return jsonify({
            "message": "Abstract not provided"
        }), 400
        
    results = []

    for abstract in abstracts:
        if not abstract or abstract == '':
            return jsonify({
                "message": "You must include abstract correctly"
            }), 400
        
        try:
            predict = Predict()
            result = predict.make_predict(abstract)
        except Exception as e:
            return jsonify({
                "message": "Abstract Value Error",
                "errors": "The abstract cannot be process"
            }), 400
        else:
            study_program, index, tensor = result
            results.append(study_program)
            app.logger.info(f"Predict success - {study_program}")

    return jsonify({
        "message": "Success",
        "results": results
    }), 200

if __name__ == '__main__':
    app.run(host="127.0.0.1", port="8080", debug=True)