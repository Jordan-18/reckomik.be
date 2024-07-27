# Import Library
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from call_function import all_similarity

# Get data
df_komikcast=pd.read_csv("./data/komikcast.csv")
df_westmanga=pd.read_csv("./data/westmanga.csv")

# Load the JSON file
with open("./data/consine_decription_similarity.json", "r") as infile:
    consine_decription_similarity = json.load(infile)
    
# Inverse: Convert lists back to NumPy arrays
consine_decription_similarity = {
    key: np.array(value) for key, value in consine_decription_similarity.items()
}

consine_sim_indo_komikcast = consine_decription_similarity['komikcast']
consine_sim_indo_westmanga = consine_decription_similarity['westmanga']

app = Flask(__name__)
CORS(app)  # This enables CORS for your Flask app.

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "Hello World"}), 404

@app.route('/recommendation', methods=['GET'])
def recommendation():
    title = request.args.get('title')
    if title is None:
        return jsonify({"error": "Title parameter is required"}), 400
    
    try:
        similarity = all_similarity(title, df_komikcast, consine_sim_indo_komikcast)
        similarity = similarity.to_json(orient='records', lines=True)
        return similarity
        return jsonify(similarity)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5234)