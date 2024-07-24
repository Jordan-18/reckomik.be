# Import Library
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from similarity_description_english import get_similarity , prepare_similarity_system 


# Get data
df_komikcast=pd.read_csv("./data/komikcast.csv")
df_westmanga=pd.read_csv("./data/westmanga.csv")
df_mangadex=pd.read_csv("./data/mangadex.csv")

consine_sim_indo_komikcast = prepare_similarity_system(df_komikcast)
consine_sim_indo_westmanga = prepare_similarity_system(df_westmanga)
consine_sim_eng_mangadex = prepare_similarity_system(df_mangadex)

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
        # similarity = get_similarity(title, df_mangadex, consine_sim_eng_mangadex)
        similarity = get_similarity(title, df_komikcast, consine_sim_indo_komikcast)
        return jsonify(similarity)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5234)