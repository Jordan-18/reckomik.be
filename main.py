# Import Library
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import asyncio
import json
from call_function import all_similarity, search_title, get_comics_komikcast, get_comics_westmanga, get_comics_mangadex, recommendation

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
# consine_sim_indo_westmanga = consine_decription_similarity['westmanga']

app = Flask(__name__)
CORS(app)  # This enables CORS for your Flask app.

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "Welcome to Reckomik"}), 404

@app.route('/search', methods=['GET'])
def search():
    try:        
        title = request.args.get('title')
        title = search_title(title, df_komikcast)
        response_df = df_komikcast[df_komikcast['title'] == title]
        response = response_df.to_json(orient='records', lines=False)
        return response, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

@app.route('/komikcast-update', methods=['GET'])
def komikcast_update():
    page = request.args.get('page')
    try:
        url = 'https://komikcast.cz/daftar-komik/page' + page
        data, pagination_count  = asyncio.run(get_comics_komikcast(url))
        return jsonify({'pagination':pagination_count,'data':data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/westmanga-update', methods=['GET'])
def westmanga_update():
    page = request.args.get('page')
    try:
        url = 'https://westmanga.fun/manga?page=' + page
        data, pagination_count = asyncio.run(get_comics_westmanga(url))
        return jsonify({'pagination':pagination_count,'data':data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mangadex-update', methods=['GET'])
def mangadex_update():
    page = request.args.get('page')
    try:
        offset = (int(page) - 1) * 32
        url = f'https://api.mangadex.org/manga?limit=32&offset={offset}'
        data, pagination_count = asyncio.run(get_comics_mangadex(url))
        return jsonify({'pagination':pagination_count,'data':data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reckomik', methods=['GET'])
def reckomik():
    title = request.args.get('title')
    criteria = request.args.get('criteria')
    
    if title is None:
        return jsonify({"error": "Title parameter is required"}), 400
    try:
        criteria = eval(criteria)
        sort =  []
        sort_criteria = sorted(criteria.items(), key=lambda x: x[1], reverse=True)
        for c, i in sort_criteria:
            sort.append(c)
        title = search_title(title, df_komikcast)
        
        # CONTERT-BASED-FILTERING
        similarity = all_similarity(title, df_komikcast, consine_sim_indo_komikcast)
        similarity_CBF = similarity[0:21].sort_values(by=sort, ascending=False)
        similarity_CBF = similarity_CBF.to_json(orient='records', lines=False)
        
        # MCDM -MOORA
        reckomik = recommendation(similarity, criteria)
        reckomik = reckomik[:20]
        # merge_df_moora = pd.merge(reckomik, similarity, on='id')
        matching_ids = reckomik['id']
        moora_filtered = df_komikcast[df_komikcast['id'].isin(matching_ids)]
        moora_result = moora_filtered.to_json(orient='records', lines=False)
        
        return {"CONTENT-BASED-FILTERING":similarity_CBF, "MOORA":moora_result}, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5234)