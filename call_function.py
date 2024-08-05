# Import Library
import pandas as pd
import numpy as np
import math
import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
from aiohttp import ClientSession, ClientTimeout
from asyncio import Semaphore

# Description Similarity System
from fuzzywuzzy import process
import imagehash

def search_title(title, df):
    if title not in df['title'].values:
        matches = process.extract(title, df['title'], limit=5)
        best_match = matches[0][0]
        title = best_match
    return title

# description similarity
def get_similarity_description(title, df, cosine_sim):
    title = search_title(title, df)
        
    idx = df[df['title'] == title].index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = [(idx, 1.0)] + [score for score in sim_scores if score[1] < 1]

    article_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    valid_indices = [i for i in article_indices if i < len(df)]
    
    # Handle cases where valid_indices and scores lengths do not match
    if len(valid_indices) != len(scores):
        min_length = min(len(valid_indices), len(scores))
        valid_indices = valid_indices[:min_length]
        scores = scores[:min_length]
    
    similarity_df = pd.DataFrame({'id': df.iloc[valid_indices]['id'], 'description_similarity': scores})
    result_df = pd.merge(similarity_df, df, on='id')
    
    return result_df

# genre similarity
def matching_genres(df, title):
    title = search_title(title, df)
        
    data = df[df['title'] == title]
    main_genres = (list(data.head(1)['genre'])[0]).split(', ')
    genre_similarity = df['genre']
    
    for i, rec in enumerate(genre_similarity):
        match_count = 0
        if isinstance(rec, float): 
            rec = str(rec) 
        genres = rec.split(', ')
        for j, genre in enumerate(genres):
            if genre in main_genres:
                match_count += 1
        
        genre_similarity = (match_count / len(main_genres)) * 100
        df.loc[i, 'genre_similarity'] = genre_similarity
        
    filtered_df = df.copy()
        
    return filtered_df

# image similarity
def compute_image_similarity(hash1, hash2):
    return 1 - (hash1 - hash2) / len(hash1.hash) ** 2

def get_similarity_image(title, df):
    title = search_title(title, df)
        
    target_row = df[df['title'] == title].iloc[0]
    target_hash_str = target_row['image_hash']
    
    if not isinstance(target_hash_str, str):
        print("Target image hash is not valid.")
        return df
    
    target_hash = imagehash.hex_to_hash(target_hash_str)
    
    similarities = []
    for index, row in df.iterrows():
        if row['title'] == title:
            similarities.append((row['id'], 1.0))  # Default value for the same title
            continue
        
        other_hash_str = row['image_hash']
        if not isinstance(other_hash_str, str):
            similarities.append((row['id'], 0.0))
            continue
        try:
            other_hash = imagehash.hex_to_hash(other_hash_str)
            similarity = compute_image_similarity(target_hash, other_hash)
            similarities.append((row['id'], similarity))
        except Exception as e:
            similarities.append((row['id'], 0.0))
            continue
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    similarity_df = pd.DataFrame(similarities, columns=['id', 'image_similarity'])
    result_df = pd.merge(similarity_df, df, on='id')
    
    return result_df

def all_similarity(title, df, cosine_sim):
    columns_to_drop = ['image_similarity','genre_similarity', 'description_similarity']
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    if existing_columns:
        df = df.drop(existing_columns, axis=1)

    df = matching_genres(df, title) # Genre Similarity Function
    df = get_similarity_image(title, df) # Image Similarity Function
    df = get_similarity_description(title, df, cosine_sim) # Description Similarity Function
    
    df = df.sort_values(by=existing_columns, ascending=False)

    return df

def rank_avg(value, data, order=0):
  if not isinstance(data, np.ndarray):
    data = np.array(data)  # Convert data to NumPy array for efficient operations
  ranks = np.argsort(np.argsort(-data)) + 1 if order == 0 else np.argsort(data) + 1
  return np.mean(ranks[data == value])

def moora(df, criteria): # fix
    ## Denominator
    denominator = [0] * len(criteria)
    denominator = {c: 0 for c in criteria}
    for c, i in criteria.items():
        for _, row in df.iterrows():
            denominator[c] += row[c]**2
        denominator[c] = math.sqrt(denominator[c])
    denominator_json = {'id' : 0,'title' : 'DENOMINATOR'}
    
    for c, i in criteria.items():
        denominator_json[c] = denominator[c]
    
    ## Normalize
    ri_result = []
    for j, row in df.iterrows():
        row_dict = {'id': row['id']}
        for c, i in criteria.items():
            row_dict[f"ri_{c}"] = row[c] / denominator[c]
            # row_dict[c] = row[c] / denominator[c]
        ri_result.append(row_dict)
    
    ## Weighting
    weight_ri = []
    weight_np = [i[1]+1 for c, i in criteria.items()]
    weight_json = {c: ((i[1]+1)/math.fsum(weight_np)) for c, i in criteria.items()}
    for j, row in df.iterrows():
        row_dict = {'id': row['id']}
        matching_item = next((item for item in ri_result if item['id'] == row['id']), None)
        if matching_item:
            for c, i in criteria.items():
                row_dict[f"ri_weight_{c}"] = matching_item.get(f"ri_{c}", 0) * weight_json[c]
                # row_dict[c] = matching_item.get(c, 0) * weight_json[c]
        weight_ri.append(row_dict)
    
    ## Declaration S+1 & S-i
    si = []
    for row in weight_ri:
        Splus = []
        Smin = []
        for c, i in criteria.items():
            if i[0] == 1:
                Splus.append(row.get(f"ri_weight_{c}", 0))
            elif i[0] == 0:
                Smin.append(row.get(f"ri_weight_{c}", 0))
            else:
                pass
        row_dict = {'id': row['id'], 'S+i': math.fsum(Splus), 'S-i': math.fsum(Smin)}
        si.append(row_dict)
    
    ## Qi
    qi = []
    for row in si:
        qi.append({'id': row['id'],'Qi': row['S+i'] - row['S-i']})
    
    ## Ranking
    rangking = []
    for row in qi:
        qi_values = [item['Qi'] for item in qi]
        rangking.append({'id': row['id'],'rank': rank_avg(row['Qi'], qi_values)})

    df_ri = pd.DataFrame(ri_result)
    df_Weight_ri = pd.DataFrame(weight_ri)
    df_si = pd.DataFrame(si)
    df_qi = pd.DataFrame(qi)
    df_ranking = pd.DataFrame(rangking)
    # df_combine = pd.merge(df, df_ri, df_Weight_ri, df_si, df_qi, df_ranking, on='id')
    df_combine = df.merge(df_ri, on='id').merge(df_Weight_ri, on='id').merge(df_si, on='id').merge(df_qi, on='id').merge(df_ranking, on='id')
    df_result = df_combine.sort_values(by=['rank'], ascending=True)
    
    return df_result

def recommendation(df, criteria):
    criteriaValues = list(criteria.keys())
    df = df.sort_values(by=criteriaValues, ascending=False)
    df = df[1:101][['id', 'title']+criteriaValues]
    df = moora(df, criteria) # 1: S+i & 0: s-i
    return df

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def get_comics_komikcast(url):
    data = []
    semaphore = Semaphore(10)  # Limit concurrent requests

    async with ClientSession(timeout=ClientTimeout(total=60)) as session:
        response = await fetch(session, url)
        soup = BeautifulSoup(response, 'lxml')
        comics = soup.find_all('div', class_='list-update_item')
        # pages = soup.find_all('div', class_='page-numbers')
        
        async def process_comic(comic):
            async with semaphore:
                comic_url = comic.find('a').get('href')
                sub_response = await fetch(session, comic_url)
                sub_soup = BeautifulSoup(sub_response, 'lxml')
                
                title = comic.find('h3', class_='title').text
                img = comic.find('img', class_='ts-post-image').get('src')
                raw_rate = comic.find('div', class_='numscore').text.replace(',', '.').replace('..', '.').strip()
                rate = float(raw_rate) if re.match(r'^\d+(\.\d+)?$', raw_rate) else 0.0
                type = comic.find('span', class_='type').text
                
                raw_description = sub_soup.find('div', class_="komik_info-description-sinopsis").text.strip()
                description = re.sub(r'[^a-zA-Z0-9\s]', '', raw_description)
                alt_title = sub_soup.find('span',class_='komik_info-content-native').text.strip()
                released = (sub_soup.find('span', class_='komik_info-content-info-release').text.strip()).replace('Released:', '').strip() 
                author = (sub_soup.find('span', class_='komik_info-content-info').text.strip()).replace('Author:', '').strip()
                
                raw_genre = sub_soup.find('span', class_='komik_info-content-genre')
                if raw_genre:
                    genres = [a.text.strip() for a in raw_genre.find_all('a')]
                    genre = ', '.join(genres)
                else:
                    genre = ""
                
                data.append({
                    'title': title,
                    'alt_title': alt_title,
                    'type': type,
                    'description': description,
                    'genre': genre,
                    'author': author,
                    'artist': "-",
                    'rate': rate,
                    'image': img,
                    'released': released,
                    'comic_url': comic_url
                })
        
        tasks = [process_comic(comic) for comic in comics]
        await asyncio.gather(*tasks)
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    pagination = soup.find('div', class_='pagination')
    page_numbers = pagination.find_all('a', class_='page-numbers')
    pagination_count = page_numbers[-2].text

    return data, pagination_count

async def get_westmanga_comics_details(session, comic_url):
    content = await fetch(session, comic_url)
    soup = BeautifulSoup(content, 'html.parser')
    raw_description = soup.find('div', class_="entry-content").text.strip()
    description = re.sub(r'[^a-zA-Z0-9\s]', '', raw_description)
    raw_alt_title = soup.find('div', class_='seriestualt')
    alt_title = raw_alt_title.text.strip() if raw_alt_title and raw_alt_title.text.strip() else ""
    
    raw_genre = soup.find('div', class_='seriestugenre')
    if raw_genre:
        genres = [a.text.strip() for a in raw_genre.find_all('a')]
        genre = ', '.join(genres)
    else:
        genre = ""
    
    released = "-"
    author = "-"
    artist = "-"
    table = soup.find('table', class_='infotable')
    
    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) > 1:
            key = cells[0].text.strip()
            value = cells[1].text.strip()
            if key == 'Released':
                released = value
            elif key == 'Author':
                author = value
            elif key == 'Artist':
                artist = value
    
    return {
        'description': description,
        'alt_title': alt_title,
        'genre': genre,
        'released': released,
        'author': author,
        'artist': artist
    }

async def get_comics_westmanga(url):
    data = []
    async with aiohttp.ClientSession() as session:
        content = await fetch(session, url)
        soup = BeautifulSoup(content, 'html.parser')
        comics = soup.find_all('div', class_='bs')
        
        tasks = []
        for comic in comics:
            raw_title = comic.find('div', class_='tt').text
            title = re.sub(r'[^a-zA-Z0-9\s]', '', raw_title.replace('\n', ' ').replace('\t', ' ')).strip()
            title = re.sub(r'\s+', ' ', title)
            
            img = comic.find('img', class_='ts-post-image').get('src')
            
            raw_rate = comic.find('div', class_='numscore').text.replace(',', '.').replace('..', '.').strip()
            rate = float(raw_rate) if re.match(r'^\d+(\.\d+)?$', raw_rate) else 0.0
            
            raw_type = comic.find('span', class_='type')
            type = raw_type['class'][1]
            
            comic_url = comic.find('a').get('href')
            tasks.append(get_westmanga_comics_details(session, comic_url))
            
            data.append({
                'title': title,
                'type': type,
                'rate': rate,
                'image': img,
                'comic_url': comic_url
            })
        
        details = await asyncio.gather(*tasks)
        
        for i, detail in enumerate(details):
            data[i].update(detail)
    
    pagination_count = 314
    
    return data, pagination_count

def get_comics_mangadex(url):
    url_stat = 'https://api.mangadex.org/statistics/manga?manga[]='
    response = requests.get(url)
    if response.status_code == 200:
        json = []
        lan = ['en', 'ja', 'ko', 'ru', 'zh', 'ko-ro', 'zh-hk','es-la']
        data = response.json()

        for comic in data['data']:
            url_manga = 'https://api.mangadex.org/manga'
            url_cover = 'https://mangadex.org/covers'
            id = comic['id']
            
            title = next((v for k, v in comic['attributes']['title'].items() if k in lan), None)
            description = next((v for k, v in comic['attributes']['description'].items() if k in lan), None)
            clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
            formatted_title = clean_title.replace(' ', '-')
            comic_url = f'https://mangadex.org/title/{id}/{formatted_title}'
            
            alt_titles = []
            for alt_title_dict in comic['attributes'].get('altTitles', []):
                alt_title = next((v for k, v in alt_title_dict.items() if k in lan), None)
                if alt_title:
                    alt_titles.append(alt_title)
            alt_titles = ', '.join(alt_titles)
            
            genres = []
            for tag in comic['attributes'].get('tags', []):
                tag_name = next((v for k, v in tag['attributes']['name'].items() if k in ['en', 'ja', 'ko']), None)
                if tag_name:
                    genres.append(tag_name)
            genres = ', '.join(genres)
        
            released = comic['attributes']['year'] if comic['attributes']['year'] and int(comic['attributes']['year']) else '-'
            
            rate = 0
            response_rate = requests.get(url_stat+id)
            if response_rate.status_code == 200:
                data_stat = response_rate.json()
                rate = data_stat['statistics'][id]['rating']['average']

            author = ''
            artist = ''
            img = ''
            url_manga += '/'+id+'?includes[]=artist&includes[]=author&includes[]=cover_art'
            response_manga = requests.get(url_manga)
            if response_manga.status_code == 200:
                data_manga = response_manga.json()
                for relation in data_manga['data']['relationships']:
                    if relation['type'] == 'author' and 'attributes' in relation and 'name' in relation['attributes']:
                        author = relation['attributes']['name']
                    if relation['type'] == 'artist' and 'attributes' in relation and 'name' in relation['attributes']:
                        artist = relation['attributes']['name']
                    if relation['type'] == 'cover_art' and 'attributes' in relation and 'fileName' in relation['attributes']:
                        img = f"{url_cover}/{id}/{relation['attributes']['fileName']}"

            json.append({
                'title': title,
                'alt_title': alt_titles,
                'type': comic['type'],
                'description': description,
                'genre': genres,
                'author':author,
                'artist':artist,
                'rate': rate,
                'image': img,
                'released': released,
                'comic_url': comic_url,
            })
            
        pagination_count = 314
        
        return json, pagination_count