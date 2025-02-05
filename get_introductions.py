import requests
import gzip 
import pandas as pd
 

filename = "/mnt/c/Users/emmaclai/Documents/Master/thesis/cross-verified-database.csv.gz"
def get_wikidata_codes(filename):
    csvFilename = gzip.open(filename, 'rb')
    df = pd.read_csv(csvFilename, encoding='latin-1')
    females = df[df.gender=='Female']
    female_codes = females['wikidata_code']
    female_codes = female_codes.to_list()
    males = df[df.gender=='Male']
    male_codes = males['wikidata_code']
    male_codes = male_codes.to_list()
    other = df[df.gender=='Other']
    other_codes = other['wikidata_code']
    other_codes = other_codes.to_list()
    return female_codes, male_codes, other_codes


def fetch_article_intro(wikidata_id):
    # Step 1: Get the Wikipedia article title from Wikidata ID
    wikidata_url = f"https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": wikidata_id,
        "props": "sitelinks",
        "sitefilter": "enwiki",
        "format": "json"
    }
    response = requests.get(wikidata_url, params=params)
    data = response.json()
    
    try:
        title = data['entities'][wikidata_id]['sitelinks']['enwiki']['title']
    except KeyError:
        return None, None  # No English Wikipedia article found
    
    # Step 2: Get the article introduction from the title
    wikipedia_url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": "true",
        "titles": title,
        "format": "json"
    }
    response = requests.get(wikipedia_url, params=params)
    data = response.json()
    
    page = next(iter(data['query']['pages'].values()))
    intro = page.get('extract', None)
    
    return title, intro

def batches(lst, n):
    """Yield successive n-sized batches from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Example usage
female_codes, male_codes , other_codes = get_wikidata_codes(filename)
wikidata_ids = ["Q1000542", "Q1000596", "Q1000619"]  # Replace with your list of IDs
id_list = []
title_list = []
intro_list = []
for wikidata_code in other_codes:
    title, intro = fetch_article_intro(wikidata_code)
    if title and intro:
        id_list.append(wikidata_code)
        title_list.append(title)
        intro_list.append(intro)
        # print(f"Title: {title}\nIntroduction: {intro}\n")
    # else:
    #     print(f"No data found for ID: {wikidata_id}")

intros_dict = {'wikidata_code': id_list, 'title': title_list, 'introduction': intro_list} 
intros_df = pd.DataFrame(intros_dict)
intros_df.to_csv("/mnt/c/Users/emmaclai/Documents/Master/thesis/other_intros.csv", encoding='utf-8') 