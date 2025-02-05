import requests
import gzip 
import pandas as pd
import numpy as np

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

def fetch_titles_from_wikidata(wikidata_ids):
    """Fetch Wikipedia titles for multiple Wikidata IDs."""
    wikidata_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(wikidata_ids),  # Batch Wikidata IDs with a pipe
        "props": "sitelinks",
        "sitefilter": "enwiki",
        "format": "json"
    }
    response = requests.get(wikidata_url, params=params)
    data = response.json()
    
    titles = []
    wiki_data_codes = {}
    for wikidata_id in wikidata_ids:
        try:
            title = data['entities'][wikidata_id]['sitelinks']['enwiki']['title']
            titles.append(title)
            wiki_data_codes[wikidata_id] = title
        except KeyError:
            titles.append(None)  # Handle missing titles gracefully
            wiki_data_codes[wikidata_id] = None
    return titles, wiki_data_codes

def fetch_introductions_from_titles(titles):
    """Fetch introductions for multiple Wikipedia titles."""
    wikipedia_url = "https://en.wikipedia.org/w/api.php"
    # Remove None values and join titles with a pipe
    valid_titles = [title for title in titles if title]
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": "true",
        # "exsentences": "1",
        # "explaintext": "true",
        "titles": "|".join(valid_titles),  # Batch titles with a pipe
        "format": "json"
    }
    response = requests.get(wikipedia_url, params=params)
    data = response.json()
    
    introductions = {}
    for page_id, page_data in data['query']['pages'].items():
        title = page_data.get('title')
        extract = page_data.get('extract')
        if title and extract:
            introductions[title] = extract
    return introductions

def batches(lst, n):
    """Yield successive n-sized batches from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Example usage
female_codes, male_codes , other_codes = get_wikidata_codes(filename)
rng = np.random.default_rng()
# randomly sample from the larger gender subsets (female, male)
female_codes = rng.choice(female_codes, 50000)
male_codes = rng.choice(male_codes, 50000)
# get batches of the wikidata codes for queries
other_codes_batches = list(batches(other_codes, 15))
female_codes_batches = list(batches(female_codes, 25))
male_codes_batches = list(batches(male_codes, 25))

intros_dict = {"titles": [], "introductions": []}
wiki_data_codes_all_dict = {"wikidata_id": [], "title": []}

for batch in male_codes_batches: 
    titles, wiki_data_codes_all = fetch_titles_from_wikidata(batch)
    introductions = fetch_introductions_from_titles(titles)
    titles_key = list(introductions.keys())
    introductions_values = list(introductions.values())
    intros_dict["titles"].extend(titles_key)
    intros_dict["introductions"].extend(introductions_values)
    wiki_data_codes_all_keys = list(wiki_data_codes_all.keys())
    wiki_data_codes_all_values = list(wiki_data_codes_all.values())
    wiki_data_codes_all_dict["wikidata_id"].extend(wiki_data_codes_all_keys)
    wiki_data_codes_all_dict["title"].extend(wiki_data_codes_all_values)

# titles_key = introductions.keys()
# introductions_values = introductions.values()
# intros_dict = {"titles": titles_key, "introductions": introductions_values}
intros_df = pd.DataFrame(intros_dict)
# wiki_data_codes_all_keys = wiki_data_codes_all.keys()
# wiki_data_codes_all_values = wiki_data_codes_all.values()
# wiki_data_codes_all_dict = {"wikidata_id": wiki_data_codes_all_keys, "title": wiki_data_codes_all_values}
wiki_data_codes_all_df = pd.DataFrame(wiki_data_codes_all_dict)
intros_df.to_csv("/mnt/c/Users/emmaclai/Documents/Master/thesis/datasets/male_intros.csv", encoding='utf-8')
wiki_data_codes_all_df.to_csv("/mnt/c/Users/emmaclai/Documents/Master/thesis/datasets/male_wikidata_codes.csv", encoding='utf-8')


