import requests
import gzip 
import pandas as pd
import numpy as np
import time

filename = "/mount/studenten/projects/caulfiea/cross-verified-database.csv.gz"
def get_wikidata_codes(filename, wiki_language):
    csvFilename = gzip.open(filename, 'rb')
    df = pd.read_csv(csvFilename, encoding='latin-1')
    df = df[df['list_wikipedia_editions'].str.contains(wiki_language)]
    print(len(df))
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

def get_wikipedia_titles(wikidata_ids, lang="en"):
    """get Wikipedia titles for multiple Wikidata IDs."""
    wikidata_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(wikidata_ids),
        "props": "sitelinks",
        "sitefilter": f"{lang}wiki",
        "format": "json"
    }
    try:
        response = requests.get(wikidata_url, params=params).json()

        if "entities" not in response:
            print("Error: 'entities' key missing in response", response)
            return {}
    
        titles = {qid: entity.get("sitelinks", {}).get(f"{lang}wiki", {}).get("title") for qid, entity in response["entities"].items()}
        return titles
    except Exception as e:
        print(f"Error fetching Wikidata properties: {e}")
        return {}

def get_introductions(titles, lang="en"):
    """get introductions from Wikipedia for multiple titles."""
    wikipedia_url = f"https://{lang}.wikipedia.org/w/api.php"
    valid_titles = [title for title in titles.values() if title]
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": "true",
        "titles": "|".join(valid_titles),
        "redirects": "true",
        "format": "json"
    }
    response = requests.get(wikipedia_url, params=params).json()
    intros = {
    page.get("title", "Unknown"): page.get("extract", "No intro available") for page in response.get("query", {}).get("pages", {}).values()
    if "title" in page  # make sure the page has a title before accessing it
    }

    return intros

def get_wikidata_properties(wikidata_ids):
    """Fetch gender and occupation from Wikidata for multiple IDs."""
    wikidata_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(wikidata_ids),
        "props": "claims",
        "format": "json"
    }
    response = requests.get(wikidata_url, params=params).json()
    
    person_data = {}
    for qid, entity in response["entities"].items():
        claims = entity.get("claims", {})
        
        # get gender (P21)
        gender_id = claims.get("P21", [{}])[0].get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")

        # get occupations (P106)
        occupation_ids = [oc["mainsnak"]["datavalue"]["value"]["id"] for oc in claims.get("P106", [])]

        person_data[qid] = {"gender_id": gender_id, "occupation_ids": occupation_ids}
    
    return person_data

def get_wikidata_properties_2(wikidata_ids):
    """get gender and occupation from Wikidata for multiple IDs."""
    wikidata_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(wikidata_ids),
        "props": "claims",
        "format": "json"
    }

    try:
        response = requests.get(wikidata_url, params=params).json()

        if "entities" not in response:
            print("Error: 'entities' key missing in response", response)
            return {}

        person_data = {}
        for qid, entity in response["entities"].items():
            claims = entity.get("claims", {})

            # get gender (P21)
            gender_id = None
            if "P21" in claims:
                gender_claim = claims["P21"][0].get("mainsnak", {}).get("datavalue", {})
                if "value" in gender_claim and "id" in gender_claim["value"]:
                    gender_id = gender_claim["value"]["id"]

            # get occupations (P106)
            occupation_ids = []
            if "P106" in claims:
                for oc in claims["P106"]:
                    mainsnak = oc.get("mainsnak", {})
                    datavalue = mainsnak.get("datavalue", {})
                    if "value" in datavalue and "id" in datavalue["value"]:
                        occupation_ids.append(datavalue["value"]["id"])

            person_data[qid] = {"gender_id": gender_id, "occupation_ids": occupation_ids}

        return person_data

    except Exception as e:
        print(f"Error fetching Wikidata properties: {e}")
        return {}


def resolve_wikidata_labels(ids):
    """Convert Wikidata IDs to readable names."""
    wikidata_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "labels",
        "languages": "de",
        "format": "json"
    }
    response = requests.get(wikidata_url, params=params).json()
    
    labels = {qid: entity.get("labels", {}).get("de", {}).get("value") for qid, entity in response["entities"].items()}
    return labels

def resolve_wikidata_labels_2(ids, lang="en"):
    """Convert Wikidata IDs to readable names."""
    if not ids:
        return {}  # return an empty dictionary if no IDs are provided

    wikidata_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "labels",
        "languages": lang,
        "format": "json"
    }

    try:
        response = requests.get(wikidata_url, params=params).json()

        if "entities" not in response:
            print("Error: 'entities' key missing in response", response)
            return {}

        # process and return labels
        return {
            qid: entity.get("labels", {}).get("en", {}).get("value", "Unknown")
            for qid, entity in response["entities"].items()
        }

    except Exception as e:
        print(f"Error fetching Wikidata labels: {e}")
        return {}


def batches(lst, n):
    """Yield successive n-sized batches from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



female_codes, male_codes, other_codes = get_wikidata_codes(filename, 'eswiki')
print(len(female_codes), len(male_codes),len(other_codes))
rng = np.random.default_rng()

wikidata_ids = female_codes
female_numb = len(female_codes) 
print(len(female_codes))
# get random number of male codes to match number of female codes
male_codes = rng.choice(male_codes, female_numb)

# split female qid codes into sections, run queries on one list at a time 
# to avoid overloading the wiki servers
female_codes_3_split = np.array_split(female_codes,3)
female_codes_1 = female_codes_3_split[0]
female_codes_2 = female_codes_3_split[1]
female_codes_3 = female_codes_3_split[2]
# female_codes_4 = female_codes_3_split[3]
# female_codes_5 = female_codes_3_split[4]
# female_codes_6 = female_codes_3_split[5]
# female_codes_batches = list(batches(female_codes, 25))

# also split male codes into sections for querying 
male_codes_3_split = np.array_split(male_codes,3)
male_codes_1 = male_codes_3_split[0]
male_codes_2 = male_codes_3_split[1]
male_codes_3 = male_codes_3_split[2]



def get_wiki_queries_batches(batches, lang):
    person_dict = {}
    start = time.time()
    for batch in batches:
        titles = get_wikipedia_titles(batch, lang)
        introductions = get_introductions(titles, lang)
        person_data = get_wikidata_properties_2(batch)

        # Collect all Wikidata IDs (gender + occupations) to resolve their labels
        all_ids_to_resolve = set()
        for data in person_data.values():
            if data["gender_id"]:
                all_ids_to_resolve.add(data["gender_id"])
            all_ids_to_resolve.update(data["occupation_ids"])

        # it seems that the labels are only coming up in english (even when the article is only in non-english)
        labels = resolve_wikidata_labels_2(all_ids_to_resolve, "en")
        # labels = resolve_wikidata_labels_2(all_ids_to_resolve, lang)

        # Display final results
        for qid in batch:
            title = titles.get(qid, "Unknown")
            intro = introductions.get(title, "No intro available")
            gender = labels.get(person_data[qid]["gender_id"], "Unknown")
            occupations = [labels.get(oid, "Unknown") for oid in person_data[qid]["occupation_ids"]]
            person_dict[qid] = {'title':title, 'intro':intro, 'gender':gender, 'occupations':occupations}
            
        # add a bit of time in between query batches to not overwhelm the server 
        time.sleep(1)    
    print("total time taken for queries: ", time.time() - start)
    return person_dict


# example to run queries 
en_other_codes_batches = list(batches(other_codes, 25))
en_other_dict = get_wiki_queries_batches(en_other_codes_batches, lang="es")
en_other_df = pd.DataFrame.from_dict(en_other_dict, orient='index')
# save output
en_other_df.to_csv("/path", encoding='utf-8')




