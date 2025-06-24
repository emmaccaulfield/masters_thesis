import os
from mistralai import Mistral
import pandas as pd
import ast
import time
import json
import csv
import ahocorasick
import re
from collections import Counter
from itertools import chain
import numpy as np
import random
import gzip

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-2503"

client = Mistral(api_key=api_key)

# Steps
# 1. Get the ids and names of the people from wikipedia bios (titles) of interest
# 2. Create loop of api calls
# 3. Save the output of the api call along with the wiki id of the original bio 

es_female_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/redo/es_female_occ_decoded.csv', encoding="utf-8")
# es_female_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/redo/it_female_occ.csv', encoding="utf-8")
# es_female_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/redo/en_female_occ_full_data.csv', encoding="utf-8")

es_female_df['overlapping_occupations_sentence1'] = es_female_df['overlapping_occupations_sentence1'].apply(ast.literal_eval)
es_female_df = es_female_df[es_female_df['gender'] == 'female']


professions_it = ["medico","medica","capitano","capitana","sindaco","sindaca","avvocato",'avvocata',"architetto","architetta","sportivo","sportiva","sottosegretario",
"sottosegretaria","allenatore","allenatrice","critico","critica","magistrato","magistrata","deputato","deputata","ministro","ministra","senatore","senatrice","dottore","dottoressa","assessore","assessora",
"carabiniere","carabiniera","deputato","deputata","direttore","direttrice"]

professions_en = [
    "actress", "waitress", "actress", "seamstress", "hostess", "stewardess",
    "headmistress", "comedienne", "barmaid", "landlady", "policewoman",
    "businesswoman", "saleswoman", "mailwoman", "weathergirl", "anchorwoman",
    "camerawoman", "chairwoman", "clergywoman", "crewwoman", "stuntwoman",
    "waiter", "steward", "headmaster", "barman", "policeman", "businessman",
    "salesman", "mailman", "weatherman", "anchorman", "cameraman", "chairman",
    "clergyman", "crewman", "stuntman", "fireman", "server", "sewist",
    "flight attendant", "headteacher", "barkeeper", "bartender",
    "police officer", "businessperson", "salesperson", "postal worker",
    "mail carrier", "meteorologist", "weather forecaster", "news anchor",
    "camera operator", "chairperson", "minister/pastor", "crew member",
    "stuntperson", "firefighter", "actor", "host", "comedian", "landlord",
    "chairman"
]

professions_es = ['presidenta','presidente','vicepresidenta','vicepresidente']


people_dict = {}

occ_numb = {}
for profession in professions_es:
        def row_contains_profession(row):
            return any(profession in d for d in row)
        filtered = es_female_df[es_female_df['overlapping_occupations_sentence1'].apply(row_contains_profession)]
        occ_numb[profession] = filtered.shape[0]
        dict = pd.Series(filtered.title.values,index=filtered.wikidata_code).to_dict()
        people_dict.update(dict)

response_dict = {}

for i,(id,person) in enumerate(people_dict.items()):
    try:
        chat_response = client.chat.complete(
        model= model,
        max_tokens = 50,
        messages = [
                {
                    "role": "user",
                    "content": f"¿Quién es {person}?"
                },
            ]
        )
        response=chat_response.choices[0].message.content
        response_dict[id] = response
        time.sleep(6)
    except Exception as e:
        print(f"Error on prompt {i}: {e}")
        # if API limits hit, write partial results before exiting
        with open(f'mistral_output/spanish_female_juez{i}.csv','w') as f:
            w = csv.writer(f)
            w.writerows(response_dict.items())
        print("number of people in output:", i)
        break
    

    

with open('mistral_output/spanish_female_juez.csv','w') as f:
        w = csv.writer(f)
        w.writerows(response_dict.items())
        
    

# Prompts: 
# "content": f"¿Quién es {person}?"
# "content": f"Chi è {person}?"
# "content": f"Who is {person}?"
