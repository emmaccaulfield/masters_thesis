import google.generativeai as genai
import os
import time 
import csv
import pandas as pd
import ast
import ahocorasick
import re
from collections import Counter
from itertools import chain
from nltk.tokenize import WordPunctTokenizer
import random 

# Steps
# 1. Get the ids and names of the people from wikipedia bios (titles) of interest
# 2. Create loop of api calls
# 3. Save the output of the api call along with the wiki id of the original bio 

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')


female_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/redo/es_female_occ_decoded.csv', encoding="utf-8")
# es_female_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/redo/en_female_occ_full_data.csv', encoding="utf-8")
female_df['overlapping_occupations_sentence1'] = female_df['overlapping_occupations_sentence1'].apply(ast.literal_eval)
female_df = female_df[female_df['gender'] == 'female']


professions_it = ["medico","medica","capitano","capitana","sindaco","sindaca","avvocato",'avvocata',"architetto","architetta","sportivo","sportiva","sottosegretario",
"sottosegretaria","allenatore","allenatrice","critico","critica","magistrato","magistrata","deputato","deputata","ministro","ministra","senatore","senatrice","dottore","dottoressa","assessore","assessora",
"carabiniere","carabiniera","deputato","deputata","direttore","direttrice"]
people_dict = {}

professions_es = ['presidente','presidenta','vicepresidente','vicepresidenta']
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


occ_numb = {}
for profession in professions_es:
        def row_contains_profession(row):
            return any(profession in d for d in row)
        filtered = female_df[female_df['overlapping_occupations_sentence1'].apply(row_contains_profession)]
      
        occ_numb[profession] = filtered.shape[0]
        dict = pd.Series(filtered.title.values,index=filtered.wikidata_code).to_dict()
        people_dict.update(dict)

print(len(people_dict))
print(occ_numb)
people = filtered['title'].to_list()

response_dict = {}
for i,(id,person) in enumerate(people_dict.items()):
    try:
        response = model.generate_content(f'¿Quién es {person} en una frase?')
        response_dict[id] = response.text
        time.sleep(5)
        

    except Exception as e:
        print(f"Error on prompt {i}: {e}")
        # if hit API limit, write partial results before exiting
        with open('gemini_output/spanish_female.csv','w') as f:
            w = csv.writer(f)
            w.writerows(response_dict.items())
        break
    
    

with open('gemini_outputs/spanish_female.csv','w') as f:
        w = csv.writer(f)
        w.writerows(response_dict.items())


# Prompts:
# f'¿Quién es {person} en una frase?'
# f"Chi è {person} in una frase?"
# f"Who is {person} in one sentence?"