import os
from mistralai import Mistral
import pandas as pd
import ast
import time
import json
import csv

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-2503"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "Who is Q8076615",
        },
    ]
)
print(chat_response.choices[0].message.content)


# Steps
# 1. Get the ids and names of the people from wikipedia bios (titles) of interest
# 2. Create loop of api calls with the format "Who is (title) in one sentence?"
# 3. Save the output of the api call along with the wiki id of the original bio 


def create_batch():
    es_female_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/es/es_female_occ_decoded.csv', encoding="utf-8")
    es_female_df['overlapping_occupations_sentence1'] = es_female_df['overlapping_occupations_sentence1'].apply(ast.literal_eval)

    professions = ['presidenta','presidente']
    for profession in professions:
            def row_contains_profession(row):
                return any(profession in d for d in row)
            filtered = es_female_df[es_female_df['overlapping_occupations_sentence1'].apply(row_contains_profession)]

    print(filtered[['wikidata_code','title']])
    people = filtered['title'].to_list()

    # open the output file
    with open("spanish_names.jsonl", "w") as f:
        for idx, person in enumerate(people, start=1):
            data = {
                "custom_id": str(idx),
                "body": {
                    "max_tokens": 100,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Quién es {person} en una frase?"
                        }
                    ]
                }
            }
            f.write(json.dumps(data) + "\n")

def get_batch():
    start = time.time()
    batch_data = client.files.upload(
        file={
            "file_name": "spanish_names.jsonl",
            "content": open("spanish_names.jsonl", "rb")
        },
        purpose = "batch"
    )

    created_job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model="mistral-small-2503",
        endpoint="/v1/chat/completions",
        metadata={"job_type": "testing"}
    )

    retrieved_job = client.batch.jobs.get(job_id=created_job.id)
    output_file_stream = client.files.download(file_id=retrieved_job.output_file)

    # Write and save the file
    with open('spanish_batch_results.jsonl', 'wb') as f:
        f.write(output_file_stream.read())

    end = time.time()
    print("total elapsed time:",end-start)


# es_female_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/es/es_female_occ_decoded.csv', encoding="utf-8")
# es_female_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/it/it_female_occ_2.csv', encoding="utf-8")
es_female_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/en_female_occ_all_sentences_full_data.csv', encoding="utf-8")
print(es_female_df.head())
es_female_df['overlapping_occupations_sentence1'] = es_female_df['overlapping_occupations_sentence1'].apply(ast.literal_eval)
es_female_df = es_female_df[es_female_df['gender'] == 'female']
print(es_female_df.head())
profession_es = ['presidenta','presidente']
professions_it = ["medico","medica","capitano","capitana","sindaco","sindaca","avvocato",'avvocata',"architetto","architetta","sportivo","sportiva","sottosegretario",
"sottosegretaria","allenatore","allenatrice","critico","critica","magistrato","magistrata","deputato","deputata","ministro","ministra","senatore","senatrice","dottore","dottoressa","assessore","assessora",
"carabiniere","carabiniera","deputato","deputata","direttore","direttrice"]
professions = [
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
professions = ["actress"]
for profession in professions:
        def row_contains_profession(row):
            return any(profession in d for d in row)
        filtered = es_female_df[es_female_df['overlapping_occupations_sentence1'].apply(row_contains_profession)]

print(filtered[['wikidata_code','title']])
people = filtered['title'].to_list()
response_dict = {}

# for person in people[:30]:
#     chat_response = client.chat.complete(
#     model= model,
#     max_tokens = 50,
#     messages = [
#             {
#                 "role": "user",
#                 "content": f"Who is {person}?"
#             },
#         ]
#     )
#     response=chat_response.choices[0].message.content
#     response_dict[person] = response
#     # print(chat_response.choices[0].message.content)
#     time.sleep(2)

# with open('mistral_output/english_female_output.csv','w') as f:
#         w = csv.writer(f)
#         w.writerows(response_dict.items())


# "content": f"¿Quién es {person}?"
# "content": f"Chi è {person}?"
