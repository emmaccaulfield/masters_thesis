import pandas as pd
from collections import Counter
from itertools import chain
import ahocorasick
import ast
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from operator import itemgetter
import gzip
from nltk.tokenize import WordPunctTokenizer

# This file finds English gendered occupational titles in language model outputs 
# and compares them to corresponding Wikipedia biographies

filename = "/mount/studenten/projects/caulfiea/cross-verified-database.csv.gz"
csvFilename = gzip.open(filename, 'rb')
df = pd.read_csv(csvFilename, encoding='latin-1')

professions_en_all = [
    "actress", "actor", "hostess", "host", "comedienne","comedian", "barmaid","barkeeper", "bartender","barman", 
    "landlady",  "landlord", "policewoman", "policeman","police officer",
    "businesswoman", "businessperson", "businessman","saleswoman", "salesperson","salesman",
    "mailwoman", "mail carrier","postal worker","mailman",  
    "crewwoman", "crew member", "crewman", "waitress", "waiter", "stewardess", "steward",   
    "clergywoman","priest", "clergyman", "firewoman", "firefighter","fireman", "server", "seamstress","sewist",
    "flight attendant", "headmistress","headteacher","headmaster", "barkeeper", "bartender",
    "weathergirl","meteorologist", "weather forecaster",  "weatherman", "anchorwoman","news anchor","anchorman", 
    "camerawoman", "camera operator", "cameraman", "chairwoman", "chairperson", "chairman", 
    "stuntwoman","stuntperson","stuntman"
]

professions_en_groups = [
    ["actress", "actor",], ["hostess", "host",], ["comedienne","comedian",], ["barmaid","barkeeper", "bartender","barman"], 
    ["landlady", "landlord",], ["policewoman", "policeman","police officer"],
    ["businesswoman", "businessperson", "businessman"],["saleswoman", "salesperson","salesman"],
    ["mailwoman", "mail carrier","postal worker","mailman"],  
    ["crewwoman", "crew member", "crewman"], ["waitress", "server","waiter"], ["stewardess", "flight attendant","steward"],   
    ["clergywoman","priest", "clergyman"], ["firewoman", "firefighter","fireman"], ["seamstress","sewist"],
    ["headmistress","headteacher","headmaster"],["weathergirl","meteorologist", "weather forecaster",  "weatherman"], 
    ["anchorwoman","news anchor","anchorman"], ["camerawoman", "camera operator", "cameraman"], ["chairwoman", "chairperson", "chairman"], 
    ["stuntwoman","stuntperson","stuntman"]
]

def get_split_occupations_en(filename):
    new_df = pd.read_csv(filename, names= ['occupation'],header=0, encoding='utf-8')
    new_df["occupation"] = new_df['occupation'].astype(str)
    # lowercase everything 
    new_df['occupation'] = new_df['occupation'].str.casefold()
    # get rid of extra quotation marks
    new_df['occupation'] = new_df['occupation'].apply(lambda x: x.replace('"', ''))
    # split on commas 
    new_df['occupation'] = new_df['occupation'].apply(lambda x: x.split(','))
    # only take first item in the list split on commas
    new_df['occupation'] = new_df['occupation'].str[0]
    
    unique_occupations = set(new_df['occupation'].to_list())
    
    # separate strings with and without whitespace
    no_whitespace = [s for s in unique_occupations if " " not in s]
    with_whitespace = [s for s in unique_occupations if " " in s]
    print("All:", len(unique_occupations))
    print("No Whitespace:", len(no_whitespace))
    print("With Whitespace:", len(with_whitespace))
    return no_whitespace,with_whitespace

# reused from Wikipedia data analysis
def new_english_analysis(df):
    df = df
    #Want to: 
    # 1. Split up english occupation titles list into two lists: one token occupations and multi token
    # 2. For one token occupations, want to find matches (in first sentence) with hard word boundaries (like in german code)
    # 3. For multi token occupations, want to just look up matches 
    # 4. Want to remove named entities from matched tokens in that line 
    # 5. Want to get final matched titles list 
    en_1token_occ_titles, en_multitoken_occ_titles = get_split_occupations_en('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/occ_title-lists/eng_occ_titles.csv')
    
    df['response'] = df['response'].astype(str)
    df['response'] = df['response'].str.casefold()
   
    # add wiki_occupations to 1 token list
    wiki_occupations = set(chain.from_iterable(df['occupations']))
    en_1token_occ_titles.extend(wiki_occupations)

    # get matching tokens for one word tokens 
    automaton = ahocorasick.Automaton()
    for id, term in enumerate(en_1token_occ_titles):
        automaton.add_word(term, (term))
    automaton.remove_word('-')
    automaton.remove_word('former')
    automaton.remove_word('first')
    automaton.remove_word('second')

    
    automaton.make_automaton()
    def find_terms_en(text):
        tokenizer = WordPunctTokenizer()
        words = tokenizer.tokenize(text)
        matches = list(term for _, term in automaton.iter(text) if term.lower() in words)
        return matches
    
    # get matches for multi word tokens (simpler lookup)
    automaton_multi = ahocorasick.Automaton()
    for id, term in enumerate(en_multitoken_occ_titles):
        automaton_multi.add_word(term, (term))
    automaton_multi.remove_word('-')
    automaton_multi.make_automaton()
    def find_terms_en_multi(text):
        return list(term for _, term in automaton_multi.iter(text))
    
    def find_all_terms(text): 
        single_terms = find_terms_en(text)
        multi_terms = find_terms_en_multi(text)
        single_terms.extend(multi_terms)
        return single_terms
    
    def filter_NER(row):
        col1_counts = Counter(row['overlapping_occupations_sentence1'])  # Count occurrences in col1
        col2_counts = Counter(row['named_entities'])  # Count occurrences in col2

        # Remove only up to the number of times they appear in col2
        for word, count in col2_counts.items():
            if word in col1_counts:
                col1_counts[word] -= count  # Reduce count but not below zero
                if col1_counts[word] <= 0:
                    del col1_counts[word]  # Remove word if count reaches zero

        # Reconstruct the filtered list based on updated counts
        filtered_list = []
        for word, count in col1_counts.items():
            filtered_list.extend([word] * count)  # Add back only remaining occurrences

        return filtered_list

    df['overlapping_occupations'] = df['response'].apply(find_all_terms)
    smaller_df = df[df['overlapping_occupations'].map(len)>0]
    print("length of dataset:",df.shape[0])
    print("first sentences with occ title found:", smaller_df.shape[0])


# prompt outputs to analyze
english_female= pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/gemini_outputs/english_female.csv',names = ['wikidata_code','response'],encoding='utf-8')
# english_female = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/mistral_outputs/english_female.csv',names = ['wikidata_code','response'],encoding='utf-8')

new_english_analysis(english_female)
overlapping_occ_counts_chairperson = Counter(chain.from_iterable(english_female['overlapping_occupations']))

# add notability scores 
notability_scores = df[['wikidata_code','number_wiki_editions','total_count_words_b','total_noccur_links_b','sum_visib_ln_5criteria']]
english_female = pd.merge(english_female, notability_scores, on='wikidata_code')

# Compare occupations found in model output to wikipedia 
# Get the wiki-ids of the model output and get the same ones from wikipedia data
# Get number of biographies where there is any overlap in terms 
wikidata_codes = english_female['wikidata_code'].to_list()
en_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/redo/en_female_occ_full_data.csv', encoding="utf-8") 
en_df2 = en_df.loc[en_df['wikidata_code'].isin(wikidata_codes)]
print(en_df2.shape[0])
compare_df_en = pd.merge(english_female, en_df2[["wikidata_code","overlapping_occupations_sentence1"]], on='wikidata_code')
compare_df_en['overlapping_occupations_sentence1'] = compare_df_en['overlapping_occupations_sentence1'].apply(ast.literal_eval)
print(compare_df_en.shape[0])

compare_df_en['overlap'] = [list(set(a)&set(b)) for a,b in zip(compare_df_en['overlapping_occupations'], compare_df_en['overlapping_occupations_sentence1'])]
compare_df_en['difference'] = [list(set(a)-set(b)) for a,b in zip(compare_df_en['overlapping_occupations'], compare_df_en['overlapping_occupations_sentence1'])]
overlap_df_en = compare_df_en[compare_df_en['overlap'].map(len)>0]
overlapping_occ_counts_chairperson = Counter(chain.from_iterable(overlap_df_en['overlap']))
print(overlapping_occ_counts_chairperson)
print(overlap_df_en.shape[0]/english_female.shape[0])


def find_other_titles(df):
    output = defaultdict(lambda: defaultdict(int))
    for group in professions_en_groups:
        for title in group:
            def row_contains_profession(row):
                return any(title.lower() == s.lower() for s in row)
            
            # Step 1: Filter rows where col1 contains the current title
            filtered = df[df['overlapping_occupations_sentence1'].apply(row_contains_profession)]
           
            # Step 2: Count occurrences of all titles in the group in 'col2'
            for row in filtered['difference']:
                for other_title in group:
                    output[title][other_title] += row.count(other_title)
    output = {k: dict(v) for k, v in output.items()}
    return output
other_titles = find_other_titles(compare_df_en)

other_titles_df = pd.DataFrame({
    'counts': list(other_titles.values())
}, index=list(other_titles.keys()))

print(other_titles_df.to_string())

def get_avg_notability_scores(result):
    all_professions = set()
    for row in result['overlap']:
        for entry in row:
            all_professions.add(entry)

    # Step 2: Compute average score per profession
    avg_scores_wikipedia = {}
    avg_scores_matched = {}
    avg_scores_unmatched = {}
    numb_matched = {}
    numb_unmatched = {}
    percent_matched = {}
    all_professions = list(all_professions)
    # make all professions the gendered titles of note 
    all_professions = professions_en_all
    for profession in all_professions:
        def row_contains_profession(row):
            return any(profession.lower() == s.lower() for s in row)

        filtered = result[result['overlapping_occupations_sentence1'].apply(row_contains_profession)]

        avg_scores_wikipedia[profession] = filtered['number_wiki_editions'].mean()
        mask = filtered['overlap'].apply(lambda titles: profession in titles)
        # create two dataframes, one with overlap of that title between LM output and wiki bio and one without
        matched = filtered[mask].copy()
        unmatched = filtered[~mask].copy()
        
        if not matched.empty:
            avg_scores_matched[profession] = matched['number_wiki_editions'].mean()
            numb_matched[profession] = matched.shape[0]
            percent_matched[profession] = matched.shape[0]/filtered.shape[0]
        else:
            avg_scores_matched[profession] = None 
            numb_matched[profession] = 0
            percent_matched[profession] = 0


        if not unmatched.empty:
            avg_scores_unmatched[profession] = unmatched['number_wiki_editions'].mean()
            numb_unmatched[profession] = unmatched.shape[0]
        else:
            avg_scores_unmatched[profession] = None  
            numb_unmatched[profession] = 0

    # create dataframes of the results
    avg_scores_wikipedia_df = pd.DataFrame.from_dict(avg_scores_wikipedia, orient='index', columns=['average_score_wikipedia'])
    avg_scores_matched_df = pd.DataFrame.from_dict(avg_scores_matched, orient='index', columns=['average_score_matched'])
    avg_scores_unmatched_df = pd.DataFrame.from_dict(avg_scores_unmatched, orient='index', columns=['average_score_unmatched'])
    numb_matched_df = pd.DataFrame.from_dict(numb_matched, orient='index', columns=['numb_matched'])
    numb_unmatched_df = pd.DataFrame.from_dict(numb_unmatched, orient='index', columns=['numb_unmatched'])
    percent_matched_df = pd.DataFrame.from_dict(percent_matched, orient='index', columns=['percent_matched'])

    return avg_scores_wikipedia_df, avg_scores_matched_df, percent_matched_df, avg_scores_unmatched_df, numb_matched_df, numb_unmatched_df

avg_scores_wikipedia, avg_scores_matched, percent_matched, unmatched, numb_matched, numb_unmatched = get_avg_notability_scores(compare_df_en)
merged_df = pd.merge(avg_scores_wikipedia, avg_scores_matched, left_index=True, right_index=True)
merged_df1 = pd.merge(merged_df, percent_matched, left_index=True, right_index=True)
merged_df2 = pd.merge(unmatched, numb_matched, left_index=True, right_index=True)
merged_df3 = pd.merge(merged_df1,merged_df2,left_index=True, right_index=True)
merged_df4 = pd.merge(merged_df3,numb_unmatched,left_index=True, right_index=True)
merged_df5 = pd.merge(merged_df4,other_titles_df,left_index=True, right_index=True)
merged_final = merged_df5.sort_values(by='percent_matched')
print(merged_final.to_string())

merged_df5.to_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/prompt_analysis/gemini_english.csv')
no_matches = compare_df_en[compare_df_en['overlap'].map(len)==0]
some_matches = compare_df_en[compare_df_en['overlap'].map(len)>0]
avg_notability_no_matches = no_matches['number_wiki_editions'].mean()
print("no matches avg notability:",avg_notability_no_matches)
print("at least 1 match avg notability:",some_matches['number_wiki_editions'].mean())