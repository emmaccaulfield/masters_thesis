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
from unidecode import unidecode

# This file finds Spanish or Italian gendered occupational titles in language model outputs 
# and compares them to corresponding Wikipedia biographies

filename = "/mount/studenten/projects/caulfiea/cross-verified-database.csv.gz"
csvFilename = gzip.open(filename, 'rb')
df = pd.read_csv(csvFilename, encoding='latin-1')

professions_it = ["medico","medica","capitano","capitana","sindaco","sindaca","avvocato",'avvocata',"architetto","architetta","sportivo","sportiva","sottosegretario",
"sottosegretaria","magistrato","magistrata","ministro","ministra","dottore","dottoressa","assessore","assessora",
"carabiniere","carabiniera","direttore","direttrice","critico","critica","senatore","senatrice","deputato","deputata","deputato","deputata","allenatore","allenatrice"]
professions_it_groups = [["medico","medica"],["capitano","capitana"],["sindaco","sindaca"],["avvocato",'avvocata'],["architetto","architetta"],["sportivo","sportiva"],["sottosegretario",
"sottosegretaria"],["magistrato","magistrata"],["ministro","ministra"],["dottore","dottoressa"],["assessore","assessora"],
["carabiniere","carabiniera"],["direttore","direttrice"],["critico","critica"],["senatore","senatrice"],["deputato","deputata"],["deputato","deputata"],["allenatore","allenatrice"]]

professions_es = ['presidente', 'presidenta','vicepresidente','vicepresidenta']
professions_es_groups = [['presidente','presidenta'],['vicepresidente','vicepresidenta']]


def prompt_analysis(df, occ_titles):
    df = df
    occ_titles = occ_titles
    df['response'] = df['response'].astype(str)
    df['response'] = df['response'].str.casefold()
   
    # get matching tokens for one word tokens 
    automaton = ahocorasick.Automaton()
    for id, term in enumerate(occ_titles):
        automaton.add_word(term, (term))
    automaton.make_automaton()
    def find_terms(text):
        tokenizer = WordPunctTokenizer()
        words = tokenizer.tokenize(text)
        matches = list(term for _, term in automaton.iter(text) if term.lower() in words)
        return matches
    
    df['overlapping_occupations'] = df['response'].apply(find_terms)
    smaller_df = df[df['overlapping_occupations'].map(len)>0]
    print("length of dataset:",df.shape[0])
    print("first sentences with occ title found:", smaller_df.shape[0])

# prompt outputs to analyze
# it_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/mistral_outputs/italian_female.csv',names = ['wikidata_code','response'],encoding='utf-8')
# it_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/gemini_outputs/italian_female.csv',names = ['wikidata_code','response'],encoding='utf-8')
# it_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/gemini_outputs/spanish_female.csv',names = ['wikidata_code','response'],encoding='utf-8')
it_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/mistral_outputs/spanish_female.csv',names = ['wikidata_code','response'],encoding='utf-8')

prompt_analysis(it_df, professions_es)


# add notability scores 
notability_scores = df[['wikidata_code','number_wiki_editions','total_count_words_b','total_noccur_links_b','sum_visib_ln_5criteria']]
it_df_notable = pd.merge(it_df, notability_scores, on='wikidata_code')
it_df_notable.drop_duplicates(subset=['wikidata_code'])

# Compare occupations found in language model output to wikipedia 
# Get the wiki-ids of the language model output and get the same ones from wikipedia data
# Get number of biographies where there is any overlap in terms 
wikidata_codes = it_df_notable['wikidata_code'].to_list()

# Italian female wikipedia biographies
# it_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/redo/it_female_occ.csv', encoding="utf-8") 
# Spanish female wikipedia biographies
it_df = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/redo/es_female_occ_decoded.csv', encoding="utf-8") 

it_df2 = it_df.loc[it_df['wikidata_code'].isin(wikidata_codes)]
it_df_notable = it_df_notable.loc[it_df_notable['wikidata_code'].isin(wikidata_codes)]
it_df_notable = it_df_notable.drop_duplicates(subset=['wikidata_code'])

compare_df_it = pd.merge(it_df_notable, it_df2[["wikidata_code","overlapping_occupations_sentence1"]], on='wikidata_code')
compare_df_it['overlapping_occupations_sentence1'] = compare_df_it['overlapping_occupations_sentence1'].apply(ast.literal_eval)

# occupation titles are a list of dictionaries with titles as keys and gender as values, this retrieves the titles
def get_keys(list_of_dicts):
        keys = []
        for d in list_of_dicts:
            keys.extend(list(d.keys()))
        return keys
    
compare_df_it['overlapping_occupations_sentence1'] = compare_df_it['overlapping_occupations_sentence1'].apply(get_keys)
compare_df_it.drop_duplicates(subset=['wikidata_code'])
print("overlap shape:",compare_df_it.shape[0])

compare_df_it['overlap'] = [list(set(a)&set(b)) for a,b in zip(compare_df_it['overlapping_occupations'], compare_df_it['overlapping_occupations_sentence1'])]
compare_df_it['difference'] = [list(set(a)-set(b)) for a,b in zip(compare_df_it['overlapping_occupations'], compare_df_it['overlapping_occupations_sentence1'])]

overlap_df_en = compare_df_it[compare_df_it['overlap'].map(len)>0]

overlapping_occ_counts_chairperson = Counter(chain.from_iterable(overlap_df_en['overlap']))
print(overlapping_occ_counts_chairperson)
print(overlap_df_en.shape[0]/it_df_notable.shape[0])

def find_other_titles(df):
    output = defaultdict(lambda: defaultdict(int))
    for group in professions_es_groups:
        for title in group:
            def row_contains_profession(row):
                return any(title.lower() == s.lower() for s in row)
            # Step 1: Filter rows where wiki bio occupations column contains the current title
           
            filtered = df[df['overlapping_occupations_sentence1'].apply(row_contains_profession)]
            
            # Step 2: Count occurrences of all titles in the group in the 'difference' column, 
            # which contain the titles that occur in the language model output but not in the wikipedia bios
            for row in filtered['difference']:
                for other_title in group:
                    output[title][other_title] += row.count(other_title)
    output = {k: dict(v) for k, v in output.items()}
    return output
other_titles = find_other_titles(compare_df_it)

other_titles_df = pd.DataFrame({
    'counts': list(other_titles.values())
}, index=list(other_titles.keys()))

print(other_titles_df.to_string())

def get_avg_notability_scores(result):
    all_professions = set()
    for row in result['overlap']:
        for entry in row:
            all_professions.add(entry)

#   compute average score per profession
    avg_scores_wikipedia = {}
    avg_scores_matched = {}
    avg_scores_unmatched = {}
    percent_matched = {}
    numb_matched = {}
    numb_unmatched = {}
    partial_matches = {}
    all_professions = list(all_professions)
    print(all_professions)
    # make all professions the gendered titles of note 
    all_professions = professions_es
    for profession in all_professions:
        def row_contains_profession(row):
            return any(profession.lower() == s.lower() for s in row)

        filtered = result[result['overlapping_occupations_sentence1'].apply(row_contains_profession)]
        avg_scores_wikipedia[profession] = filtered['number_wiki_editions'].mean()
        partial_matches_numb = filtered['partial_matches'].apply(len).sum()
        partial_matches[profession] = partial_matches_numb
        mask = filtered['overlap'].apply(row_contains_profession)
        # create two dataframes, one with overlap of that title between LM output and wiki bio and one without
        matched = filtered[mask].copy()
        unmatched = filtered[~mask].copy()
        if not matched.empty:
            avg_scores_matched[profession] = matched['number_wiki_editions'].mean()
            numb_matched[profession] = matched.shape[0]
            percent_matched[profession] = matched.shape[0]/filtered.shape[0]
        else:
            avg_scores_matched[profession] = None  # or np.nan if preferred
            numb_matched[profession] = 0
            percent_matched[profession] = 0

        if not unmatched.empty:
            avg_scores_unmatched[profession] = unmatched['number_wiki_editions'].mean()
            numb_unmatched[profession] = unmatched.shape[0]
        else:
            avg_scores_unmatched[profession] = None  # or np.nan if preferred
            numb_unmatched[profession] = 0

    # create dataframes of the results
    avg_scores_wikipedia_df = pd.DataFrame.from_dict(avg_scores_wikipedia, orient='index', columns=['average_score_wikipedia'])
    avg_scores_matched_df = pd.DataFrame.from_dict(avg_scores_matched, orient='index', columns=['average_score_matched'])
    avg_scores_unmatched_df = pd.DataFrame.from_dict(avg_scores_unmatched, orient='index', columns=['average_score_unmatched'])
    numb_matched_df = pd.DataFrame.from_dict(numb_matched, orient='index', columns=['number_matched'])
    numb_unmatched_df = pd.DataFrame.from_dict(numb_unmatched, orient='index', columns=['number_unmatched'])
    percent_matched_df = pd.DataFrame.from_dict(percent_matched, orient='index', columns=['percent_matched'])
    partial_matches_df = pd.DataFrame.from_dict(partial_matches, orient='index', columns=['partial_matches'])

    return avg_scores_wikipedia_df,avg_scores_matched_df, avg_scores_unmatched_df, percent_matched_df, numb_matched_df,numb_unmatched_df, partial_matches_df

avg_scores_wikipedia, avg_scores_matched, percent_matched, unmatched, numb_matched, numb_unmatched, partial_matches = get_avg_notability_scores(compare_df_it)
merged_df = pd.merge(avg_scores_wikipedia, avg_scores_matched, left_index=True, right_index=True)
merged_df1 = pd.merge(merged_df, percent_matched, left_index=True, right_index=True)
merged_df2 = pd.merge(unmatched, numb_matched, left_index=True, right_index=True)
merged_df3 = pd.merge(merged_df1,merged_df2,left_index=True, right_index=True)
merged_df4 = pd.merge(numb_unmatched,other_titles_df,left_index=True, right_index=True)
merged_final = pd.merge(merged_df3, merged_df4,left_index=True, right_index=True)
merged_final_sorted = merged_final.sort_values(by='percent_matched')
print(merged_final.to_string())

merged_final.to_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/prompt_analysis/mistral_spanish.csv')

no_matches = compare_df_it[compare_df_it['overlap'].map(len)==0]
some_matches = compare_df_it[compare_df_it['overlap'].map(len)>0]
avg_notability_no_matches = no_matches['number_wiki_editions'].mean()
print("no matches avg notability:",avg_notability_no_matches)
print("at least 1 match avg notability:",some_matches['number_wiki_editions'].mean())