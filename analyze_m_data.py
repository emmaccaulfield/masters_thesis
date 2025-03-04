import pandas as pd 
import numpy as np
import itertools
from itertools import chain
import ast
from collections import Counter
import ahocorasick
import re

def get_occupations_list(filename, gender:str):
    new_df = pd.read_csv(filename, names= ['male','female'],header=0, encoding='utf-8')
    new_df[["male", "female"]] = new_df[["male", "female"]].astype(str)
    new_df[gender] = new_df[gender].astype(str)
    # lowercase everything
    new_df[gender] = new_df[gender].str.casefold()
    # get rid of extra quotation marks
    new_df[gender] = new_df[gender].apply(lambda x: x.replace('"', ''))
    # split on commas or whitespace
    new_df[gender] = new_df[gender].apply(lambda x: x.split(' '))
    # only take first item in the list split on whitespace
    new_df[gender] = new_df[gender].str[0]
    # print(new_df[gender])
    # unique_occupations = set(chain.from_iterable(new_df[gender]))
    unique_occupations = set(new_df[gender].to_list())
    print(len(unique_occupations))
    for i, val in enumerate(itertools.islice(unique_occupations, 10)):
        print(val)
    return unique_occupations

def get_occupations_list_en(filename):
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
    # print(new_df['occupation'].head(20))
    unique_occupations = set(new_df['occupation'].to_list())
    # print(len(unique_occupations))
    # for i, val in enumerate(itertools.islice(unique_occupations, 10)):
    #     print(val)
    return unique_occupations

def get_combined_occ_list_en(df):
    occ_list_1 = get_occupations_list_en('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/eng_occ_titles.csv')
    occ_list_2 = set(df['occupation'].to_list())
    print(occ_list_2)
    final_occ_list = occ_list_1.union(occ_list_2)
    return final_occ_list
    

# get_occupations_list_en('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/eng_occ_titles.csv')


def analyze_dataset_en(filename):
    new_df = pd.read_csv(filename, names= ['wikidata_code', 'title', 'intro', 'gender', 'occupations'],header=0, encoding='utf-8')
    unique_gender_names = new_df['gender'].unique()
    unique_gender_counts = new_df['gender'].value_counts()
    # print(unique_gender_counts)
    # number of biographies that returned no intro 
    print("num intros unavailable")
    print((new_df['intro'] == 'No intro available').sum())
    new_df = new_df[new_df.intro != 'No intro available']
    small_df = new_df.copy()
    # lowercase intro strings
    small_df['intro'] = small_df['intro'].str.casefold()
    small_df['title'] = small_df['title'].str.casefold()
    # the lists in the occupations column were not actual lists but rather string literals
    # this code converts them back into lists  
    small_df['occupations'] = small_df['occupations'].apply(ast.literal_eval)
    # count of instances of unique occupation words from wikidata
    value_counts = Counter(chain.from_iterable(small_df['occupations']))
    # print(value_counts)
    # get a set of these unique occupation words 
    wiki_occupations = set(chain.from_iterable(small_df['occupations']))
    unique_occupations = get_occupations_list_en('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/eng_occ_titles.csv')
    full_unique_occupations = unique_occupations.union(wiki_occupations)
    small_df['intro'] = small_df['intro'].astype(str)
    small_df['title'] = small_df['title'].astype(str)
    # small_df['intro'] = small_df['intro'].apply(lambda x: x.split(' '))
    # small_df['intro'] = small_df['intro'].astype(str)
    # print(small_df['intro'].head(10))
    print(small_df.dtypes)
    # want to find overlapping terms in the intros column and the wikidata occupation words set 
    automaton = ahocorasick.Automaton()
    for id, term in enumerate(full_unique_occupations):
        automaton.add_word(term, (term))
    automaton.remove_word('-')
    automaton.remove_word('former')
    automaton.remove_word('first')
    automaton.remove_word('second')
    automaton.make_automaton()
    def find_terms_en(text):
        return list(term for _, term in automaton.iter(text))
    def find_terms_de(text):
        words = list(re.findall(r'\b\w+\b', text.lower()))  # Tokenize text into words
        matches = list(term for _, term in automaton.iter(text) if term.lower() in words)
        return matches
    
    small_df['overlapping_occupations'] = small_df['intro'].apply(find_terms_en)
    small_df['overlapping_occupations'] = small_df['intro'].apply(lambda x: list(set(x) & unique_occupations))
    overlapping_occ_counts = small_df['overlapping_occupations'].value_counts()
    overlapping_occ_counts = Counter(chain.from_iterable(small_df['overlapping_occupations']))
    last_name_occ_counts = Counter(chain.from_iterable(small_df['occupation_in_last_name']))
    # print(small_df[['wikidata_code','intro', 'overlapping_occupations']].head(20))
    # print(overlapping_occ_counts)
    print(last_name_occ_counts)
    # smaller_df = small_df.head(20)
    smaller_df = small_df[small_df['overlapping_occupations'].map(len)>0]
    print("length of dataset:")
    print(small_df.shape[0])
    print("intros with occupation titles found:")
    print(smaller_df.shape[0])
    # small_df.to_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/en_other_occ_found_1.csv')

def analyze_dataset(filename, gender):
    new_df = pd.read_csv(filename, names= ['wikidata_code', 'title', 'intro', 'gender', 'occupations'],header=0, encoding='utf-8')
    unique_gender_names = new_df['gender'].unique()
    unique_gender_counts = new_df['gender'].value_counts()
    # print(unique_gender_counts)
    # number of biographies that returned no intro 
    print("num intros unavailable")
    print((new_df['intro'] == 'No intro available').sum())
    new_df = new_df[new_df.intro != 'No intro available']
    print(new_df.shape[0])
    small_df = new_df.copy()
    # lowercase intro strings
    small_df['intro'] = small_df['intro'].str.casefold()
    small_df['title'] = small_df['title'].str.casefold()
    # the lists in the occupations column were not actual lists but rather string literals
    # this code converts them back into lists  
    small_df['occupations'] = small_df['occupations'].apply(ast.literal_eval)

    # get a set of german occupation words 
    full_unique_occupations = get_occupations_list('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/de_occ_titles.csv', gender)
    
    small_df['intro'] = small_df['intro'].astype(str)
    small_df['title'] = small_df['title'].astype(str)
    # small_df['intro'] = small_df['intro'].apply(lambda x: x.split(' '))
    # small_df['intro'] = small_df['intro'].astype(str)
    # print(small_df['intro'].head(10))
    print(small_df.dtypes)
    # want to find overlapping terms in the intros column and the wikidata occupation words set 
    automaton = ahocorasick.Automaton()
    for id, term in enumerate(full_unique_occupations):
        automaton.add_word(term, (term))
    automaton.remove_word('-')
    automaton.remove_word('former')
    automaton.remove_word('first')
    automaton.remove_word('second')
    automaton.make_automaton()
    def find_terms_en(text):
        return list(term for _, term in automaton.iter(text))
    def find_terms_de(text):
        words = list(re.findall(r'\b\w+\b', text.lower()))  # Tokenize text into words
        matches = list(term for _, term in automaton.iter(text) if term.lower() in words)
        return matches
    
    small_df['overlapping_occupations'] = small_df['intro'].apply(find_terms_de)
    # want to see if male occupation names show up in title field as last names 
    small_df['occupation_in_last_name'] = small_df['title'].apply(find_terms_de)
    # small_df['overlapping_occupations'] = small_df['intro'].apply(find_terms_en)
    # small_df['overlapping_occupations'] = small_df['intro'].apply(lambda x: list(set(x) & unique_occupations))
    # overlapping_occ_counts = small_df['overlapping_occupations'].value_counts()
    overlapping_occ_counts = Counter(chain.from_iterable(small_df['overlapping_occupations']))
    last_name_occ_counts = Counter(chain.from_iterable(small_df['occupation_in_last_name']))
    print(small_df[['wikidata_code','intro', 'overlapping_occupations']].head(20))
    # print(overlapping_occ_counts)
    # print(last_name_occ_counts)
    # smaller_df = small_df.head(20)
    smaller_df = small_df[small_df['overlapping_occupations'].map(len)>0]
    print("length of dataset:")
    print(small_df.shape[0])
    print("intros with occupation titles found:")
    print(smaller_df.shape[0])
    # smaller_df.to_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/de_female_with_male_names.csv')

# analyze_dataset("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/es_female_dataset.csv")
# analyze_dataset("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/it_female_dataset.csv")
# analyze_dataset("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/de_female_dataset.csv")

analyze_dataset("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/de_male_dataset.csv", 'female')

def analyze_dataset2(filename):
    new_df = pd.read_csv(filename, names= ['wikidata_code', 'title', 'intro', 'gender', 'occupations'],header=0, encoding='utf-8')
    unique_gender_names = new_df['gender'].unique()
    unique_gender_counts = new_df['gender'].value_counts()
    print(unique_gender_counts)
    print("num biographies")
    print(new_df.shape[0])
    # number of biographies that returned no intro 
    print("num intros unavailable")
    print((new_df['intro'] == 'No intro available').sum())
    print("percent intros unavailable")
    print((new_df['intro'] == 'No intro available').sum()/(new_df.shape[0]))

# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/en_female_dataset.csv")
# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/en_male_dataset.csv")
# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/en_other_dataset.csv")
# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/bad_sets/en_other_dataset.csv")

# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/de_female_dataset.csv")
# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/de_male_dataset.csv")
# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/de_other_dataset.csv")

# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/es_female_dataset.csv")
# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/es_male_dataset.csv")
# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/es_other_dataset.csv")

# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/it_female_dataset.csv")
# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/it_male_dataset.csv")
# analyze_dataset2("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/it_other_dataset.csv")