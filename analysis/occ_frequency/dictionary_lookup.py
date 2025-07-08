import spacy 
import re 
import pandas as pd
from spacy.lang.en import English
from collections import Counter
from itertools import chain
import ahocorasick
import ast
from analysis.occ_frequency.clean_data import get_occupations_list_en, get_occupations_list
from spacy.lang.de import German
from spacy.lang.it import Italian
from spacy.lang.es import Spanish 

def remove_html_en(filename):
    # remove html, clean and lowercase intro texts
    new_df = pd.read_csv(filename, names= ['wikidata_code', 'title', 'intro', 'gender', 'occupations'],header=0, encoding='utf-8')
    print((new_df['intro'] == 'No intro available').sum())
    # remove biographies with no intro 
    new_df = new_df[new_df.intro != 'No intro available']
    small_df = new_df.copy()
    # lowercase intro string
    small_df['intro'] = small_df['intro'].astype(str)
    # small_df['intro'] = small_df['intro'].str.casefold()
    
    # remove html tags from intros 
    small_df['intro'] = small_df['intro'].str.replace('\n',"")
    small_df['intro'] = small_df['intro'].str.replace('""',"")
    def remove_hmtl(text): 
        return re.sub('<[^<]+?>', '', text.rstrip())
    small_df['intro'] = small_df['intro'].apply(remove_hmtl)
    # print(small_df['intro'].head(10))
    return small_df

df = remove_html_en("/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/redo/en_female_dataset.csv")

def get_first_sentence(text):
  # split intros into sentences
  nlp = German()
  nlp.add_pipe("sentencizer")
  # text = "karl m. baer (20 may 1885 – 26 june 1956) was a german-israeli author, social worker, reformer, suffragist and zionist. born intersex and assigned female at birth, he came out as a trans man in 1904 at the age of 19. in december 1906, he became the first transgender person to undergo sex reassignment surgery, and he became one of the first transgender people to gain full legal recognition of his gender identity by having a male birth certificate issued in january 1907. however, some researchers have disputed his label as a trans man, theorizing that he was intersex, and not transgender.baer wrote notes for sexologist magnus hirschfeld on his experiences growing up female while feeling inside that he was male. together they developed these notes into the semi-fictional, semi-autobiographical aus eines mannes mädchenjahren (memoirs of a man's maiden years) (1907) which was published under the pseudonym n.o. body. the book ""was immensely popular,"" being ""adapted twice to film, in 1912 and 1919."" baer also gained the right to marry and did so in october 1907.despite him having undergone gender reaffirming surgery in 1906, exact records of the medical procedures he went through are unknown, as his medical records were burned in the 1930s nazi book burning, that targeted hirschfield studies specifically."
  doc = nlp(text)
  sentences = []
  for i, sent in enumerate(doc.sents): 
    sentences.append(sent)
  if len(sentences) > 0:
    return str(sentences[0])
  else: 
     return str(sentences)

df["first_sentence"] = df["intro"].apply(get_first_sentence)


def get_named_entities(text):
  ner = spacy.load('en_core_web_sm',disable = ['tagger', 'parser']) 
  # ner = spacy.load('de_core_news_sm',disable = ['tagger', 'parser'])
  # ner = spacy.load('it_core_news_sm') 
  # ner = spacy.load('es_core_news_sm')
  doc = ner(text) 
  # removed named entities from text
#   return " ".join([ent.text for ent in doc if not ent.ent_type_])
# return list of named entities 
  named_entities = []
  for ent in doc: 
     if ent.ent_type_:
        named_entities.append(ent.text)
  return named_entities


df["named_entities"] = df["intro"].apply(get_named_entities)

def get_occupations_en(df): 
    small_df = df
    # lowercase intro strings
    small_df['intro'] = small_df['intro'].str.casefold()
    small_df['title'] = small_df['title'].str.casefold()
    small_df['first_sentence'] = small_df['first_sentence'].str.casefold()
    # the lists in the occupations column were not actual lists but rather string literals
    # this code converts them back into lists  
    small_df['occupations'] = small_df['occupations'].apply(ast.literal_eval)
    # count of instances of unique occupation words from wikidata
    value_counts = Counter(chain.from_iterable(small_df['occupations']))
    # print(value_counts)
    # get a set of these unique occupation words 
    wiki_occupations = set(chain.from_iterable(small_df['occupations']))
    unique_occupations = get_occupations_list_en('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/occ_title-lists/eng_occ_titles.csv')
    full_unique_occupations = unique_occupations.union(wiki_occupations)
    
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
    
    small_df['overlapping_occupations'] = small_df['intro'].apply(find_terms_en)
    small_df['overlapping_occupations_sentence1'] = small_df['first_sentence'].apply(find_terms_en)
    overlapping_occ_counts = Counter(chain.from_iterable(small_df['overlapping_occupations']))
    smaller_df = small_df[small_df['overlapping_occupations'].map(len)>0]
    print(overlapping_occ_counts)
    print("length of dataset:")
    print(small_df.shape[0])
    print("intros with occupation titles found:")
    print(smaller_df.shape[0])
    small_df.to_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/redo/en_female_occ_full_data.csv')

get_occupations_en(df)

def get_occupations_de(df):
    new_df = df
    # print(unique_gender_counts)
    # number of biographies that returned no intro 
    print("num intros unavailable")
    print((new_df['intro'] == 'No intro available').sum())
    # remove biographies with no intro 
    new_df = new_df[new_df.intro != 'No intro available']
    print(new_df.shape[0])
    small_df = new_df.copy()
    # lowercase intro strings
    small_df['intro'] = small_df['intro'].str.casefold()
    small_df['title'] = small_df['title'].str.casefold()
    small_df['first_sentence'] = small_df['first_sentence'].str.casefold()
   

    # get a set of german occupation words 
    de_full_unique_occupations_male = get_occupations_list('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/de_occ_titles.csv', 'male')
    de_full_unique_occupations_female = get_occupations_list('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/de_occ_titles.csv', 'female')

    small_df['intro'] = small_df['intro'].astype(str)
    small_df['title'] = small_df['title'].astype(str)
    
    print(small_df.dtypes)
    # want to find overlapping terms in the intros column and the wikidata occupation words set 
    def make_automaton(occupation_list):
      automaton = ahocorasick.Automaton()
      for id, term in enumerate(occupation_list):
          automaton.add_word(term, (term))
      automaton.remove_word('-')
      automaton.remove_word('erster')
      automaton.remove_word('erste')
      automaton.make_automaton()
      return automaton
    automaton_male = make_automaton(de_full_unique_occupations_male)
    automaton_female = make_automaton(de_full_unique_occupations_female)
    def find_terms_de_male(text):
        words = list(re.findall(r'\b\w+\b', text.lower()))  # Tokenize text into words
        matches = list(term for _, term in automaton_male.iter(text) if term.lower() in words)
        return matches
    def find_terms_de_female(text):
        words = list(re.findall(r'\b\w+\b', text.lower()))  # Tokenize text into words
        matches = list(term for _, term in automaton_female.iter(text) if term.lower() in words)
        return matches
    
    small_df['overlapping_occupations_male'] = small_df['intro'].apply(find_terms_de_male)
    small_df['overlapping_occupations_sentence1_male'] = small_df['first_sentence'].apply(find_terms_de_male)
    small_df['overlapping_occupations_female'] = small_df['intro'].apply(find_terms_de_female)
    small_df['overlapping_occupations_sentence1_female'] = small_df['first_sentence'].apply(find_terms_de_female)

    print("length of dataset:")
    print(small_df.shape[0])
    
    small_df.to_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/redo/de_male_occ_full_data.csv')

# get_occupations_de(df)
