import spacy 
import re 
import pandas as pd
from spacy.lang.en import English
from collections import Counter
from itertools import chain
import ahocorasick
import ast
from analyze_m_data import get_occupations_list_en, get_occupations_list
from spacy.lang.de import German
from spacy.lang.it import Italian
from spacy.lang.es import Spanish 
import nltk
from bs4 import BeautifulSoup

# Steps 
# 1. Stem Italian/Spanish occupations list 
# 2. Do dictionary lookup of occupations 
# 3. Do POS tagging on the matched occupations to get the gender 
it_occ_titles = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/it_occ_list_from_en.csv', encoding='utf-8')
it_occ_list = it_occ_titles['occupations'].to_list()

es_occ_titles = pd.read_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/es_occ_list_from_en.csv', names=['occupations'],encoding='utf-8')
es_occ_list = es_occ_titles['occupations'].to_list()
es_occ_list = [x for x in es_occ_list if pd.notnull(x)]
print(es_occ_list[:10])

def stem_titles(titles):
  from nltk.stem import SnowballStemmer 
  stemmer = SnowballStemmer("spanish") # Choose a language
  stems = []
  for title in titles:
     stem = stemmer.stem(title)
     stems.append(stem)
  return stems


def lemmatize_titles(titles): 
  titles = " ".join(titles)
  lemmatizer = spacy.load('it_core_news_sm')
  doc = lemmatizer(titles)
  words_lemmas_list = [token.lemma_ for token in doc]
  return words_lemmas_list

# lemmas = lemmatize_titles(it_occ_list)
# print(lemmas[:10])

def remove_html_en(filename):
    # remove html, clean and lowercase intro texts
    new_df = pd.read_csv(filename, names= ['wikidata_code', 'title', 'intro', 'gender', 'occupations'],header=0, encoding='utf-8')
    # new_df = new_df.head(100)
    print(new_df.shape[0])
    print((new_df['intro'] == 'No intro available').sum())
    # remove biographies with no intro 
    new_df = new_df[new_df.intro != 'No intro available']
    small_df = new_df.copy()
    # lowercase intro string
    small_df['intro'] = small_df['intro'].astype(str)
    # small_df['intro'] = small_df['intro'].str.encode(encoding='utf-8')
    # small_df['intro'] = small_df['intro'].str.casefold()
    
    # remove html tags from intros 
    small_df['intro'] = small_df['intro'].str.replace('\n',"")
    small_df['intro'] = small_df['intro'].str.replace('""',"")
    def remove_hmtl(text): 
        new_text = re.sub(r"\[\s*(\d+|[a-zA-Z]|nota\s*\d+)\s*\]", "", text)
        # new_text = re.sub(r"<sup\b.*", "", text, flags=re.DOTALL)
        new_text = new_text.replace("\u200b", "").replace("\xa0", " ")
        # new_text = re.sub(r'\[\d+\](?:\s|\u200b)*', '', text)
        # new_text = re.sub(r'\[\d+\]\u200b', '', text)
        # new_text = re.sub(r'\[\d+\]', '', text)
        # text = re.sub(r'\[\d+\]\s*[\u200b\u200c\u200d\ufeff]*', '', text)
        # text = re.sub(r'\[\d+\][^\w\s]?', '', text)
        # text = re.sub(r'\[\d+\]\u200b?', '', text)
        # new_text = re.sub(r'\[\d+\]\s*[\u200b\u200c\u200d\ufeff\u202a-\u202e\u2060-\u206f]*', '', text)
        # text = text.replace('\u200b', '').strip()
        return re.sub('<[^<]+?>', '', new_text.rstrip())
    small_df['intro'] = small_df['intro'].apply(remove_hmtl)
    # print(small_df['intro'].head(10))
    return small_df

def remove_html_es(filename):
    new_df = pd.read_csv(filename, names= ['wikidata_code', 'title', 'intro', 'gender', 'occupations'],header=0, encoding='utf-8')
    # new_df = new_df.head(100)
    print(new_df.shape[0])
    print((new_df['intro'] == 'No intro available').sum())
    # remove biographies with no intro 
    new_df = new_df[new_df.intro != 'No intro available']
    small_df = new_df.copy()
    small_df['intro'] = small_df['intro'].astype(str)
    # remove html tags from intros 
    small_df['intro'] = small_df['intro'].str.replace('\n',"")
    small_df['intro'] = small_df['intro'].str.replace('""',"")
    def beautiful_soup(text):
      soup = BeautifulSoup(text, "html.parser")
      # Remove all <sup> tags and their content
      for sup in soup.find_all("sup"):
          sup.decompose()
      cleaned_text = soup.get_text()
      cleaned_text = cleaned_text.replace('\u200b', '').strip()
      return cleaned_text
    small_df['intro'] = small_df['intro'].apply(beautiful_soup)
    return small_df
    


def get_first_sentence(text):
  # split intros into sentences
  nlp = Spanish()
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
  
def get_named_entities(text):
  # ner = spacy.load('en_core_web_sm') 
  # ner = spacy.load('de_core_news_sm')
  ner = spacy.load('it_core_news_sm') 
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

def get_occupations(df):
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
    # small_df['first_sentence'] = small_df['first_sentence'].str.casefold()
    # the lists in the occupations column were not actual lists but rather string literals
    # this code converts them back into lists  
    # small_df['occupations'] = small_df['occupations'].apply(ast.literal_eval)

    # get a set of occupation words 
    # full_unique_occupations = set(stem_titles(it_occ_list))
    full_unique_occupations = set(stem_titles(es_occ_list))
    # print(full_unique_occupations)
    # full_unique_occupations = get_occupations_list('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/de_occ_titles.csv', gender)
    # full_unique_occupations = get_occupations_list_en('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/it_occ_list_full.csv')
    # full_unique_occupations = get_occupations_list_en('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/es_occ_list_unique.csv')
    small_df['intro'] = small_df['intro'].astype(str)
    small_df['title'] = small_df['title'].astype(str)
    
    print(small_df.dtypes)
    # want to find overlapping terms in the intros column and the wikidata occupation words set 
    automaton = ahocorasick.Automaton()
    for id, term in enumerate(full_unique_occupations):
        automaton.add_word(term, (term))
    automaton.remove_word('-')
    automaton.make_automaton()
    def find_terms_en(text):
        return list(term for _, term in automaton.iter(text))
    def find_terms_de(text):
        words = list(re.findall(r'\b\w+\b', text.lower()))  # Tokenize text into words
        matches = list(term for _, term in automaton.iter(text) if term.lower() in words)
        return matches
    def find_terms_stems1(text):
      matching_tokens = []
      substrings = list(full_unique_occupations)
      tokens = text.split()
      for token in tokens:
        if any(sub in token for sub in substrings):
          matching_tokens.append(token)
      return matching_tokens
    
    def find_terms_stems(text):
      # ner = spacy.load('en_core_web_sm') 
      # ner = spacy.load('de_core_news_sm')
      # ner = spacy.load('it_core_news_sm') 
      ner = spacy.load('es_core_news_sm')
      doc = ner(text) 
      occupational_titles = list(full_unique_occupations)
      # occupational_titles.append("tore")
      # occupational_titles.append("trice")
      matched_occupations = []
      for ent in doc: 
        if any(sub in ent.text.casefold() for sub in occupational_titles):
          if not ent.ent_type_:
              # matched_occupations.append(ent.text)
              matched_occupations.append({ent.text: ent.morph.get("Gender")})
        # print(ent.text, ent.pos_, ent.dep_)
          # print(ent,ent.morph.get("Gender"))
      return (matched_occupations)
    
    small_df['overlapping_occupations'] = small_df['intro'].apply(find_terms_stems)
    small_df['overlapping_occupations_sentence1'] = small_df['first_sentence'].apply(find_terms_stems)
    # want to see if male occupation names show up in title field as last names 
    small_df['occupation_in_last_name'] = small_df['title'].apply(find_terms_stems)
    # small_df['overlapping_occupations'] = small_df['intro'].apply(find_terms_en)
    # small_df['overlapping_occupations'] = small_df['intro'].apply(lambda x: list(set(x) & unique_occupations))
    # overlapping_occ_counts = small_df['overlapping_occupations'].value_counts()
    occ_counts_counter = Counter(
    key
    for row in small_df['overlapping_occupations_sentence1']
    for d in row
    for key in d.keys()
    )
    # overlapping_occ_counts = Counter(chain.from_iterable([small_df['overlapping_occupations_sentence1'].keys()]))
    # print(small_df[['wikidata_code','intro', 'overlapping_occupations']].head(20))
    print(occ_counts_counter)
    smaller_df = small_df[small_df['overlapping_occupations_sentence1'].map(len)>0]
    # smaller_df = smaller_df[smaller_df['overlapping_occupations'].map(len)>0]

    print("length of dataset:")
    print(small_df.shape[0])
    print("first sentences with occ title found:", smaller_df.shape[0])
    
    small_df.to_csv('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/sets_for_analysis/es/es_male_occ_2.csv')

it_female_df = remove_html_es('/mount/arbeitsdaten/studenten2/caulfiea/masters_thesis/datasets/final_sets/es_male_dataset.csv')
it_female_df["first_sentence"] = it_female_df["intro"].apply(get_first_sentence)
get_occupations(it_female_df)