import gzip 
import pandas as pd

# need to take information from the nature dataset (gender, wikidata code, other things?) and combine this with the queried information 
# this will combine the three different files of queries into one 

filename = "/mnt/c/Users/emmaclai/Documents/Master/thesis/cross-verified-database.csv.gz"
csvFilename = gzip.open(filename, 'rb')
df = pd.read_csv(csvFilename, encoding='latin-1')

filtered_df = df[["wikidata_code", "gender", "name", "level3_all_occ"]]
filtered_df = filtered_df.rename(columns={'wikidata_code': 'wikidata_id'})

# print(filtered_df.head(10))

# my datasets have the wikidata_code, title, and introduction of the wikipedia page 
# need to combine the datasets based on wikidata_code, then remove any that dont have introductions/titles

other_filename1 = "/mnt/c/Users/emmaclai/Documents/Master/thesis/datasets/other_intros_1.csv"
other_filename2 = "/mnt/c/Users/emmaclai/Documents/Master/thesis/datasets/other_wikidata_codes_1.csv"

other_intros_df = pd.read_csv(other_filename1, encoding='utf-8')
other_wikidata_codes_df = pd.read_csv(other_filename2, encoding='utf-8')
# other_wikidata_codes_df = other_wikidata_codes_df.dropna(subset=['title'])

other_df = other_intros_df.merge(other_wikidata_codes_df[['wikidata_id','title']], on=["title"])

all_df = filtered_df.merge(other_df[['wikidata_id', 'title', 'introductions']], on=["wikidata_id"])
all_df = all_df.dropna(subset=["title"])
occupational_titles = all_df[["title", "level3_all_occ"]]
print(all_df.head(20))
