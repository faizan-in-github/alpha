import search
import pandas as pd
import os

df = pd.read_csv(".\\description.csv")

searcher = search.Search()

model_files= all([
    os.path.exists(searcher.dictionary_path),
    os.path.exists(searcher.corpus_path),
    os.path.exists(searcher.tfidf_model_path),
    os.path.exists(searcher.lsi_model_path),
    os.path.exists(searcher.index_path)
])

if not model_files:
    searcher.run(df['Description'], query=None) 

result = searcher.run(df['Description'], 'I am looking for plumbers who can fix my pipe')
details = []

for idx, row in result.iterrows():
    if row['Relevance'] >= 50.0:
        details.append(df.iloc[int(row['Value'])])

df_details = pd.DataFrame(details)
print(df_details)