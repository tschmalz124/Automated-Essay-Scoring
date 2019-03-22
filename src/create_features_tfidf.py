#Run load_and_process_text.py before running this script
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

#load dataframe with preprocessed text
df = pd.read_csv('data\interim\processed_text.csv', index_col=0)

#Store index for later use in dataframe
essay_index = df.index

#Convert essay_set to categorical variable.  Then extract dummy features into numpy array
df['essay_set'] = df['essay_set'].astype('category')
df = pd.get_dummies(df, columns = ['essay_set'])
dummy_vecs = df.loc[:, df.columns.str.startswith('essay_set')].values
#Save dummy col names to include in dataframe later
dummy_col_names = df.columns[df.columns.str.startswith('essay_set')]

#Create tf-idf vectors for each essay_set
vectorizer = TfidfVectorizer(min_df=0.05, max_df=0.95)
essay_vecs = vectorizer.fit_transform(df['cleaned_text']).toarray()
essay_col_names = vectorizer.get_feature_names()

#Concatenate features into single numpy array
feature_vecs = np.hstack((df.rating_discrepancy, dummy_vecs, essay_vecs))
col_names = [*dummy_col_names, *essay_col_names]
pd.DataFrame(feature_vecs, index = essay_index, columns=col_names).to_csv('data\interim\\feature_vecs.csv')
