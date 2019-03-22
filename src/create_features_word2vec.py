#Run load_and_process_text.py before running this script
import pandas as pd
import numpy as np
import gensim
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import ToktokTokenizer

#load dataframe with preprocessed text
df = pd.read_csv('data\interim\processed_text.csv', index_col=0)

#Convert essay_set to categorical variable.  Then extract dummy features into numpy array
df['essay_set'] = df['essay_set'].astype('category')
df = pd.get_dummies(df, columns = ['essay_set'])
dummy_vecs = df.loc[:, df.columns.str.startswith('essay_set')].values

#Tokenize all essays separately
tokenizer = ToktokTokenizer()
essays_tokenized = []
for doc in df.cleaned_text:
    doc = tokenizer.tokenize(doc)
    essays_tokenized.append(doc)

#Instantiate and train model on the tokenized, cleaned, essays
model = gensim.models.Word2Vec(essays_tokenized, size=100)
model.train(essays_tokenized, total_examples=model.corpus_count, epochs=model.epochs)

#Define functions to create vector from an essay of words by summing all vectors and dividing by number of words
def create_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features), dtype='float32')
    nwords = 0
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords += 1
            feature_vec = np.add(feature_vec, model[word])
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec

#Define function to calculate vector for each document in a corpus
def get_essay_vecs(essays, model, num_features):
    counter = 0
    essay_feature_vecs = np.zeros((len(essays), num_features), dtype='float32')
    for essay in essays:
        essay_feature_vecs[counter] = create_feature_vec(essay, model, num_features)
        counter += 1
    return essay_feature_vecs

#Calculate vector for each tokenized essay
essay_vecs = get_essay_vecs(essays_tokenized, model, 100)

#Concatenate features into single numpy array
feature_vecs = np.hstack((dummy_vecs, essay_vecs))
pd.DataFrame(feature_vecs).to_csv('data\interim\\feature_vecs.csv')
