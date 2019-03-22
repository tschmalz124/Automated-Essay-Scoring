#Packages and specific functions necessary
import pandas as pd
import unicodedata
import re
import spacy
import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords

#Load spacy english model, create tokenizer, and stopword list for later use.
nlp = spacy.load('en')
tokenizer = ToktokTokenizer()
nltk.download('stopwords')
stopword_list = set(stopwords.words('english'))

#Load raw training data
file_path = "data/raw/training_set_rel3.xlsx"
df = pd.read_excel(io=file_path, header=0, index_col=0, usecols=range(0,7))

#Drop essay 10534.  Missing all essay scores.
df = df.drop(10534, axis=0)

#Drop essay #6123, 6239, 6669, 7177.  For these three essays, the 2nd rater score is 0 compared to a high 1st rater score.
#Assuming that the 2nd rater score is actually missing.
df = df.drop([6123, 6239, 6669, 7177], axis=0)

#Converting essay_set to categorical variable
df.essay_set = df.essay_set.astype('category')

#Calculate metric for variability in rater and final resolved scores.
#Finds the difference between the largest and smallest value in a list
def difference(ls):
    diff = max(ls) - min(ls)
    return diff

#Returns the largest abs difference between ratings if rating3 is included or not.
def max_diff(row):
    rating1 = row['rater1_domain1']
    rating2 = row['rater2_domain1']
    rating3 = row['rater3_domain1']
    domain1 = row['domain1_score']
    max_diff = difference([rating1, rating2, rating3, domain1])
    return max_diff

#Normalize difference for the maximum score of each type of essay
max_essay_score = {1:12, 2:6, 3:3, 4:3, 5:4, 6:4, 7:30, 8:60}
def compare_raters(row):
    essay_set = row['essay_set']
    normalized_diff = max_diff(row)/max_essay_score[essay_set]
    return normalized_diff

#Creating metric for discrepancy in ratings by rater 1, 2, and 3.  It is normalized with respect to the maximum score for that particular set of essays
#Then drops the individual rating scores.
df['rating_discrepancy'] = df.apply(lambda row: compare_raters(row), axis=1)
df = df.drop(['rater1_domain1', 'rater2_domain1', 'rater3_domain1'], axis=1)


#PROCESSING TEXT
#Remove any accented characters in the text (not typically used in English)
def strip_accented_char(text):
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)

#Function to remove all words starting with @ such as "@LOC".  These are leftover from the anonymization
#of the text to remove any identifying locations, people, etc.
def strip_anon(text):
    pattern = r'@\w+\d+'
    return re.sub(pattern, '', text)

#In the future, add in a spell checker here to replace words.

#Expand contractions (i.e. change don't into do not) so that they aren't interpreted as different words
#CONTRACTION_MAP can be found in project directory under contractions.py
from src.contractions import CONTRACTION_MAP
def expand_contract(text, contraction_mapping=CONTRACTION_MAP):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

#Remove other special characters not found by earlier methods but leave periods.
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

#Lemmatize the text to return words back to their root form
def lemm_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

#Removing common stop words from the text
def remove_sw(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

#Function to normalize corpus incorporating previously defined functions.
def normalize_corpus(corpus, strip_accented_characters=True, strip_anonymization=True,
                     expand_contractions=True, text_lower_case=True, remove_special_char=True,
                     lemmatize_text=True, remove_stop_words=True):
    #Instantiating empty list for later appending
    normalized_corpus = []

    #tqdm adds progress bar when calling function
    from tqdm import tqdm
    for doc in tqdm(corpus):
        #Remove accented characters
        if strip_accented_characters:
            doc = strip_accented_char(doc)
        #Remove words anonymized (starts with @)
        if strip_anonymization:
            doc = strip_anon(doc)
        #Expand all contractions
        if expand_contractions:
            doc = expand_contract(doc)
        #Lower case the text
        if text_lower_case:
            doc = doc.lower()
        #Remove special characters and digits
        if remove_special_char:
            doc = remove_special_characters(doc)
        #Lemmatize text
        if lemmatize_text:
            doc = lemm_text(doc)
        #Remove stop words
        if remove_stop_words:
            doc = remove_sw(doc)
        #Remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        normalized_corpus.append(doc)

    return normalized_corpus


#Normalize text and save into dataframe
df['cleaned_text'] = normalize_corpus(df['essay'])

#Save process_text to interim data folder for quicker loading later
df.to_csv('data\interim\processed_text.csv')
