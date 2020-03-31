# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:03:05 2019

@author: geetha
"""
from nltk.corpus import stopwords
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance
from collections import Counter
from itertools import chain
from nltk import word_tokenize, pos_tag
from nltk import FreqDist
#%%
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
#%%
def remove_stopwords(tweet_tokens):
    stop_words = stopwords.words('english')
    tweet_tokens = [word for word in tweet_tokens if word not in (stop_words)]
    return tweet_tokens
#%%
def lemmatization(tweet_tokens):
    lem = WordNetLemmatizer()
    tweet_tokens = [lem.lemmatize(token,get_wordnet_pos(token)) for token in tweet_tokens]
    return tweet_tokens
#%%
def remove_count_user_mentions(tweet):
    tweet_mentions_removed = re.subn(r'@[A-Za-z0-9]+','',tweet)
    tweet = tweet_mentions_removed[0]
    no_user_mentions = tweet_mentions_removed[1]
    return tweet,no_user_mentions
#%%
def remove_count_urls(tweet):
    tweet_url_removed = re.subn('https?://[A-Za-z0-9./]+','',tweet)
    tweet = tweet_url_removed[0]
    no_urls = tweet_url_removed[1]
    return tweet,no_urls
#%%
def remove_count_hashtags(tweet):
    no_hashtags = len({tag.strip("#") for tag in tweet.split() if tag.startswith("#")})
    tweet = re.sub("[^a-zA-Z]", " ",tweet)
    return tweet,no_hashtags    
#%%
def preprocessing(tweet):
    preprocessed = []
    tweet = tweet.lower()
    #removing @mentions
    tweet,no_user_mentions = remove_count_user_mentions(tweet)
    preprocessed.append(no_user_mentions)
    
    #removing URLs
    tweet,no_urls = remove_count_urls(tweet)
    preprocessed.append(no_urls)
    
    #Remove unicode kind of characters \x99s
    tweet = tweet.encode('ascii', 'ignore').decode("utf-8")
    
    #Identify hashtags and Remove non letter characters including '#'
    tweet,no_hashtags = remove_count_hashtags(tweet)
    preprocessed.append(no_hashtags)
    
    #Compute number of characters in a tweet
    no_chars = len(tweet) - tweet.count(' ')
    preprocessed.append(no_chars)
    
    #remove short words
    tweet = " ".join(word for word in tweet.split() if len(word)>2)
    
    #Generate tokens for Stop word removal and Stemming
    tok = WordPunctTokenizer()
    tweet_tokens = tok.tokenize(tweet)
    no_words = len(tweet_tokens)
    preprocessed.append(no_words)
    
    #Stop Word Removal
    tweet_tokens = remove_stopwords(tweet_tokens)
    
    #POS Tagging
    #Lemmatization on POS Tag is better than Lemmatization only or Stemming
    tweet_tokens = lemmatization(tweet_tokens)
        
    #Convert tokens to string and Remove unnecessary white spaces 
    tweet = " ".join(tweet_tokens).strip()
    preprocessed.append(tweet)
    #create a list of required attributes obtained after preprocessing
    return (preprocessed)
#%%
def clean_and_get_features(data):
    print('Cleaning the Tweets, Stop Word Removal, Lemmatization, Feature Extraction ')
    tweets = data['Sentence']
    tweets = tweets.astype(str)
    preprocessed = tweets.apply(preprocessing)
    preprocessed_df = pd.DataFrame(columns=['no_user_mentions', 'no_urls', 'no_hashtags',
                                            'no_chars','no_words','tweet'])
    preprocessed_df['no_user_mentions']=preprocessed_df['no_user_mentions'].astype(int)
    preprocessed_df['no_urls']=preprocessed_df['no_urls'].astype(int)
    preprocessed_df['no_hashtags']=preprocessed_df['no_hashtags'].astype(int)
    preprocessed_df['no_chars']=preprocessed_df['no_chars'].astype(int)
    preprocessed_df['no_words']=preprocessed_df['no_words'].astype(int)
    
    for i in range(len(preprocessed)):
        print(i)
        preprocessed_df.loc[i] = preprocessed.iloc[i]

    cleaned_tweet = preprocessed_df['tweet'] 
    return preprocessed_df, cleaned_tweet
#%%
def generate_TFIDF_matrix(text):
    print('Generating TFIDF matrix - Uni-grams and Bi-grams')
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, 
                                           max_features=2000, stop_words='english',
                                           ngram_range=(1,2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(text)
    tfidf_matrix = tfidf_matrix.todense()
    feature_names = tfidf_vectorizer.get_feature_names()
    TFIDF_df = pd.DataFrame(tfidf_matrix,columns=feature_names)
    #Write features to a file to get frequency from Google Corpus
    filepath = '../data/features.txt'
    with open(filepath, 'w') as file_handler:
        for item in feature_names:
            file_handler.write("{}\n".format(item))  
            
    return TFIDF_df
#%%
def get_average_cosine_similarity(TFIDF):
    cosine_similarity_df  = cosine_similarity(TFIDF)
    avg_cosine_sim = cosine_similarity_df.mean(axis=1)
    avg_cosine_sim = pd.DataFrame(avg_cosine_sim,columns=['avg_cosine_sim'])
    return avg_cosine_sim
#%%
def get_average_familiarity_score(TFIDF):
    #Get the frequencies from Google Web Corpus before executing this function
    tweet_tokens = TFIDF[TFIDF!=0].stack()
    avg_familiarity_score_list = []
    feature_frequencies = pd.read_csv('../data/features_freq.csv',index_col=0)
    for i in range(TFIDF.shape[0]):
        avg_familiarity_score = 0
        if(TFIDF.iloc[i].sum()!=0):
            toks = tweet_tokens.loc[i].index
            frequencies = feature_frequencies.loc[toks]
            avg_familiarity_score = frequencies.mean()[0]
        avg_familiarity_score_list.append(avg_familiarity_score)
    avg_familiarity_score_df = pd.DataFrame(avg_familiarity_score_list,
                                            columns=['avg_familiarity_score'])
    return(avg_familiarity_score_df)
    
#%%
def get_average_edit_distance(tweets):
    edit_distance_Matrix = np.zeros((len(tweets),len(tweets)),dtype=np.int)
    for i in range(0,len(tweets)):
        for j in range(0,len(tweets)):
            edit_distance_Matrix[i,j] = distance(tweets[i],tweets[j])
    edit_distance_df = pd.DataFrame(edit_distance_Matrix)
    avg_edit_distance = edit_distance_df.mean(axis=1)
    avg_edit_distance = pd.DataFrame(avg_edit_distance,
                                     columns=['avg_edit_distance'])
    return(avg_edit_distance)
#%%   
def get_pos_tag_count_matrix(data):
    tokens= data.Sentence.apply(nltk.word_tokenize)
    pos_tag  = tokens.apply(nltk.pos_tag)
    noun_count = pos_tag.apply(NounCounter).str.len()
    pronoun_count = pos_tag.apply(PRPCounter).str.len()
    verb_count = pos_tag.apply(VerbCounter).str.len()
    adj_count = pos_tag.apply(AdjCounter).str.len()
    adverb_count = pos_tag.apply(AdVCounter).str.len()
    pattern = pos_tag.apply(Pattern)
    pos_count_df = pd.concat([noun_count,pronoun_count,verb_count,adj_count,adverb_count],axis=1)
    pos_count_df.columns=['noun_count','pronoun_count','verb_count','adj_count','adverb_count']
    return(pos_count_df)

#%%
def NounCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("NN"):
            nouns.append(word)
    return nouns      
#%%
def PRPCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("PRP"):
            nouns.append(word)
    return nouns      
#%%
def VerbCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("VB"):
            nouns.append(word)
    return nouns      
#%%
def AdjCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("JJ"):
            nouns.append(word)
    return nouns      
#%%
def AdVCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("RB"):
            nouns.append(word)
    return nouns      
#%%
def Pattern(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith('NN'):
            nouns.append("sw")
        elif pos in ["VB", "VBD", "VBG", "VBN" , "VBP"]: 
            nouns.append('sw')
        elif pos.startswith("JJ"): 
            nouns.append('sw')
        elif pos.startswith("RB"): 
            nouns.append('sw')
        else:
            nouns.append('cw')   
  
    return nouns      
#%%
#%% 
def generate_data_matrix(data):
    preprocessed_df, cleaned_tweets = clean_and_get_features(data)
    TFIDF_df = generate_TFIDF_matrix(cleaned_tweets)
    print('Dimension of TFIDF matrix')
    print(TFIDF_df.shape)
    print('Computing Average Familiarity Score')
    avg_familiarity_score = get_average_familiarity_score(TFIDF_df)
    print('Computing Average Cosine Similarity')
    avg_cosine_sim = get_average_cosine_similarity(TFIDF_df)
    print('Computing Edit Distance')
    avg_edit_dist = get_average_edit_distance(cleaned_tweets)
    print('Computing count of pos tags')
    pos_tag_count_df = get_pos_tag_count_matrix(data)
    pos_tag_count_df.reset_index(drop=True)
    
    print('Binding other features with TFIDF matrix')
    data_matrix = pd.concat([preprocessed_df.iloc[:, :-1],avg_cosine_sim,
                             avg_familiarity_score,
                             avg_edit_dist,
                             pos_tag_count_df,
                             TFIDF_df],axis=1)
    print('Dimension of Final Data Matrix')
    print(data_matrix.shape)
    #print(data_matrix.columns.values)
    return(data_matrix)
#%%
    
#Akash's Script to Compute Lexical Chains
yy= clean_and_get_features(data)
t2 = 0
for i in yy:
    print([(FreqDist(i)).items()])
    qy = [(k,v) for k,v in ((FreqDist(i)).items()) if v>1]
    for k in qy:
        t2=t2+k[1]


df = pd.DataFrame()
tl= 0
lst=[]
for i in yy:
    # print([(FreqDist(i)).items()])
    qq = [(k,v) for k,v in ((FreqDist(i)).items()) if v>1]
    for k in qq:
        tl=tl+k[1]
    #print(tl,len(qq))
    lst.append([tl,len(qq),tl/t2])
    tl = 0

df =pd.DataFrame(data = lst,columns = ['Total Length of Exact Chain ','No of Exact Chains','Average Length of Exact Chain'])
df3 = pd.DataFrame()
yy= clean_and_get_features(data)
list3 = []
totat_syn_chain = 0
for k in yy:
    #print(k)
    synonyms = []
    No_of_chains = 0
    for i in k:
       # print(i)
        for syn in wordnet.synsets(i):
            for l in syn.lemmas():
                synonyms.append(l.name())

        synonyms = (list(set(synonyms)))
        #print(synonyms,i)
        if i in synonyms:
            synonyms.remove(i)

        for w in k:

            if w in synonyms:
                #print(w,k)
                No_of_chains = No_of_chains + 1
                #print(No_of_chains)
                totat_syn_chain = No_of_chains + totat_syn_chain
                No_of_chains = 0
        No_of_chains = 0
        synonyms = []
    list3.append(totat_syn_chain)
    totat_syn_chain = 0


df3 =pd.DataFrame(data = list3,columns = ['No Of Synonymys Chains'])
df['No_Of_Synonyms_Chains'] = df3['No Of Synonymys Chains']
df.to_csv('/storage/work/a/asb5870/EmotionClassification/emotion_classification/data/ExactSynonymnsLexical.csv',index=False)
    
#%%
#def add_pos_with_zero_counts(counter, keys_to_add):
#    for k in keys_to_add:
#        counter[k] = counter.get(k, 0)
#    return counter
#%%    
#def get_postag_count_vector(tags,possible_tags):
#    if(tags !=[]):
#        pos_counts = Counter(list(zip(*tags))[1])
#    else:
#        pos_counts = Counter({'NN':0})
#    pos_counts_with_zero = add_pos_with_zero_counts(pos_counts, possible_tags)
#    postag_count_vector = [count for tag, count in sorted(pos_counts_with_zero.most_common())]
#    return(postag_count_vector)
#%%    
#def get_pos_tag_count_matrix(cleaned_tweets):    
#    pos_tag_count_matrix = pd.DataFrame()
#    tok_and_tag = lambda x: pos_tag(word_tokenize(x))
#    tags = cleaned_tweets.apply(tok_and_tag)
#    possible_tags = sorted(set(list(zip(*chain(*tags)))[1]))
#    pos_tag_count_matrix = tags.apply(lambda x: get_postag_count_vector(x,possible_tags))
#    pos_tag_count_df = pd.DataFrame(pos_tag_count_matrix.tolist())
#    string = 'postag_'
#    possible_tags = [string + x for x in possible_tags]
#    pos_tag_count_df.columns = possible_tags
#    return(pos_tag_count_df)