#!/usr/bin/env python
# coding: utf-8

# # Search Engine

# #### Amparo Alías <br> Judit Boronat<br> Marcelo Sanchez

# **We get the data obtained from the twitter scrapper and process it and rank it according to a query. <br>This way, given a query by the user, the search engine returns the topN ranked tweets**

# In[1]:


get_ipython().system('pip install unidecode')


# ### Import required libraries 

# In[2]:


import nltk
nltk.download('stopwords');


# In[3]:


import json
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import defaultdict,Counter
from array import array
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import numpy as np
import collections
from numpy import linalg as la
import unidecode
import re


# In[4]:


import statistics as st


# ### Read data obtained from the scrapper

# In[11]:


#Whole dataset
with open("../search_engine/tweets.json", "rb") as f:
    data = f.readlines()
    data = [json.loads(str_) for str_ in data]
df_tweets = pd.DataFrame.from_records(data)


# #### Remove RTs

# In[12]:


df_tweets = df_tweets.loc[~df_tweets['full_text'].str.startswith("RT")]
df_tweets.reset_index(drop = True, inplace=True)


# #### Remove duplicates

# In[13]:


df_tweets = df_tweets.drop_duplicates(subset='full_text', keep="last")
df_tweets.reset_index(drop = True, inplace=True)


# ### Data processing

# In[15]:


def getTerms(text):
    """
    Preprocess the article text (title + body) removing stop words, stemming,
    transforming in lowercase and return the tokens of the text.
    
    Argument:
    line -- string (text) to be preprocessed
    
    Returns:
    line - a list of tokens corresponding to the input text after the preprocessing
    """
        
    stemming = PorterStemmer()
    stops = set(stopwords.words("english"))
    
    ## Transform in lowercase:
    text =  text.lower() 
    
    ## Tokenize the text to get a list of terms:
    text =  text.split(" ") 
    
    ## Remove accents:
    text = [unidecode.unidecode(x) for x in text]
    
    ## Remove emojis:
    text = [x.encode(encoding = 'ascii', errors = 'ignore').decode(encoding = 'ascii', errors='ignore') for x in text ]
    
    ## Replace characters like it\'s by its:
    text = [re.sub(r"\'","", x) for x in text]
    
    ## Replace *, ?, ... by spaces
    text = [re.sub(r'[^\w]', " ", x) for x in text]

    ## Eliminate the stopwords:
    text = [x for x in text if x not in stops]  
    
    ## Perform stemming 
    text =  [stemming.stem(word) for word in text]   
    
    return text


# ### Indexing with inverted-index

# In[16]:


def create_index_tfidf(tweets, numDocuments):
    
    index=defaultdict(list)
    tf=defaultdict(list) #term frequencies of terms in documents (documents in the same order as in the main index)
    df=defaultdict(int)         #document frequencies of terms in the corpus
    idf=defaultdict(float)
    
    for i in range(numDocuments):
        text = tweets.loc[i,'full_text'] #tweet
        line_arr = text.split(" ") #array with words of the tweet separated
        terms = getTerms(' '.join(line_arr)) #the phrse with the preprocessing
        
        #print(terms)
        
        termdictPage={}
        for position, term in enumerate(terms): 
            try:
                # if the term is already in the dict append the position to the corrisponding list
                termdictPage[term][1].append(position) 
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                termdictPage[term]=[i, array('I',[position])] #'I' indicates unsigned int (int in python)
        #print(termdictPage)
        
        
        #normalize term frequencies
        norm=0
        for term, posting in termdictPage.items(): 
            norm+=len(posting[1])**2
        norm=math.sqrt(norm)
        
        #calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in termdictPage.items():     
            tf[term].append(np.round(len(posting[1])/norm,4))  
            df[term]= df[term] +1  # increment df for current term
        
        #merge the current page index with the main index
        for termpage, postingpage in termdictPage.items():
            index[termpage].append(postingpage)
            
    # Compute idf following the formula (3) above. HINT: use np.log
    for term in df:
        idf[term] = np.round(np.log(float(numDocuments/df[term])),4)
    

    return index, tf, df, idf        
    


# In[17]:


numDocuments = len(df_tweets)
index, tf, df, idf = create_index_tfidf(df_tweets, numDocuments)


# ### Our score based on popularity

# In[19]:


def ourScoring(docScores):
    df = pd.DataFrame(columns = ['numdoc', 'score', 'num_ret', 'num_fav'])

    for score, numdoc in docScores:
        df = df.append({'numdoc':numdoc, 
                   'score':score, 
                   'num_ret':df_tweets.loc[numdoc]['retweet_count'], 
                   'num_fav':df_tweets.loc[numdoc]['favorite_count']}, 
                       ignore_index = True)

    mean_score = st.mean(df['score'])

    df = df.set_index('numdoc')
    df = df.sort_values(by='num_ret', ascending=False)
    count = collections.Counter(df['num_ret'])

    l = np.linspace(0,0.5,len(count))
    l = sorted(l, reverse= True)


    x = 0
    for i, j in count.items():
        count[i] = l[x]
        x += 1

    newscores = []
    for i in df.index:
        newscore = df.loc[i]['score'] + (mean_score*count[df.loc[i]['num_ret']])
        newscores.append(newscore)

    df['newscore'] = newscores


    df = df.sort_values(by='num_fav', ascending=False)
    count = collections.Counter(df['num_fav'])
    l = np.linspace(0,0.5,len(count))
    l = sorted(l, reverse= True)


    x = 0
    for i, j in count.items():
        count[i] = l[x]
        x += 1

    newscores = []
    for i in df.index:
        newscore = df.loc[i]['newscore'] + (mean_score*count[df.loc[i]['num_fav']])
        newscores.append(newscore)

    df['lastscore'] = newscores

    df = df.sort_values(by='lastscore', ascending=False)
    
    docScores = []
    for i in df.index:
        docScores.append([df.loc[i]['lastscore'], i])
        
    return docScores


# ### Ranking the tweets

# In[20]:


def rankDocuments(terms, docs, index, idf, tf,scoring):
    
    print("\nDoing the ranking.. ")
    docVectors=defaultdict(lambda: [0]*len(terms)) 
    queryVector=[0]*len(terms)    

    # Compute the norm for the query tf
    query_terms_count = collections.Counter(terms) # get the frequency of each term in the query. 
    query_norm = la.norm(list(query_terms_count.values()))
    
    for termIndex, term in enumerate(terms): #termIndex is the index of the term in the query
        if term not in index:
            continue
                    
        # Compute tf*idf(normalize tf as done with documents)
        queryVector[termIndex]=query_terms_count[term]/query_norm * idf[term]
        
        # Generate docVectors for matching docs
        for docIndex, (doc, postings) in enumerate(index[term]):

            if doc in docs:
                docVectors[doc][termIndex]=tf[term][docIndex] * idf[term]  # TODO: check if multiply for idf

        
    # calculate the score of each doc
    # compute the cosine similarity between queryVector and each docVector:
    docScores=[ [np.dot(curDocVec, queryVector), doc] for doc, curDocVec in docVectors.items() ]
    docScores.sort(reverse=True)

    if scoring == 2:
        print("Computing our score.. ")
        start_time = time.time()
        docScores = ourScoring(docScores)
        end_time = time.time()
        print("Time spent: {} seconds" .format(np.round(end_time - start_time,2)))
    resultDocs=[x[1] for x in docScores]
    
    if len(resultDocs) == 0:
        print("No results found, try again")
        query = input()
        docs = search_tweets(query, index, scoring) 
        
    return resultDocs


# ### Search for the relevant tweets

# In[21]:


def search_tweets(query, index, scoring):
    '''
    output is the list of documents that contain any of the query terms. 
    So, we will get the list of documents for each query term, and take the union of them.
    '''
    query=getTerms(query)

    docs=set()
    for term in query:
        if len(docs) == 0:
            docs = set([posting[0] for posting in index[term]])
        else:
            try:
            # store in termDocs the ids of the docs that contain "term"                        
                termDocs=set([posting[0] for posting in index[term]])
                
            # docs = docs Union termDocs
                docs = docs.intersection(termDocs)
                
            except:
            #term is not in index
                docs = docs
    docs = list(docs)
    ranked_docs = rankDocuments(query, docs, index, idf, tf,scoring)   
    
    return ranked_docs
    


# ### Interaction with the user

# In[22]:


def userInteraction(data):
    print("Insert: \n1 if you want to use the inverted index \n2 if you want to use our score")
    scoring = input()
    scoring = int(scoring)

    print("\nHow many returned tweets you want?")
    top = input()
    top = int(top)

    print("\nInsert your query:")
    query = input()
    ranked_docs = search_tweets(query, index, scoring)  


    print("\n======================\nTop {} results out of {} for the searched query:\n".format(top, len(ranked_docs)))
    results = pd.DataFrame(columns=['tweet', 'username', 'date', 'hashtags', 'likes', 'retweets', 'url'])
    count = 0
    for d_id in ranked_docs[:top] :
        results.loc[count,'tweet'] = data.loc[d_id, "full_text"]
        results.loc[count,'username'] = data.loc[d_id, "user"]['screen_name']
        results.loc[count,'date'] = data.loc[d_id, "created_at"]
        results.loc[count,'hashtags'] = [data.loc[0,'entities']['hashtags'][i]['text'] for i in range(len(data.loc[0,'entities']['hashtags']))]
        results.loc[count,'likes'] = data.loc[d_id, "favorite_count"]
        results.loc[count,'retweets'] = data.loc[d_id, "retweet_count"]
        results.loc[count,'url'] = "https://twitter.com/twitter/statuses/"+str(data.loc[d_id, "id"])
        count +=1

    return results


# In[ ]:




