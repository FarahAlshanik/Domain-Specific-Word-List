
# coding: utf-8

# In[75]:


from gensim import utils
import numpy
from sklearn import preprocessing
# Import required libraries
import pandas as pd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()
import os
# load the word2vec algorithm from the gensim library
from gensim.models import word2vec

from gensim import utils
# Import required libraries
import pandas as pd
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import numpy
import math
from scipy.spatial import distance


# In[76]:


import os
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from gensim.models import Word2Vec
#from tensorflow.python.keras import models
#from tensorflow.python.keras.layers import Dense
#from tensorflow.python.keras.layers import Dropout


# In[77]:


def shuffle(X, y):
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    return X, y


# In[24]:


import collections
import numpy as np
from itertools import combinations
import pandas as pd
import numpy as np
import argparse
from os import listdir
from os.path import isfile, join
import sys
import re
import os

import time



parser = argparse.ArgumentParser()

parser.add_argument("--input",help="sliding window for entropy", required=True)

args = parser.parse_args()

w=(args.input)


year=w.split('/')[2].split('.')[0]




data_consumer_complaint = pd.read_csv(w,encoding='latin-1')

print(w)
len(data_consumer_complaint)


train_texts=data_consumer_complaint['speech']


# In[91]:



train_labels=data_consumer_complaint['party']


# In[92]:


train_texts = np.array(train_texts)
train_labels = np.array(train_labels)
     

    # Shuffle the dataset
train_texts, train_labels = shuffle(train_texts, train_labels)


# In[93]:


trX, trY =train_texts ,train_labels

print ('Train samples shape :', trX.shape)
print ('Train labels shape  :', trY.shape)
 


# In[94]:


uniq_class_arr, counts = np.unique(trY, return_counts=True)

#print ('Unique classes :', uniq_class_arr)
#print ('Number of unique classes : ', len(uniq_class_arr))

j0=uniq_class_arr[0]
j1=uniq_class_arr[1]






# In[95]:


df = pd.DataFrame({'x':trX, 'y':trY})


# In[96]:


df.head()


# In[97]:




# function to clean text
import string

# function to clean text
def review_to_words(raw_review):
    

    
    result = re.sub(r"(\.\s)" , " ", raw_review, flags=re.I)
    result = re.sub(r"(\.\Z)" , " ", result, flags=re.I)
    result = re.sub(r"(,\Z)" , " ", result, flags=re.I)
    result = re.sub(r"(-\s)" , " ", result, flags=re.I)
    result = re.sub(r"(,\s)" , " ", result, flags=re.I)
    result=" ".join(result.split())
    result = re.sub(r"([@|\(~:{*?^%;}\!\"#<\)$>&+=])" , " ", result, flags=re.I)
    words = result.lower().split()
      
   
    return( " ".join( words ))

 


# In[98]:


#df['x'] = [review_to_words(text) for text in df['x']]


import re, string, unicodedata
import nltk
#import contractions
#import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
            text=text.lower()
            text=re.sub('\[[^]]*\]', '', text)
            text=re.sub('\[(^)]*\]', '', text)
            text=re.sub('\[{^}]*\]', '', text)
            text=re.sub('\[{\{^}\}]*\]', '', text)
            text=re.sub(r'[^\x00-\x7f]',r' ', text)
            new_words = []
            for word in text.split():
                new_word = re.sub(r'[^\w\s]', '', word)
                if new_word != '':
                    new_words.append(new_word)
     
            return ( " ".join( new_words ))

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#sample = denoise_text(sample)
#print(sample)

df['x'] = [denoise_text(text) for text in df['x']]








# In[99]:


train_texts=df['x']
train_labels=df['y']


# In[102]:


cell_jou=df.loc[df.y == uniq_class_arr[0]]
dentist_jou=df.loc[df.y == uniq_class_arr[1]]



dentist_jou_proc = dentist_jou['x']
cell_jou_proc = cell_jou['x']


# In[108]:


from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()
def tokenize(tweet):
  
        #tweet = unicode(tweet.decode('utf-8').lower())
        tweet=tweet.lower()
        tokens = tweet.split(" ")
        
        return tokens
    
    
    
def postprocess(data):
   
    data['tokens'] = data['x'].map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    #data = data[data.tokens != 'NC']
    #data.reset_index(inplace=True)
    #data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(df) 


from tqdm import tqdm
x_train=np.array(data.tokens)

import os
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(TaggedDocument (v, [label]))
    return labelized



# In[109]:


x_train = labelizeTweets(x_train, 'TRAIN')
 



# In[ ]:





# In[110]:




start_time = time.time()


tweet_w2v = Word2Vec(size=100, min_count=0,sg=1,negative=5)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)    
    


# In[111]:


words=[w for w in tweet_w2v.wv.vocab]


# In[112]:


ee=[]
for i, word in enumerate(tweet_w2v.wv.vocab):
    
    ee.append([word,tweet_w2v[word]])


# In[113]:




dis = open('/scratch2/falshan/centers/'+j0+'.txt', 'w')
for item in cell_jou_proc:
  sutf8 = item.encode('UTF-8')
  dis.write("%s\n" % sutf8)
dis.close()

dis = open('/scratch2/falshan/centers/'+j1+'.txt', 'w')
for item in dentist_jou_proc:
  sutf8 = item.encode('UTF-8')
  dis.write("%s\n" % sutf8)
dis.close()


file=open('/scratch2/falshan/centers/'+j0+'.txt')
 
wordcount_cell={}

for word in file.read().split():
    if word not in wordcount_cell:
        wordcount_cell[word] = 1
    else:
        wordcount_cell[word] += 1
        
        
file=open('/scratch2/falshan/centers/'+j1+'.txt')
 
wordcount_dentist={}

for word in file.read().split():
    if word not in wordcount_dentist:
        wordcount_dentist[word] = 1
    else:
        wordcount_dentist[word] += 1





# In[117]:


words_cell=[]
for k,v in wordcount_cell.items():
   # if(k=='cell'):
        #print(k,v)
        words_cell.append([k,v])


# In[118]:


words_cell_sorted=[]
words_cell_sorted=sorted(words_cell, key=lambda v: v[1],reverse=True)


# In[119]:


words_dentist=[]
for k,v in wordcount_dentist.items():
   # if(k=='cell'):
        #print(k,v)
        words_dentist.append([k,v])


# In[120]:


words_dent_sorted=[]
words_dent_sorted=sorted(words_dentist, key=lambda v: v[1],reverse=True)


# In[121]:


voc_dim=[]
 
for i in words:
    voc_dim.append([i,tweet_w2v[i]])
   


# In[122]:


#https://www.codingame.com/playgrounds/6233/what-is-idf-and-how-is-it-calculated
#this code for compute the idf only

#lowest idf is noise like the is a 
#Loowes idf like domain based 

def extract_features( document ):
  terms = tuple(document.lower().split())
  features = set()
  for i in range(len(terms)):
     for n in range(1,2):
         #if i+n <= len(terms):
           features.add(terms[i:i+n])
  return features





# In[123]:



def calculate_idf( documents ):
   N = len(documents)
   from collections import Counter
   tD = Counter()
   for d in documents:
      features = extract_features(d)
      for f in features:
          tD[" ".join(f)] += 1
   IDF = []
   import math
   for (term,term_frequency) in tD.items():
       term_IDF = np.log2(float(N) / term_frequency)
       IDF.append(( term_IDF, term ))
       #print(term,term_frequency)
   IDF.sort(reverse=True)
   return IDF

    


# In[124]:


#all journal

idf=[]
for (IDF, term) in calculate_idf(df['x']):
    idf.append([IDF, term])


# In[125]:


idf_cell=[]
for (IDF, term) in calculate_idf(cell_jou_proc):
    idf_cell.append([IDF, term])


# In[126]:


idf_dent=[]
for (IDF, term) in calculate_idf(dentist_jou_proc):
    idf_dent.append([IDF, term])


# In[127]:


idf_dic={}
for i in idf:
    idf_dic[i[1]]=i[0]


# In[128]:


idf_dic_cell={}
for i in idf_cell:
    idf_dic_cell[i[1]]=i[0]


# In[129]:


idf_dic_dent={}
for i in idf_dent:
    idf_dic_dent[i[1]]=i[0]


# In[130]:


from collections import defaultdict


# In[131]:


def get_tf(corpus, emb, idf, D):  #return frequency of words in corpus
    
     
   
    #D is number of dimension
    # Computing Terms' Frequency
    tf = defaultdict(int)
    for doc in corpus:
        
        tokens = doc.split()
        for word in tokens:
        #print(word)
            if word in emb:
                tf[word] += 1
            #print(tf[word])

    # Computing the centroid
   
 

         
    return tf


# In[132]:


emb={}
for i in ee:
    
    emb[i[0]]=i[1]


# In[133]:


tf=get_tf(df['x'],emb,idf_dic,100)


# In[134]:


def get_centroid_idf_new(corpus, emb, idf,tf, D):
    c=[]
    centroid = np.zeros((1, 100))
    div = 0
    #D is number of dimension
    # Computing Terms' Frequency
    

    for word in tf:
        if word in emb:
            p = tf[word] * idf[word]
                
                #c.append([word,p])
            #centroid = np.add(centroid, emb[word]*p)
            centroid = np.add(centroid, emb[word])
            c.append([word,p])
                #centroid = np.add(centroid, emb[word]*p)
            #div += p
            div+=1
    if div != 0:
        centroid = np.divide(centroid, div)
    return centroid,c


# In[135]:


tf_cell=get_tf(cell_jou_proc,emb,idf_dic_cell,100)




avg,tfidf=get_centroid_idf_new(df['x'],emb,idf_dic,tf,100)




s_tfidf=[]
s_tfidf=sorted(tfidf, key=lambda v: v[1],reverse=True)




cent_array = []
cent_array.append(np.array(avg, dtype=np.float64))




final_cent_array = np.array(cent_array, dtype=np.float64).reshape(( 100))




avg=final_cent_array


# In[148]:





tf_cell=get_tf(cell_jou_proc,emb,idf_dic_cell,100)

tf_dent=get_tf(dentist_jou_proc,emb,idf_dic_dent,100)


# In[151]:


def get_centroid_idf_new(corpus, emb, idf,tf, D):
    c=[]
    centroid = np.zeros((1, 100))
    div = 0
    #D is number of dimension
    # Computing Terms' Frequency
    

    for word in tf:
        if word in emb:
            p = tf[word] * idf[word]
                
                #c.append([word,p])
            #centroid = np.add(centroid, emb[word]*p)
            centroid = np.add(centroid, emb[word])
            c.append([word,p])
                #centroid = np.add(centroid, emb[word]*p)
            #div += p
            div+=1
    if div != 0:
        centroid = np.divide(centroid, div)
    return centroid,c


# In[152]:



def get_centroid(corpus, emb, idf,tf, D):
    
    centroid = np.zeros((1, 100))
    div=0
    
    for word in tf:
        if word in emb:
               # p = tf[word] * idf[word]
                
                #c.append([word,p])
            centroid = np.add(centroid, emb[word])
                #centroid = np.add(centroid, emb[word]*p)
            div += 1
    if div != 0:
        centroid = np.divide(centroid, div)
    return centroid


# In[153]:



def get_centroid( emb, idf,tf, D):
    
    centroid = np.zeros((1, 100))
    div=0
   
    for word in tf:
        if word in emb:
               # p = tf[word] * idf[word]
                
                #c.append([word,p])
            centroid = np.add(centroid, emb[word])
                #centroid = np.add(centroid, emb[word]*p)
            div += 1
    if div != 0:
        centroid = np.divide(centroid, div)
    return centroid


# In[154]:



avg_cell, wc=get_centroid_idf_new(cell_jou_proc,emb,idf_dic_cell,tf_cell,100)
avg_dentist, wd=get_centroid_idf_new(dentist_jou_proc,emb,idf_dic_dent,tf_dent,100)
#avg_cell=get_centroid(emb,idf_dic_cell,tf_cell,100)

#avg_dentist=get_centroid(emb,idf_dic_dent,tf_dent,100)







s_tfidf_cell=[]
s_tfidf_cell=sorted(wc, key=lambda v: v[1],reverse=True)







s_tfidf_dent=[]
s_tfidf_dent=sorted(wd, key=lambda v: v[1],reverse=True)





final_cent_array_cell = np.array(avg_cell, dtype=np.float64).reshape(( 100))

final_cent_array_dent= np.array(avg_dentist, dtype=np.float64).reshape(( 100))

avg_cell=final_cent_array_cell
avg_dent=final_cent_array_dent


# In[163]:


def dot_pro(a, b):

    dot_product = np.dot(a, b)
    #dot_product=ecul_similarity(a,b)
    return dot_product


# In[164]:


def ecul_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx = 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += math.pow(x-y,2)
         
    return math.sqrt(sumxx)


# In[165]:


mid_point=numpy.zeros(100, dtype=float)
#for i in range(100):
for j in range(100):
    mid_point[j]=(avg_cell[j]+avg_dent[j])/2
    
#vec=e(c1)-e(c2)
vec=numpy.zeros(100, dtype=float)
for j in range(100):
    vec[j]=(avg_cell[j]-avg_dent[j])

    
    
w=vec

b=-dot_pro(mid_point,w)
mag_w=np.linalg.norm(w)

voc2_ecu_dis_hyper=[]

for i,word in enumerate(voc_dim):
    voc2_ecu_dis_hyper.append([word[0],(abs(dot_pro(voc_dim[i][1],w )+b)/mag_w)])
     
    
sorted_voc2_ecu_dis_hyper=[]
#for i in range(len(dis)):
    #sorted_dis.appen
sorted_voc2_ecu_dis_hyper=sorted(voc2_ecu_dis_hyper, key=lambda v: v[1])

sorted_voc2_ecu_dis_hyper_rev=[]
#for i in range(len(dis)):

sorted_voc2_ecu_dis_hyper_rev=sorted(voc2_ecu_dis_hyper, key=lambda v: v[1],reverse=True)



end=time.time() - start_time
print("--- %s seconds ---" % (time.time() - start_time))

dataFile = open('/scratch1/falshan/politics/time_hyper/time_'+j0+'_'+j1+'_'+str(year)+'.txt', 'w')
dataFile.write(str(end)+'\n')
dataFile.close()




dist = open('/scratch2/falshan/politics/hyper_words_j4_new/words_distance_'+j0+'_'+j1+'_'+str(year)+'.txt', 'w')
for item in sorted_voc2_ecu_dis_hyper:
  dist.write("%s\n" % item)


