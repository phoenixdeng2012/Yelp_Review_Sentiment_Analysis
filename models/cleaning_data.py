
# coding: utf-8

# In[1]:

import matplotlib
matplotlib.use('Agg')
import json
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


lines=[]
pos_num=0
neg_num=0
num_total=140000

with open('../yelp_dataset/yelp_academic_dataset_review.json', 'r') as f:
    for line in f:
        json_data=json.loads(line)
        
        if json_data['stars']>3:
            if pos_num>num_total:
                continue
            pos_num+=1
        elif json_data['stars']<3:
            if neg_num>num_total:
                continue
            neg_num+=1
        else:
            continue
            
        lines.append(json_data)           
    


# In[3]:


review=pd.DataFrame(lines)
print(review.head())


# In[4]:


review['sentiment']=(review['stars']>3).astype(int)


# In[5]:


print('some examples of reviews')
print(review['text'][0])
print('\n')
print(review['text'][10])


# In[8]:


#data cleanning:
#expand contractions
#remove non alphabert words (punctuations, special characters such as @,# etc)
#lower case everything
#remove stopwords

print('start cleaning')
import contractions
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def clean(doc):
    
    doc=contractions.fix(doc)
    
    pattern_to_find = "[^a-zA-Z0-9' ]";
    pattern_to_repl = " ";
    doc=re.sub(pattern_to_find, pattern_to_repl, doc).lower()
    
    eng_stopwords = set(stopwords.words("english"))-{'not','no'};
    sentence = ' '.join(word for word in doc.split() if word not in eng_stopwords)
    
    return sentence  


# In[9]:


print(clean('I\'ve got a nanny##@@'))


# In[10]:


review['clean_text']=review['text'].apply(clean)


# In[11]:


print(review.head())


# Words can't be directly fed into the model, instead we need to encode each word to a unique numerical value. Therefore a review is transformed from being an array of words to an array of integer values.
# 

# In[12]:


print('sentence length analysis')
def lenth(sentence):
    return len(sentence.split(' '))
review['len']=review['clean_text'].apply(lenth)


# In[13]:


ax=review['len'].plot.hist(bins=100)
ax.set_xlim([0,150])
plt.savefig('length_distri.png')
print('done cleaning')
save=review[['text','clean_text','sentiment']]
save.to_csv('./review.csv')

# It seems that 20-60 is the most common length. We can choose to cap at 60 words for our model.
