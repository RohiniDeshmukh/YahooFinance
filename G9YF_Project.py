#!/usr/bin/env python
# coding: utf-8

# In[3]:


driver_path = r'C:\Users\senja\Desktop\final\chromedriver.exe'
#driver = webdriver.Chrome(driver_path)


# In[4]:


#Importing Libraries
import sqlite3
import re
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time


# In[5]:


#Webdriver
browser = webdriver.Chrome(driver_path)
browser.maximize_window()

#User Input
company_name = input("Enter the company name: ")

url ="https://finance.yahoo.com/quote/"+ company_name + '/history/'
browser.get(url)
time.sleep(3)

#Popup Close
#closepopup = browser.find_element(By.XPATH,"//*[@id='myLightboxContainer']/section/button[1]")
#closepopup.click()
#time.sleep(2)


# In[61]:


# Scroll
for _ in range(3):
    yahooend = browser.find_element(By.XPATH, "//*[@id='Col1-1-HistoricalDataTable-Proxy']/section/div[2]/table/tfoot/tr/td/span[2]/span")
    browser.execute_script("arguments[0].scrollIntoView();", yahooend)
    time.sleep(2)

# Get the entire HTML code of the page
html = browser.page_source

# Find all matches with the given pattern
pattern = r'<tr[^>]*>((?:\s*<td[^>]*>\s*<span[^>]*>([^<]+)<\/span>\s*<\/td>)+)\s*<\/tr>'
matches = re.findall(pattern, html, re.DOTALL)

# Connect to the SQLite database
conn = sqlite3.connect("historical_data.db")

# Create the table for the company
create_table_query = f"""
CREATE TABLE IF NOT EXISTS {company_name} (
    date TEXT PRIMARY KEY,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER
);
"""
conn.execute(create_table_query)

# Insert the scraped data into the database
for match in matches:
    span_contents = re.findall(r'(?<=<span>).*?(?=<\/span>)', match[0])
    if len(span_contents) == 7:
        date, open_, high, low, close, adj_close, volume = span_contents
        insert_query = f"""
        INSERT OR IGNORE INTO {company_name} (date, open, high, low, close, adj_close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        conn.execute(insert_query, (date, open_, high, low, close, adj_close, volume))

# Commit the changes and close the connection
conn.commit()
conn.close()


price_element = browser.find_element(By.XPATH,"//*[@id='quote-header-info']/div[3]/div[1]/div[1]/fin-streamer[1]")
price = price_element.text
print("The current stock price of " + company_name + " is: " + price)


# ## Connecting to Database

# In[62]:


import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# connect to database and fetch data
conn = sqlite3.connect("historical_data.db")
df = pd.read_sql_query(f"SELECT * FROM {company_name}", conn)


#Convert to CSV
df.to_csv('stock.csv', index = False)


# ## Descriptive Analysis

# In[63]:


#Get the Dataset, clean and change the date format
from datetime import datetime
df=pd.read_csv('stock.csv',na_values=['null'],index_col='date',parse_dates=True,infer_datetime_format=True)
df.head(20)


# In[64]:


#Checking for null values
df.isna().sum()


# In[65]:


#Data shape
df.shape


# In[66]:


#Data Columns
df.columns


# In[67]:


#Data Info
df.info()


# In[68]:


#Descriptive Analysis
df.describe().T


# In[69]:


#Correlation Check
df.corr()


# ## Visualization

# In[70]:


#Plotting close, open, high, low, volum over time

df1=df
df1['date'] = pd.to_datetime(df1.index)
stock_data = df1.set_index('date')
close_px = stock_data['adj_close']
open_px = stock_data['open']
high_px = stock_data['high']
low_px = stock_data['low']
plt.ylabel("prices")
plt.xlabel("Date")
plt.title(f"{company_name} Adjusted Close, Open, High and Low Price over Time")
plt.plot(stock_data['adj_close'],label='Adjusted Closing Price', color='green')
plt.plot(stock_data['open'],label='Open', color='red')
plt.plot(stock_data['high'],label='High', color='yellow')
plt.plot(stock_data['low'],label='Low', color='cyan')
plt.legend(loc=2)
plt.show()


# In[71]:


#Plotting close, open, high, low individually over time
stock_data.plot(subplots = True, figsize = (8, 8))
plt.suptitle(f'Open,High,Low,Close,Adj Close prices of {company_name}', fontsize=12, color='black')
plt.legend(loc = 'best')
plt.show()


# In[72]:


#volume for the time period
#stock_data["volume"] = stock_data["volume"].astype(float)
#stock_data.volume.plot()
stock_data["volume"] = stock_data["volume"].str.replace(',', '').astype(float)
stock_data.volume.plot()


# In[73]:


#Plotting closing price and volume over time
top = plt.subplot2grid((4,4), (0, 0), rowspan=3, colspan=4)
top.plot(stock_data.index, stock_data["adj_close"])
plt.title(f"{company_name} closing price over Time")

bottom = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4)
bottom.bar(stock_data.index, stock_data['volume'])
plt.title(f"\n{company_name} Trading Volume", y=-0.60)

plt.gcf().set_size_inches(15,8)


# ## Moving Average
# #### This is used to to check the impact of fluctuations on the stock price overtime.

# In[74]:


# 30-day Moving Average.
stock_data['mavg_30_days'] = stock_data.iloc[:,4].rolling(window=30).mean()
plt.plot(stock_data['mavg_30_days'],label='30 days Moving Average', color='red')
plt.plot(stock_data['adj_close'],label='Adjusted Closing Price', color='black')
plt.legend()
plt.gcf().set_size_inches(15,8)


# ## Regression analysis
# #### Setting the Target Variable and Selecting the Features

# In[76]:



#Import Library
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


#Specify independent variables (x) and dependent variables (y) from data
#Define Target Variable
y = df[['close']].to_numpy()


#Define Input variables/Features
x = df[['open', 'high', 'low']].to_numpy()


#Create and fit a linear regression model
model = LinearRegression()
model.fit(x, y)

#find out the coefficient and the R‐squared  (R2) value.
coeff = model.coef_
R_square = model.score(x, y)
print('The coefficient is %s and the R‐square is %s.' % (coeff, R_square))



#OLS linear regression
X=df[['open', 'high', 'low']].to_numpy()
Y=df[['close']].to_numpy()
X = sm.add_constant(X)
Model = sm.OLS(Y,X)
results = Model.fit()
print(results.summary())





# ## Text Analysis WordCount

# In[77]:


#Loading URL
import requests, re
response=requests.get("https://finance.yahoo.com/quote/"+ company_name)
html_source=str(response.content)


# In[78]:


#Retrieving comments
re_comments = '<p class=".*?">(.*?).<'
comments_list = re.findall(re_comments, html_source)
output_file=open("comments.txt","w")
for comment in comments_list:
    print(comment, file=output_file)
    
output_file.close()


# In[80]:


#Create a word cloud with WordCloud
from os import path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#reading the file into a text sring
text = open("comments.txt","r").read()

# Generate a word cloud image
wc = WordCloud()
wc.generate(text)

# Display the generated image using matplotlib
plt.imshow(wc)
plt.axis("off")
plt.show()


# In[81]:


from os import path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#reading the file into a text sring
text = open("comments.txt","r").read()

#update the STOPWORDS
stopwords = set(STOPWORDS)
stopwords.update(["s", "xe2","x80","x99t","x99s"])

# Generate a word cloud image
wc = WordCloud(stopwords=stopwords)
wc.generate(text)


# Display the generated image using matplotlib
plt.imshow(wc)
plt.axis("off")
plt.show()


# In[82]:


from os import path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#read the whole file into a text sring
text = open("comments.txt","r").read()

#update the STOPWORDS
stopwords = set(STOPWORDS)
stopwords.update(["s", "xe2","x80","x99t","x99s"])

#set a circle mask
import numpy as np
x, y = np.ogrid[:300, :300]

#Is point(x,y) more than 150 pixels away from the center?
mask = (x - 150) ** 2 + (y - 150) ** 2 > 150 ** 2
mask = 255 * mask.astype(int)

# Generate a word cloud image
wc = WordCloud(stopwords=stopwords,background_color="white",                mask=mask,                max_font_size=40)
wc.generate(text)

# Display the generated image using matplotlib
plt.imshow(wc)
plt.axis("off")
plt.show()


# In[83]:


#Create a word cloud with WordCloud

from os import path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#read the whole file into a text sring
text = open("comments.txt","r").read()

#update the STOPWORDS
stopwords = set(STOPWORDS)
stopwords.update(["s", "xe2","x80","x99t","x99s"])

#set a circle mask
import numpy as np
x, y = np.ogrid[:300, :300]

#Is point(x,y) more than 150 pixels away from the center?
mask = (x - 150) ** 2 + (y - 150) ** 2 > 150 ** 2
mask = 255 * mask.astype(int)

# Generate a word cloud image
wc = WordCloud(background_color="white",                mask=mask,                stopwords=stopwords,                max_font_size=40)
wc.generate(text)

# Display the generated image using matplotlib
plt.imshow(wc)
plt.axis("off")
plt.show()


# ## Text Analysis NLTK

# In[84]:


import nltk
#nltk.download()


# In[85]:


#Reading file the file into a text string
input_file=open("comments.txt","r")
#create a list of all words
all_words=[]
for line in input_file:
    tokens = nltk.word_tokenize(line)
    all_words.extend(tokens)
    
freq_dic=nltk.FreqDist(all_words)
print(freq_dic.most_common(1000))

input_file.close()


# In[86]:


import nltk
from nltk.corpus import stopwords
import string
#remove punctuations and stopword
spws = list(set(stopwords.words('english')))
for token in spws + list(string.punctuation) + ['``', "''"]:
    if token in freq_dic: 
        freq_dic.pop(token)
print(freq_dic.most_common(1000))


# In[88]:


#Checking frequency of words

print("Investment --", freq_dic["investment"])
print("Price --", freq_dic["Price"])
print("Finance --", freq_dic["Finance"])
print("market --", freq_dic["market"])


# ## NLTK Sentiment Analysis
# ****Lexicon based-Approach****

# In[89]:


#reading the file into a text sring
text = open("comments.txt","r").read()

#create a list of all words
all_words = nltk.word_tokenize(text)

#Part of Speech Tagging
tagged_words = nltk.pos_tag(all_words)

#Named Entity Recognition
entities=nltk.chunk.ne_chunk(tagged_words)    
print(entities) 


# In[90]:


#maping PennBank pos tags to pos tags to SWN tags
def penn_to_swn(tag):
    if tag.startswith('JJ'):
        return 'a'
    elif tag.startswith('NN'):
        return 'n'
    elif tag.startswith('RB'):
        return 'r'
    elif tag.startswith('VB'):
        return 'v'
    else:
        return False


# In[91]:


#Importing Libraries
import nltk
from nltk.corpus import sentiwordnet as swn

#reading the file into a text sring
text = open("comments.txt","r").read()

#Extracting words and parsing for sentiment analysis
news_pos = []
news_neg =[]
news_obj =[]
words =[]
for word in text.split():
    tagged_words = nltk.pos_tag(nltk.word_tokenize(word))
    words.append(tagged_words)
    pos,neg,obj,w_count=0,0,0,0
    for w in tagged_words: # w is a tuple (word, POS)
        penn_POS = w[1]
        swn_POS =penn_to_swn(penn_POS)
        try:
            synset = swn.senti_synset(w[0]+'.'+swn_POS+'.01')
            pos += synset.pos_score()
            neg += synset.neg_score()
            obj += synset.obj_score()
            w_count+=1
        except:
            pass
    if (w_count > 0):
        print("Sentence Level Positive Score = "+str(pos/w_count))
        print("Sentence Level Negative Score = "+str(neg/w_count))
        print("Sentence Level Objective Score = "+str(obj/w_count))
        news_pos += [pos/w_count]
        news_neg +=[neg/w_count]
        news_obj +=[obj/w_count]
print(news_pos)


# In[92]:


#Graphical view of sentiment
#Import Library
import numpy as np
import matplotlib.pyplot as plt

#createing data points
y = np.array(news_pos)
x = np.linspace(0, len(news_pos)-1, len(news_pos))

#preparing figure
plt.figure(figsize = (4,3))
ax = plt.axes()

ax.scatter(x,y)
ax.plot(x,y)
plt.show()


# In[93]:


#Creating dataframe and Descriptive analysis

import pandas as pd
pos = pd.DataFrame(y)
pos.describe()


# In[94]:


#Wordcloud of Objective Sentiment
# Flatten the list of tuples into a single list
flat_words = [word for sublist in words[:800000] for word in sublist]

# Convert each tuple element to a string
flat_words = [str(word) for word in flat_words]

# Plot a cloud of words for objective sentiments
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=1000, width=1600, height=800, collocations=False).generate(" ".join(flat_words))
plt.imshow(wc)


# In[ ]:




