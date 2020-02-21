#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 00:30:46 2020

@author: userselu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#exclude '' double quote in review as delimiter
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter ='\t', quoting =3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range (0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer();
    
    #review = [word for word in review if not word in stopwords.words('english')]
    review = [ps.stem(word) for word in review if not word in set( stopwords.words('english'))]
    review =' '.join(review)
    corpus.append(review)

# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500);

X = cv.fit_transform(corpus).toarray()
