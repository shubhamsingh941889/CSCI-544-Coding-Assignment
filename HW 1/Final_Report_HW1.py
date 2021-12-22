
#python version 3.9.7
import os
import sys
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

amazon_reviews = pd.read_csv(os.path.join(sys.path[0],"amazon.tsv"), sep='\t', error_bad_lines=False,warn_bad_lines=False)

rev_rat = amazon_reviews[['star_rating','review_body']]

# Statistics of ratings
rev_rat['star_rating'].value_counts()

rev_rat=rev_rat.dropna() # Dropping reviews that have type NaN as this will cause when randomly selecting classes with a particular class
rev_rat = rev_rat.reset_index(drop=True) # reseting the index tso that it covers up for the dropped values

rev_rat['star_rating']=rev_rat['star_rating'].astype(int) # convert values of star_rating to int so that we can use numpy where clause
rev_rat['class']=np.where(rev_rat['star_rating']<3,0,1) # set class based on the given requirements
rev_rat['class']=np.where(rev_rat['star_rating']==3,3,rev_rat['class']) # we will now change class of ratings 3 as on previous step we added it to class 1

 #rev_rat.head(50)

# Statistics of ratings after classes
ans = rev_rat['class'].value_counts()
#print(ans)
print('Class 1:',ans[0],', Class 0:',ans[1],', Class Neutral or 3:',ans[3])


# Discarding reviews with ratings 3 and class 3
rev_rat = rev_rat.loc[rev_rat["class"] != 3]
#rev_rat.head(50)

#rev_rat['class'].value_counts()

class0_random = rev_rat.star_rating[rev_rat['class'].eq(0)].sample(100000).index #randomly select class 0 sample
class1_random = rev_rat.star_rating[rev_rat['class'].eq(1)].sample(100000).index
rev_rat = rev_rat.loc[class0_random.union(class1_random)] #unify both the dataframes
#display(rev_rat['class'].value_counts())

#rev_rat.head(50)

# 3 samples before Data Cleaning
rev_rat.head(3)

#Average char length in review_body before data cleaning
from statistics import mean
char_len=[len(char) for char in rev_rat['review_body']]
print(mean(char_len))

rev_rat['review_body'] =rev_rat['review_body'].str.lower()

rev_rat['review_body'] = rev_rat['review_body'].replace(r'<.*?>+', '', regex=True) #removes html tags
rev_rat['review_body'] = rev_rat['review_body'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True) #removes url

rev_rat['review_body'] = rev_rat['review_body'].replace(r'[^a-zA-Z\' ]+', '', regex=True) #I am not removing apostrophe as they will help in contractions
#rev_rat.head(5)

rev_rat['review_body']=rev_rat['review_body'].replace(r' +',' ',regex=True)

import sys

import contractions

rev_rat['review_body']= [contractions.fix(words) for words in rev_rat['review_body']]
#rev_rat.head(5)

#Average char length in review_body after data cleaning
from statistics import mean
char_len_after=[len(char) for char in rev_rat['review_body']]
print(mean(char_len_after))

from nltk.corpus import stopwords
st_words = stopwords.words('english')
rev_rat['review_body'] = rev_rat['review_body'].apply(lambda x: ' '.join([i for i in x.split() if i not in (st_words)]))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
rev_rat['review_body']=rev_rat['review_body'].apply(lambda x: " ".join([lemmatizer.lemmatize(j) for j in nltk.word_tokenize(x)]))


# 3 samples after preprocessing
rev_rat.head(3)

#Average char length in review_body after data preprocessing
from statistics import mean
char_len_af_pre=[len(char) for char in rev_rat['review_body']]
print(mean(char_len_af_pre))

print('Change in Avg char length from normal to Data Cleaning')
print(str(mean(char_len))+","+str(mean(char_len_after)))

print('Change in Avg char length from Data Cleaning to Data Preprocessing')
print(str(mean(char_len_after))+","+str(mean(char_len_af_pre)))

from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()
tf_idf_ft = vec.fit_transform(rev_rat['review_body'])

from sklearn.model_selection import train_test_split
class_val=rev_rat['class']
amazoz_data = train_test_split(tf_idf_ft,
                            class_val,
                            test_size=0.2)

train_x, test_x, train_y, test_y = amazoz_data

from sklearn.linear_model import Perceptron
perc = Perceptron(random_state=53,
               max_iter=100000,
               tol = 0.0001
               )
perc.fit(train_x, train_y)

from sklearn.metrics import classification_report

final_report = classification_report(perc.predict(test_x), test_y)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print('Perceptron Model')
#print(accuracy_score(test_y,perc.predict(test_x) ) * 100,'%',
#',',precision_score(test_y, perc.predict(test_x), average='macro') *100,'%',
#',',recall_score(test_y,perc.predict(test_x) ) * 100,'%',
#',',f1_score(test_y,perc.predict(test_x) ) * 100,'%',
#',',accuracy_score(train_y,perc.predict(train_x) ) * 100,'%',
#',',precision_score(train_y, perc.predict(train_x), average='macro') *100,'%',
#',',recall_score(train_y,perc.predict(train_x) ) * 100,'%',
#',',f1_score(train_y,perc.predict(train_x) ) * 100,'%')

print('Accuracy of Test: ', accuracy_score(test_y,perc.predict(test_x) ) * 100,'%')
print('Precision of Test: ',precision_score(test_y, perc.predict(test_x), average='macro') *100,'%')
print('Recall of Test: ', recall_score(test_y,perc.predict(test_x) ) * 100,'%')
print('F1 Score of Test: ', f1_score(test_y,perc.predict(test_x) ) * 100,'%')

print('Accuracy of Train: ', accuracy_score(train_y,perc.predict(train_x) ) * 100,'%')
print('Precision of Train: ',precision_score(train_y, perc.predict(train_x), average='macro') *100,'%')
print('Recall of Train: ', recall_score(train_y,perc.predict(train_x) ) * 100,'%')
print('F1 Score of Train: ', f1_score(train_y,perc.predict(train_x) ) * 100,'%')


from sklearn.svm import LinearSVC
svm_model = LinearSVC(random_state=8)
svm_model.fit(train_x, train_y)

print('SVM Model')
#print(accuracy_score(test_y,svm_model.predict(test_x) ) * 100,'%',
#',',precision_score(test_y, svm_model.predict(test_x), average='macro') *100,'%',
#',',recall_score(test_y,svm_model.predict(test_x) ) * 100,'%',
#',',f1_score(test_y,svm_model.predict(test_x) ) * 100,'%',
#',',accuracy_score(train_y,svm_model.predict(train_x) ) * 100,'%',
#',',precision_score(train_y, svm_model.predict(train_x), average='macro') *100,'%',
#',',recall_score(train_y,svm_model.predict(train_x) ) * 100,'%',
#',',f1_score(train_y,svm_model.predict(train_x) ) * 100,'%')

print('Accuracy of Test: ', accuracy_score(test_y,svm_model.predict(test_x) ) * 100,'%')
print('Precision of Test: ',precision_score(test_y, svm_model.predict(test_x), average='macro') *100,'%')
print('Recall of Test: ', recall_score(test_y,svm_model.predict(test_x) ) * 100,'%')
print('F1 Score of Test: ', f1_score(test_y,svm_model.predict(test_x) ) * 100,'%')

print('Accuracy of Train: ', accuracy_score(train_y,svm_model.predict(train_x) ) * 100,'%')
print('Precision of Train: ',precision_score(train_y, svm_model.predict(train_x), average='macro') *100,'%')
print('Recall of Train: ', recall_score(train_y,svm_model.predict(train_x) ) * 100,'%')
print('F1 Score of Train: ', f1_score(train_y,svm_model.predict(train_x) ) * 100,'%')

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=10000,solver='saga',tol=0.0001)
log_reg.fit(train_x, train_y)

print('Logistic Regression Model')
#print(accuracy_score(test_y,log_reg.predict(test_x) ) * 100,'%',
#',',precision_score(test_y, log_reg.predict(test_x), average='macro') *100,'%',
#',',recall_score(test_y,log_reg.predict(test_x) ) * 100,'%',
#',',f1_score(test_y,log_reg.predict(test_x) ) * 100,'%',
#',',accuracy_score(train_y,log_reg.predict(train_x) ) * 100,'%',
#',',precision_score(train_y, log_reg.predict(train_x), average='macro') *100,'%',
#',',recall_score(train_y,log_reg.predict(train_x) ) * 100,'%',
#',',f1_score(train_y,log_reg.predict(train_x) ) * 100,'%')


print('Accuracy of Test: ', accuracy_score(test_y,log_reg.predict(test_x) ) * 100,'%')
print('Precision of Test: ',precision_score(test_y, log_reg.predict(test_x), average='macro') *100,'%')
print('Recall of Test: ', recall_score(test_y,log_reg.predict(test_x) ) * 100,'%')
print('F1 Score of Test: ', f1_score(test_y,log_reg.predict(test_x) ) * 100,'%')

print('Accuracy of Train: ', accuracy_score(train_y,log_reg.predict(train_x) ) * 100,'%')
print('Precision of Train: ',precision_score(train_y, log_reg.predict(train_x), average='macro') *100,'%')
print('Recall of Train: ', recall_score(train_y,log_reg.predict(train_x) ) * 100,'%')
print('F1 Score of Train: ', f1_score(train_y,log_reg.predict(train_x) ) * 100,'%')

from sklearn.naive_bayes import MultinomialNB
multi_nb = MultinomialNB()
multi_nb.fit(train_x, train_y)

print('Multinomial Naive Bayes Model')
#print(accuracy_score(test_y,multi_nb.predict(test_x) ) * 100,'%',
#',',precision_score(test_y, multi_nb.predict(test_x), average='macro') *100,'%',
#',',recall_score(test_y,multi_nb.predict(test_x) ) * 100,'%',
#',',f1_score(test_y,multi_nb.predict(test_x) ) * 100,'%',
#',',accuracy_score(train_y,multi_nb.predict(train_x) ) * 100,'%',
#',',precision_score(train_y, multi_nb.predict(train_x), average='macro') *100,'%',
#',',recall_score(train_y,multi_nb.predict(train_x) ) * 100,'%',
#',',f1_score(train_y,multi_nb.predict(train_x) ) * 100,'%')

print('Accuracy of Test: ', accuracy_score(test_y,multi_nb.predict(test_x) ) * 100,'%')
print('Precision of Test: ',precision_score(test_y, multi_nb.predict(test_x), average='macro') *100,'%')
print('Recall of Test: ', recall_score(test_y,multi_nb.predict(test_x) ) * 100,'%')
print('F1 Score of Test: ', f1_score(test_y,multi_nb.predict(test_x) ) * 100,'%')

print('Accuracy of Train: ', accuracy_score(train_y,multi_nb.predict(train_x) ) * 100,'%')
print('Precision of Train: ',precision_score(train_y, multi_nb.predict(train_x), average='macro') *100,'%')
print('Recall of Train: ', recall_score(train_y,multi_nb.predict(train_x) ) * 100,'%')
print('F1 Score of Train: ', f1_score(train_y,multi_nb.predict(train_x) ) * 100,'%')


