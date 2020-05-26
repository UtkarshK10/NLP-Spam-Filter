import sys
import nltk
import sklearn


import pandas as pd
import numpy as np

df= pd.read_table('SMSSpamCollection',header= None, encoding='utf-8')


classes = df[0]
print(classes.value_counts())

#Preprocess the data

"""
0= ham
1=spam
for this we use label encoder
"""
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
Y=encoder.fit_transform(classes)


#store the sms data
text_messages = df[1]




#replace email addresses with emailaddr
processed= text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddr')

#replace urls with webaddress
processed= processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')

#replace money symbols with 'moneysymb'
processed=processed.str.replace(r'£|\$','moneysymb')

#replace 10 digit number with 'phonenumber'
processed= processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumber')

#replace normal numbers with 'numbr' 
processed=processed.str.replace(r'\d+(\.\d+)?','numbr')



#remove punctuation

processed=processed.str.replace(r'[^\w\d\s]','')
processed=processed.str.replace(r'\s+',' ')
processed=processed.str.lower()


# remove stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
processed=processed.apply(lambda x : ' '.join(term for term in x.split() if term not in stop_words ))

# Stemming - like,likes,liked ~like
ps=nltk.PorterStemmer()
processed=processed.apply(lambda x : ' '.join(ps.stem(term) for term in x.split()))



#Tokenizing
nltk.download('punkt')
from nltk.tokenize import word_tokenize

all_words=[]

for message in processed:
    words=word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words= nltk.FreqDist(all_words)


#print the total number of words  and 15 most common words
'''
print('Number of words:{}'.format(len(all_words)))
print('Most Common Words:{}'.format(all_words.most_common(15)))
'''

#using the 1500 most common word as features
word_features=list(all_words.keys())[:1500]


#defining find a feature function
def find_features(message):
    words=word_tokenize(message)
    features={}
    for word in word_features:
        features[word]=(word in words)
    return features

#example
features = find_features(processed[0])
for key,value in features.items():
    if value == True:
        print(key)
        
# zipper method for appending i/p - o/p
def zipper(x, y):
	size = len(x) if len(x) < len(y) else len(y)
	retList = []
	for i in range(size):
		retList.append((x[i], y[i]))
	return retList    

  
#find features for all this messages
messages = zipper(processed,Y)

#define a seed for reproductibility
seed=1
np.random.seed=seed
np.random.shuffle(messages)
featuresets=[(find_features(text),label) for (text,label)  in messages]

#split training and testing data using sklearn
from sklearn import model_selection
training,testing = model_selection.train_test_split(featuresets,test_size=0.25,random_state=seed)
'''
print('Training: {}'.format(len(training)))
print('Testing: {}'.format(len(testing)))
'''

#Scikitlearn classifiers with nltk

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


#Define models to train and comparing best model on its accuracy
names=['K Nearest Neighbors','Decision Tree','Random Forest','Logistic Regression','SGD Classifier','Naive Bayes','SVM Linear']
classifiers=[
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter=100),
    MultinomialNB(),
    SVC(kernel='linear')
     
    ]

models = zipper(names,classifiers)

#Wrap models in nltk and find their accuracy then select best method
from nltk.classify.scikitlearn import SklearnClassifier

for name,model in models:
    nltk_model=SklearnClassifier(model)
    nltk_model.train(training)
    accuracy=nltk.classify.accuracy(nltk_model,testing)*100
    print('{}: Accuracy: {}'.format(name,accuracy))
    
#ensemble method -- Voting Classifier for better accuracy
    
from sklearn.ensemble import VotingClassifier

names=['K Nearest Neighbors','Decision Tree','Random Forest','Logistic Regression','SGD Classifier','Naive Bayes','SVM Linear']
classifiers=[
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter=100),
    MultinomialNB(),
    SVC(kernel='linear')
     
    ]
    
models = zipper(names,classifiers)
# n_jobs=-1 means all algo can run in parallel
nltk_ensemble= SklearnClassifier(VotingClassifier(estimators=models,voting='hard',n_jobs= -1))
nltk_ensemble.train(training)
accuracy=nltk.classify.accuracy(nltk_ensemble,testing)*100
print('Ensemble Method Accuracy: {}'.format(accuracy))

#make class label predictions
txt_features,labels=zip(*testing)
prediction = nltk_ensemble.classify_many(txt_features)

#print a confusion matrix and a classification report
print(classification_report(labels,prediction))
pd.DataFrame(
    confusion_matrix(labels,prediction),
    index=[['actual','actual'],['ham','spam']],
    columns=[['predicted','predicted'],['ham','spam']]
    )
