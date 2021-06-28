import pandas as pd
import pickle

# loading data
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',names=["label", "message"])

#Data cleaning and preprocessing
import re
import nltk

nltk.download('stopwords')
nltk.download('corpus')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating the TFIDf model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=2500)
X = tfidf.fit_transform(corpus).toarray()

# to handle categorical labels
y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values # we can represent spam and ham by one columns, so we removing the ham column

# Creating a pickle file for the TfidfVectorizer
pickle.dump(tfidf, open('tfidf-transform.pkl', 'wb'))

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'spam-detector-model.pkl'
pickle.dump(spam_detect_model, open(filename, 'wb'))