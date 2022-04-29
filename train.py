import pandas as pd
import sklearn
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
import joblib
import string
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
lemmatizer = WordNetLemmatizer()
import fitz
import textract

def load_files(path, target):
 

    text = pd.DataFrame(columns = ["filename", "content", "label"], dtype=object)
    dir_list = os.listdir(path)
 
    print("Files and directories in '", path, "' :")
 
    # prints all files
    for x in dir_list:
        if x.endswith(".pdf"):
            # Prints only text file present in My Folder
            print(x)
          
            with fitz.open(path+'//'+x) as f:
                contents = ""
                for page in f:
                    contents += page.get_text()
                
            try:
                    filename = x
                    content = contents
                    label = target
                    text = text.append(pd.DataFrame([[filename, content, label]],
                                     columns=['filename', 'content', 'label'], dtype = object),
                                  ignore_index=True)            
            except:
                pass
        elif (x.endswith(".doc") or x.endswith(".docx")):
            print(x)
            contents = textract.process(path+'//'+x)  
            contents = contents.decode("utf-8")
            try:
                    filename = x
                    content = contents
                    label = target
                    text = text.append(pd.DataFrame([[filename, content, label]],
                                     columns=['filename', 'content', 'label'], dtype = object),
                                  ignore_index=True)            
            except:
                pass

    return text

path_selected = input("Enter path to Selected Resumes for e.g. C://users/AI/Selected: ")

path_rejected = input("Enter path to Rejected Resumes for e.g. C://users/AI/Rejected:  ")

train_df = load_files(path_selected, 1)

train_df = train_df.append(load_files(path_rejected, 0), ignore_index= True)

nltk.download("punkt")
nltk.download("stopwords")
nltk.download('wordnet')

def clean_sentences(df):
    reviews = []

    for sent in tqdm(df['content']):
        
        #remove html content
        review_text = BeautifulSoup(sent).get_text()
        
        #remove non-alphabetic characters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
    
        #tokenize the sentences
        words = word_tokenize(review_text.lower())
    
        #stop words removal
        omit_words = set(stopwords.words('english'))
        words = [x for x in words if x not in omit_words]
        
        #lemmatize each word to its lemma
        lemma_words = [lemmatizer.lemmatize(i) for i in words]
    
        reviews.append(lemma_words)

    return(reviews)

train_df_content = clean_sentences(train_df)

target=train_df.label.values

#target_test = test_df.label.values

# Set values for various parameters of word2vec
num_features = 400  # Word vector dimensionality. Determines the no of words each word in the vocabulary will
#be associated with. Must be tuned.-- 200
min_word_count = 10   # Minimum word count. Words occuring below the threshold will be ignored
num_workers = 1       # Number of threads to run in parallel
context = 10       # Context window size to be considered for each word    --5                                         
downsampling = 1e-3   # Downsample setting for frequent words. To prevent more frequent words from dominating.
    
from gensim.models import word2vec

    #Model Word2Vec
model_cbow = word2vec.Word2Vec(train_df_content, workers=num_workers, \
            vector_size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)


model_cbow.init_sims(replace=True)

def createFeatureVector(words, model, num_features):
    #initialize a 1D array with length as num of features of word2vec model chosen by us. 
    #Here it is 200.
    featVector = np.zeros((num_features,),dtype="float32")
    
    nWords = 0
    
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, as set is faster
    index2word_set = set(model.wv.index_to_key)
    
    # Loop over each word and add it to the feature vector to get the total sum of feature vectors of the
    #entire review
    for word in words:
        if word in index2word_set: 
            nWords = nWords + 1.
            featVector = np.add(featVector,model.wv[word])
            
    # Divide the result by the number of words to get the average of the feature vectors of 
    #all words in the review
    if(nWords != 0):
        featVector = np.divide(featVector,nWords)
    return featVector

#calculates the average of the feature vectors for each review using the word2vec values assigned for 
#each word
def avgFeatureVectors(sentences, model, num_features):
    overallFeatureVectors = []
    for sentence in tqdm(sentences):
        overallFeatureVectors.append(createFeatureVector(sentence, model, num_features)) 
    return overallFeatureVectors

train_df_vect = avgFeatureVectors( train_df_content, model_cbow, num_features )

target =target.astype('int')


X_train,X_val,y_train,y_val=train_test_split(train_df_vect,target,test_size=0.2)


#Model RF
n_estimators = [100, 200, 400, 600]
max_features = ['auto', 'sqrt']
max_depth = [2, 3, 5]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4, 10]


params_grid = {'n_estimators': n_estimators, 'max_features': max_features,
                   'max_depth': max_depth, 'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}


model_rf = RandomForestClassifier(random_state=42)

model_cv = GridSearchCV(model_rf, params_grid, scoring="accuracy", cv=3, verbose=1, n_jobs=-1)
model_cv.fit(X_train, y_train)
best_params = model_cv.best_params_
print(f"Best parameters: {best_params}")
model_rf = RandomForestClassifier(**best_params)
model_rf.fit(X_train, y_train)
    
pred = model_rf.predict(X_val)
clf_report = pd.DataFrame(classification_report(y_val, pred, output_dict=True))
print("Test Result:\n================================================")        
print(f"Accuracy Score: {accuracy_score(y_val, pred) * 100:.2f}%")
print("_______________________________________________")
print(f"CLASSIFICATION REPORT:\n{clf_report}")
print("_______________________________________________")
print(f"Confusion Matrix: \n {confusion_matrix(y_val, pred)}\n")


joblib.dump(model_rf, 'modelrf.pkl')
joblib.dump(model_cbow, 'vecmodelcbow.pkl')