import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Text cleaning functions
def clean_text_basic(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def clean_text_advanced(txt):
    txt = clean_text_basic(txt)
    words = txt.split()
    lemmatizer = WordNetLemmatizer()
    custom_stopwords = set(stopwords.words('english')).union({'resume', 'cv', 'company', 'organization'})
    words = [lemmatizer.lemmatize(word) for word in words if word not in custom_stopwords]
    return ' '.join(words)

# Clean resumes
df['Resume'] = df['Resume'].apply(lambda x: clean_text_advanced(x))

# Encode categories
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])

# Vectorize resumes
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])
required_text = tfidf.transform(df['Resume'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(required_text, df['Category'], test_size=0.2, random_state=42)

# Train the classifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Train the LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(required_text)

# Save the models and vectorizer
joblib.dump(clf, 'clf.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(lda, 'lda.pkl')
