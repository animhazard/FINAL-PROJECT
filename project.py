
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function for text preprocessing
def preprocess_text(text):
    # Step 1: Convert to lowercase
    text = text.lower()

    # Step 2: Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Step 3: Remove special characters, numbers, and punctuations
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Step 4: Tokenization (breaking the text into words)
    tokens = word_tokenize(text)

    # Step 5: Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Step 6: Lemmatization (reduce words to their base form)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string
    return ' '.join(tokens)

# Load the Dataset
df = pd.read_csv(r'C:\Priya Gurung\Final Project\sentimentdataset.csv', encoding='utf-8')

# Handle Missing Values
df.dropna(subset=['Text'], inplace=True)

# Preview the dataset
print("First few rows of the dataset:\n", df.head())

# Apply the preprocessing function to the 'Text' column
df['Cleaned_Text'] = df['Text'].apply(preprocess_text)

# Display the first few rows to verify
print("\nFirst few rows after cleaning:\n", df[['Text', 'Cleaned_Text']].head())

# Function to perform sentiment analysis
def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    # Classify sentiment based on the compound score
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis to cleaned text
df['Sentiment'] = df['Cleaned_Text'].apply(analyze_sentiment)

# EDA: Check the distribution of sentiments 
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Sentiment', palette='viridis', hue='Sentiment', legend=False)
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


from wordcloud import WordCloud

def plot_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Generate and plot word clouds for different sentiments
plot_wordcloud(' '.join(df[df['Sentiment'] == 'Positive']['Cleaned_Text']))
plot_wordcloud(' '.join(df[df['Sentiment'] == 'Negative']['Cleaned_Text']))
plot_wordcloud(' '.join(df[df['Sentiment'] == 'Neutral']['Cleaned_Text']))

from collections import Counter

def plot_most_common_words(sentiment):
    words = ' '.join(df[df['Sentiment'] == sentiment]['Cleaned_Text']).split()
    most_common_words = Counter(words).most_common(10)
    plt.figure(figsize=(10, 6))
    plt.bar(*zip(*most_common_words))
    plt.title(f'Top 10 Words in {sentiment} Sentiment')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

# Plot for each sentiment
for sentiment in df['Sentiment'].unique():
    plot_most_common_words(sentiment)


# Save the preprocessed data into a new CSV file
df.to_csv(r'C:\Priya Gurung\Final Project\preprocessed_social_media_data.csv', index=False)

# Print summary statistics for sentiments
sentiment_counts = df['Sentiment'].value_counts()
print("\nSentiment Counts:\n", sentiment_counts)

# ---- Feature Engineering ----

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Step 1: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Cleaned_Text'])

# Convert to DataFrame for easier analysis (optional)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
df = pd.concat([df, tfidf_df], axis=1)  # Append TF-IDF features to the original DataFrame

# Step 2: Word2Vec
tokenized_texts = [text.split() for text in df['Cleaned_Text']]
word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

# Function to create sentence vectors by averaging word vectors
def get_sentence_vector(text):
    words = text.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if not word_vectors:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word_vectors, axis=0)

# Apply the function to create sentence vectors
df['Word2Vec_Vector'] = df['Cleaned_Text'].apply(get_sentence_vector)

# Step 3: BERT Embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Apply the function to create BERT embeddings
df['BERT_Vector'] = df['Cleaned_Text'].apply(get_bert_embedding)

# Step 4: Add Sentiment Scores
def get_sentiment_scores(text):
    score = analyzer.polarity_scores(text)
    return score['compound']

df['Sentiment_Score'] = df['Cleaned_Text'].apply(get_sentiment_scores)

# Save the feature engineered data into a new CSV file
df.to_csv(r'C:\Priya Gurung\Final Project\feature_engineered_data.csv', index=False)

print("\nFeature Engineering Completed.")

# ---- Model Selection and Training ----

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

# Choose features (TF-IDF matrix for this example)
X = tfidf_matrix  # or use df['BERT_Vector'] if you prefer BERT embeddings
y = df['Sentiment']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
    print(f"{model_name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Evaluate Logistic Regression
evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, X_test, y_test, "Random Forest")

# SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
evaluate_model(svm_model, X_test, y_test, "SVM")

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [None, 10, 20, 30],
}

rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)
print("Best parameters for Random Forest:", rf_grid_search.best_params_)

# Hyperparameter tuning for SVM
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}

svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=5, scoring='accuracy')
svm_grid_search.fit(X_train, y_train)
print("Best parameters for SVM:", svm_grid_search.best_params_)

# Create a Voting Classifier
voting_classifier = VotingClassifier(
    estimators=[
        ('lr', lr_model), 
        ('rf', rf_grid_search.best_estimator_), 
        ('svm', svm_grid_search.best_estimator_)
    ], 
    voting='hard'
)
voting_classifier.fit(X_train, y_train)

# Evaluate Voting Classifier
evaluate_model(voting_classifier, X_test, y_test, "Voting Classifier")
