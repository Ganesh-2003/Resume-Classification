import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Load the dataset
resumeDataSet = pd.read_csv('UpdatedResumeDataset.csv')

st.title("Resume Analyzer")
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose an option", ["Show Dataset", "Analyze Resumes", "Filter by Category"])

if app_mode == "Show Dataset":
    st.write("### Resume Dataset")
    st.dataframe(resumeDataSet)
elif app_mode == "Analyze Resumes":
    # Function to clean resume text
    def cleanResume(resumeText):
        resumeText = re.sub('http\S+\s*', ' ', resumeText)  # Remove URLs
        resumeText = re.sub('RT|cc', ' ', resumeText)  # Remove RT and cc
        resumeText = re.sub('#\S+', '', resumeText)  # Remove hashtags
        resumeText = re.sub('@\S+', '  ', resumeText)  # Remove mentions
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # Remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
        resumeText = re.sub('\s+', ' ', resumeText)  # Remove extra whitespace
        return resumeText

    # Clean and preprocess resumes
    resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

    # Display cleaned resume text
    st.write("### Cleaned Resume Text Example")
    st.write(resumeDataSet['cleaned_resume'][31])

    # Analyze the dataset and display the results
    st.write("### Resume Analysis Results")

    # Perform analysis on the dataset
    # For example, you can display the count of each category
    category_counts = resumeDataSet['Category'].value_counts()
    st.write("Category Counts:")
    st.write(category_counts)

    # Display Word Cloud
    st.write("### Word Cloud of Most Common Words")

    # Create a Word Cloud visualization of the most common words
    oneSetOfStopWords = set(stopwords.words('english') + ['``', "''"])
    totalWords = []
    Sentences = resumeDataSet['Resume'].values
    cleanedSentences = ""
    for i in range(0, 160):
        cleanedText = cleanResume(Sentences[i])
        cleanedSentences += cleanedText
        requiredWords = nltk.word_tokenize(cleanedText)
        for word in requiredWords:
            if word not in oneSetOfStopWords and word not in string.punctuation:
                totalWords.append(word)
    wordfreqdist = nltk.FreqDist(totalWords)
    mostcommon = wordfreqdist.most_common(50)
    wc = WordCloud().generate(cleanedSentences)

    # Display Word Cloud using a figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # Display countplot graph
    st.write("### Countplot of Resume Categories")
    plt.figure(figsize=(15, 15))
    plt.xticks(rotation=90)
    sns.countplot(y="Category", data=resumeDataSet)

    # Create a pie chart to show the distribution of categories
    st.write("### Distribution of Resume Categories")
    category_distribution = resumeDataSet['Category'].value_counts()
    labels = category_distribution.index
    sizes = category_distribution.values

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart using st.pyplot()
    st.pyplot(fig)

    
    # Display countplot graph using a figure
    fig, ax = plt.subplots()
    ax = sns.countplot(y="Category", data=resumeDataSet)
    st.pyplot(fig)

    # Train models and display results
    st.write("### Model Results")

    # Encode the 'Category' column to numerical values
    var_mod = ['Category']
    le = LabelEncoder()
    for i in var_mod:
        resumeDataSet[i] = le.fit_transform(resumeDataSet[i])

    # Split the data into training and testing sets
    requiredText = resumeDataSet['cleaned_resume'].values
    requiredTarget = resumeDataSet['Category'].values
    word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=1500)
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)
    X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)

    # Train a Multinomial Naive Bayes classifier
    clf_nb = OneVsRestClassifier(MultinomialNB()).fit(X_train, y_train)
    
    # Train a K-Nearest Neighbors classifier
    clf_knn = OneVsRestClassifier(KNeighborsClassifier()).fit(X_train, y_train)

    # Model results
    st.write("### Multinomial Naive Bayes Classifier Results")
    st.write(f"Accuracy on training set: {clf_nb.score(X_train, y_train):.2f}")
    st.write(f"Accuracy on test set: {clf_nb.score(X_test, y_test):.2f}")

    st.write("### K-Nearest Neighbors Classifier Results")
    st.write(f"Accuracy on training set: {clf_knn.score(X_train, y_train):.2f}")
    st.write(f"Accuracy on test set: {clf_knn.score(X_test, y_test):.2f}")

elif app_mode == "Filter by Category":
    st.write("### Filter Resumes by Category")
    category_filter = st.text_input("Enter a category to filter resumes:")
    filtered_resumes = resumeDataSet[resumeDataSet['Category'] == category_filter]
    
    if not filtered_resumes.empty:
        st.write("### Resumes in the Category: " + category_filter)
        st.dataframe(filtered_resumes)
    else:
        st.write("No resumes found in the specified category.")
