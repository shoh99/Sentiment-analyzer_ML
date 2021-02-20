# import required libraries

from PIL import Image
import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import tree

import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# from sklearn.preprocessing import label encoder
matplotlib.use('Agg')


# disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# set title
st.title('DevOps Team presents')

# import image
image = Image.open('photos/cover.png')
st.image(image, use_column_width=True)


# main function
def main():

    main_activities = ['Machine learning', 'Deep learning']
    main_option = st.sidebar.selectbox('Select learning type', main_activities)

    #  machine learning
    if main_option == 'Machine learning':
        st.subheader('Welcome to machine learning section')

        ml_activities = ['EDA', 'Visualisation', 'model', 'About us']
        ml_option = st.sidebar.selectbox(
            'ML Selection option: ', ml_activities)

        # EDA
        if ml_option == 'EDA':
            st.subheader('Exploratory Data Analysis')

            data = st.file_uploader('Upload dataset: ', type=[
                                    'csv', 'xlsx', 'txt', 'json'])

            if data is not None:
                st.success('Your dataset is loaded')
                df = pd.read_csv(data)
                st.dataframe(df.head(50))

                if st.checkbox('Display Shape'):
                    st.write(df.shape)

                if st.checkbox('Display columns'):
                    st.write(df.columns)

                if st.checkbox('SElect multiple columns'):
                    selected_columns = st.multiselect(
                        'Select preferred columns: ', df.columns)
                    df1 = df[selected_columns]
                    st.dataframe(df1)

                if st.checkbox('Display summmary'):
                    st.write(df.describe().T)

                if st.checkbox('Display Null Value'):
                    st.write(df.isnull().sum())

                if st.checkbox('Display data type:'):
                    st.write(df.dtypes)

                if st.checkbox('Display Correlation of dataframe'):
                    st.write(df.corr())
        # visualization
        elif ml_option == 'Visualisation':
            st.subheader('Visualization of Data')

            data = st.file_uploader('Upload dataset: ', type=[
                                    'csv', 'xlsx', 'txt', 'json'])

            if data is not None:
                st.success('Your dataset is loaded')
                df = pd.read_csv(data)
                st.dataframe(df.head(50))

                if st.checkbox('Select Multiple columns to plot'):
                    selected_columns = st.multiselect(
                        'Select yourt preferred columns', df.columns)
                    df1 = df[selected_columns]
                    st.dataframe(df1)

                if st.checkbox('Display heatmap'):
                    st.write(sns.heatmap(df1.corr(), vmax=1, square=True,
                                         annot=True, linecolor='red', cmap='viridis'))
                    st.pyplot()

                if st.checkbox('Display Pairplot'):
                    st.write(sns.pairplot(df1, diag_kind='kde'))
                    st.pyplot()

                if st.checkbox('Display pie chart'):
                    all_columns = df.columns.to_list()
                    pie_columns = st.selectbox(
                        'Select columns to display', all_columns)
                    pieChart = df[pie_columns].value_counts().plot.pie(
                        autopct='%1.1f%%')
                    st.write(pieChart)
                    st.pyplot()

                if st.checkbox('Scatter Plot'):

                    all_columns = df.columns.to_list()
                    x_column = st.selectbox(
                        'Selct x from  dataset', all_columns)
                    y_column = st.selectbox(
                        'Select y from dataset', all_columns)
                    hue = st.selectbox('Select hue from dataset', all_columns)

                    scatter_vis = sns.relplot(data=df, x=x_column,
                                              y=y_column, hue=hue)
                    st.pyplot()

        # model part
        elif ml_option == 'model':
            st.subheader('Model Building')
            data = st.file_uploader('Upload dataset: ', type=[
                                    'csv', 'xlsx', 'txt', 'json'])

            if data is not None:
                st.success('Your dataset is loaded')
                df = pd.read_csv(data)
                st.dataframe(df.head(50))

                if st.checkbox('Select multiple columns'):
                    new_data = st.multiselect(
                        'Select your prefered columns,  ', df.columns)
                    st.warning('Let your target variable to be last column.')
                    df1 = df[new_data]
                    st.dataframe(df1)
                    # dividing my data into x ans y variables
                    X = df1.iloc[:, 0: -1]
                    y = df1.iloc[:, -1]
                else:
                    st.error('Please choose your columns')

                seed = st.sidebar.slider('Seed', 1, 200)

                classifier_name = st.sidebar.selectbox(
                    'Select your pereferred classifier: ', ('KNN', 'SVM', 'LR', 'Naive_Baise'))

                # choosing parametrs
                # @st.cache(suppress_st_warning=True)
                def add_parameter(name_of_clf):
                    param = dict()

                    if name_of_clf == 'SVM':
                        C = st.sidebar.slider('C', 1, 15)
                        param['C'] = C

                    if name_of_clf == 'KNN':
                        K = st.sidebar.slider('K', 1, 15)
                        param['K'] = K

                    return param

                # calling the function

                params = add_parameter(classifier_name)

                # define function to clasifier
                # @st.cache(suppress_st_warning=True)
                def get_classifier(name_of_clf, params):
                    clf = None
                    if name_of_clf == 'SVM':
                        clf = SVC(C=params['C'])
                    elif name_of_clf == 'KNN':
                        clf = KNeighborsClassifier(n_neighbors=params['K'])

                    elif name_of_clf == 'LR':
                        clf = LogisticRegression()
                    elif name_of_clf == 'Naive_Baise':
                        clf = GaussianNB()
                    else:
                        st.warning('Select your choice of algorithm: ')

                    return clf

                clf = get_classifier(classifier_name, params)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed)

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                st.write('Prediction: ', y_pred)

                accuracy = accuracy_score(y_test, y_pred)
                st.write('Name of classifier:', classifier_name)
                st.write('Accuracy score: ', accuracy)

        elif ml_option == 'About us':
            st.subheader(
                'In About us page you can learn about different types of ML algorithms')

            if st.checkbox('About KNN algorithm'):
                st.write('K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions). KNN has been used in statistical estimation and pattern recognition already in the beginning of 1970’s as a non-parametric technique.')

                st.write('** Algorithm **')

                st.write('A case is classified by a majority vote of its neighbors, with the case being assigned to the class most common amongst its K nearest neighbors measured by a distance function. If K = 1, then the case is simply assigned to the class of its nearest neighbor. ')

                # import image
                img = Image.open('photos/KNN_similarity.png')
                st.image(img)

            if st.checkbox('About Support vector classifier'):
                st.write('The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.')

                st.write('** Hyperplanes and Support Vectors **')

                st.write('Hyperplanes are decision boundaries that help classify the data points. Data points falling on either side of the hyperplane can be attributed to different classes. Also, the dimension of the hyperplane depends upon the number of features. If the number of input features is 2, then the hyperplane is just a line. If the number of input features is 3, then the hyperplane becomes a two-dimensional plane. It becomes difficult to imagine when the number of features exceeds 3.')

                # import image
                img1 = Image.open('photos/svc.png')
                st.image(img1, use_column_width=True)

                st.write('** So what makes it so great? **')
                st.write("Well SVM it capable of doing both classification and regression. In this post I'll focus on using SVM for classification. In particular I'll be focusing on non-linear SVM, or SVM using a non-linear kernel. Non-linear SVM means that the boundary that the algorithm calculates doesn't have to be a straight line. The benefit is that you can capture much more complex relationships between your datapoints without having to perform difficult transformations on your own. The downside is that the training time is much longer as it's much more computationally intensive.")

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    elif main_option == 'Deep learning':
        st.subheader('Welcome to Deep learning')

        dl_activities = ['Sentiment analysis']
        dl_option = st.sidebar.selectbox('DL Selection option', dl_activities)

        if dl_option == 'Sentiment analysis':

            def predict(message):
                model = load_model('data/movie_sent.h5')
                with open('data/tokenize (1).pickle', 'rb') as handle:
                    tokenizer = pickle.load(handle)
                    x_1 = tokenizer.texts_to_sequences([message])
                    x_1 = pad_sequences(x_1, maxlen=500)
                    predictions = model.predict(x_1)[0][0]
                    return predictions

            st.title('Movie Review Sentiment Analyzer')
            message = st.text_area('Enter Review, Type Here ..')

            if st.button('Analyze'):
                with st.spinner('Analyzing the text …'):
                    prediction = predict(message)
                    if prediction > 0.6:
                        st.success(
                            'Positive review with {:.2f} confidence'.format(prediction))
                        st.balloons()
                    elif prediction < 0.4:
                        st.error(
                            'Negative review with {:.2f} confidence'.format(1-prediction))
                    else:
                        st.warning('Not sure! Try to add some more words')


if __name__ == "__main__":
    main()
