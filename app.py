import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import pickle
from sklearn.preprocessing import OneHotEncoder


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# home andinput page

# predict route
@app.route('/', methods=['GET', 'POST'])
def predict():

    if request.method=='GET':
        return render_template('index.html')

    elif request.method=='POST':

        df = pd.read_csv(r"C:\Users\raphj\OneDrive\Documents\EFREI_S9\mlops\Anime_data.csv")
        
        # preprocess as in the model

        df = df.drop(['Title'], axis=1)
        df = df.drop(['Synopsis'], axis=1)
        df = df.drop(['Link'], axis=1)
        df = df.drop(['Aired'], axis=1)
        df = df.drop(['Rating'], axis=1)

        mean_scoredby = df['ScoredBy'].mean()
        mean_popularity = df['Popularity'].mean()
        mean_members = df['Members'].mean()
        mean_episodes = df['Episodes'].mean()

        df['ScoredBy'].fillna(mean_scoredby, inplace=True)
        df['Popularity'].fillna(mean_popularity, inplace=True)
        df['Members'].fillna(mean_members, inplace=True)
        df['Episodes'].fillna(mean_episodes, inplace=True)

        most_frequent_type = df['Type'].value_counts().idxmax()
        most_frequent_source = df['Source'].value_counts().idxmax()
        most_frequent_studio = df['Studio'].value_counts().idxmax()

        df['Type'].fillna(most_frequent_type, inplace=True)
        df['Source'].fillna(most_frequent_source, inplace=True)
        df['Studio'].fillna(most_frequent_studio, inplace=True)


        # input from form
        Genre = request.form['genre']
        Type = request.form['type']
        Producer = request.form['producer']
        Studio = request.form['studio']
        Popularity = request.form['popularity']
        Members = request.form['members']
        Episodes = request.form['episodes']
        Source = request.form['source']

        # store input in df
        df.loc[0] = ({'Genre': str(Genre), 'Type': str(Type), 'Producer': str(Producer), 'Studio': str(Studio), 'Popularity': int(Popularity), 'Members': int(Members), 'Episodes': int(Episodes), 'Source': str(Source)})

        # removing unauthorized char/symbols
        df['Genre'] = df['Genre'].str.replace('[', '')
        df['Genre'] = df['Genre'].str.replace(']', '')
        df['Genre'] = df['Genre'].str.replace("'", '')

        df['Producer'] = df['Producer'].str.replace('[', '')
        df['Producer'] = df['Producer'].str.replace(']', '')
        df['Producer'] = df['Producer'].str.replace("'", '')

        df['Studio'] = df['Studio'].str.replace('[', '')
        df['Studio'] = df['Studio'].str.replace(']', '')
        df['Studio'] = df['Studio'].str.replace("'", '')

        # one hot encode and add dummies
        encoder = OneHotEncoder()

        # type
        type_column = df['Type'].values
        encoded_type = encoder.fit_transform(type_column.reshape(-1,1))
        encoded_type_column = encoder.get_feature_names_out([['Type']])
        encoded_type = encoded_type.toarray()
        encoded_df = pd.DataFrame(encoded_type, columns=encoded_type_column)

        df = pd.concat([df, encoded_df], axis=1)
        df.drop("Type", axis=1, inplace=True)

        # studio
        dummies_studio = df["Studio"].str.get_dummies(', ').add_prefix('Studio_')
        df = pd.concat([df, dummies_studio], axis=1)
        df = df.drop('Studio', axis=1)

        # source
        source_column = df['Source'].values
        encoded_source = encoder.fit_transform(source_column.reshape(-1,1))
        encoded_source_column = encoder.get_feature_names_out([['Source']])
        encoded_source = encoded_source.toarray()
        encoded_df = pd.DataFrame(encoded_source, columns=encoded_source_column)

        df = pd.concat([df, encoded_df], axis=1)
        df.drop("Source", axis=1, inplace=True)

        # genre
        dummies_genre = df["Genre"].str.get_dummies(', ').add_prefix('Genre_')
        df = pd.concat([df, dummies_genre], axis=1)
        df = df.drop('Genre', axis=1)

        # producer
        dummies_producer = df["Producer"].str.get_dummies(', ').add_prefix('Producer_')
        df = pd.concat([df, dummies_producer], axis=1)
        df = df.drop('Producer', axis=1)

        prediction = model.predict(df.iloc[[0]])

        return render_template('result.html', genre=Genre, type=Type, producer=Producer, studio=Studio, popularity=Popularity, members=Members, episodes=Episodes, source=Source, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)