# imports

import pandas as pd
import pickle


df = pd.read_csv(r"C:\Users\raphj\OneDrive\Documents\EFREI_S9\mlops\Anime_data.csv")

# keep anime_id, genre, type, producer, studio, scoredby, popularity, members, episodes, source

df = df.drop(['Title'], axis=1)
df = df.drop(['Synopsis'], axis=1)
df = df.drop(['Link'], axis=1)
df = df.drop(['Aired'], axis=1)

# replacing nans by mean in rating, scoredby, popularity, members, episodes

mean_rating = df['Rating'].mean()
mean_scoredby = df['ScoredBy'].mean()
mean_popularity = df['Popularity'].mean()
mean_members = df['Members'].mean()
mean_episodes = df['Episodes'].mean()

df['Rating'].fillna(mean_rating, inplace=True)
df['ScoredBy'].fillna(mean_scoredby, inplace=True)
df['Popularity'].fillna(mean_popularity, inplace=True)
df['Members'].fillna(mean_members, inplace=True)
df['Episodes'].fillna(mean_episodes, inplace=True)

# replacing nans by most frequent values in type, studio, source

most_frequent_type = df['Type'].value_counts().idxmax()
most_frequent_source = df['Source'].value_counts().idxmax()
most_frequent_studio = df['Studio'].value_counts().idxmax()

df['Type'].fillna(most_frequent_type, inplace=True)
df['Source'].fillna(most_frequent_source, inplace=True)
df['Studio'].fillna(most_frequent_studio, inplace=True)

# removing unauthorized char/symbols

df['Genre'] = df['Genre'].str.replace('[', '', regex=False)
df['Genre'] = df['Genre'].str.replace(']', '', regex=False)
df['Genre'] = df['Genre'].str.replace("'", '', regex=False)

df['Producer'] = df['Producer'].str.replace('[', '', regex=False)
df['Producer'] = df['Producer'].str.replace(']', '', regex=False)
df['Producer'] = df['Producer'].str.replace("'", '', regex=False)

df['Studio'] = df['Studio'].str.replace('[', '', regex=False)
df['Studio'] = df['Studio'].str.replace(']', '', regex=False)
df['Studio'] = df['Studio'].str.replace("'", '', regex=False)

# one hot encode and add dummies

from sklearn.preprocessing import OneHotEncoder

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

# split dataset

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)

X_train = train.drop('Rating', axis=1)
y_train = train['Rating']

X_test = test.drop('Rating', axis=1)
y_test = test['Rating']

# create classifier and train model
import xgboost as xgb

model = xgb.XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# save model with pickle
pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl', 'rb'))