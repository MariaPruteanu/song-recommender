from credentials import *
import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import pandas as pd
import pickle

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=Client_ID,
                                                           client_secret=Client_secret))

songs_clusters = pd.read_csv('songs_clusters.csv')


def search_song(artist_name, track_title):
    search_query = f"artist:{artist_name} track:{track_title}"
    results = sp.search(q=search_query, type='track', limit=10)
    track_names_list = []
    artist_list = []
    album_name_list = []
    id_list = []
    if results['tracks']['items']:
        for track in results['tracks']['items']:
            track_name = track['name']
            artists = ", ".join([artist['name'] for artist in track['artists']])
            album_name = track['album']['name']
            track_uri = track['uri']
            track_id = track['id']
            track_href = track['href']
            track_names_list.append(track_name)
            artist_list.append(artists)
            album_name_list.append(album_name)
            id_list.append(track_id)
    else:
        print(f"No results found for '{track_title}' by '{artist_name}'.")
        raise Exception()
    results_df = pd.DataFrame({"Title": track_names_list, "Artist": artist_list, "Album": album_name_list, "ID": id_list})
    return results_df


def get_audio_features(list_of_songs_ids):

    '''
    This function gets a list that contains song ids as input,
    and returns a dataframe with all the audio features of each song id. 
    
    Inputs:
    list_of_songs_ids: list
    
    Output:
    DataFrame with the audio features of each song id.
    '''
    
    df = pd.DataFrame()
    
    for song_id in list_of_songs_ids:
        my_dict = sp.audio_features(song_id)[0]
        my_dict_new = { key : [my_dict[key]] for key in list(my_dict.keys()) }
        df = pd.concat([df, pd.DataFrame(my_dict_new)], axis=0)
    
    return df


def add_audio_features(df, audio_features_df):

    '''
    This function gets two dataframes as input,
    and returns a concatenated dataframe. 
    
    Inputs:
    df: pd.DataFrame
    audio_features_df: pd.DataFrame
    
    Output:
    DataFrame with both dataframes concatenated.
    '''

    df_features = pd.concat([df, audio_features_df], axis=1)
    return df_features


def load_scaler():

    '''
    This function loads the scaler that was fitted to the audio features of the song database, 
    so it can be used to scale the audio features of the user input song.
    '''

    with open('./scaler.pickle', 'rb') as file:
        return pickle.load(file)


def load_kmean():

    '''
    This function loads the kmeans model that was fitted to the audio features of the song database, 
    so it can be used to select the cluster of the user input song.
    '''

    with open('./kmeans_8.pickle', 'rb') as file:
        return pickle.load(file)


def hot_select_same_cluster(df: pd.DataFrame, user_song_cluster: int):

    '''
    This function takes a DataFrame and an integer as inputs and displays 
    5 songs at most that are hot and in the specified cluster 
    '''

    df['Spotify Link'] = "https://open.spotify.com/track/" + df['track_id']
    selected_rows = df[(df['dataset'] == 'Hot') & (df['kmeans'] == user_song_cluster)]
    if (len(selected_rows) <5):
        selected_rows = selected_rows.sample(len(selected_rows))
        display(selected_rows[["Song","Artist","Spotify Link"]])
    else:
        selected_rows = selected_rows.sample(5)
        display(selected_rows[["Song","Artist","Spotify Link"]].style.hide(axis="index"))


def not_hot_select_same_cluster(df: pd.DataFrame,user_song_cluster: int):

    '''
    This function takes a DataFrame and an integer as inputs and displays 
    5 songs at most that are not hot and in the specified cluster 
    '''

    df['Spotify Link'] = "https://open.spotify.com/track/" + df['track_id']
    selected_rows = df[(df['dataset'] == 'Not Hot') & (df['kmeans'] == user_song_cluster)]
    if (len(selected_rows) <5):
        selected_rows = selected_rows.sample(len(selected_rows))
        display(selected_rows[["Song","Artist","Spotify Link"]])
    else:
        selected_rows = selected_rows.sample(5)
        display(selected_rows[["Song","Artist","Spotify Link"]].style.hide(axis="index"))


def song_recommender():
    
    user_input = 'Yes'
    
    while user_input.lower()=='yes':
    
        user_input_song = input("Enter the song: ")
        user_input_artist = input("Enter the artist: ")
        
        try:
            search_song_df = search_song(user_input_artist, user_input_song)
        except:
            continue
        
        song_options = search_song_df.drop('ID',axis=1)
        
        print()
        print(song_options)
        print()
        
        user_select_song = float('inf')
        while (not(user_select_song >= 0 and user_select_song<len(song_options))):
            if user_select_song != float('inf'):
                print(f"{user_select_song} is not a valid number")
            user_select_song = int(input("Choose the song number: "))

        id_list = []

        song_id = search_song_df.iloc[user_select_song, search_song_df.columns.get_loc('ID')]

        id_list.append(song_id)

        user_audio_features = get_audio_features(id_list)

        song_row_df = search_song_df.iloc[user_select_song].to_frame().transpose().reset_index(drop=True)


        features_user_song = add_audio_features(song_row_df, user_audio_features)


        features_user_song.drop(['uri','track_href','type','id','duration_ms','time_signature','analysis_url'], axis=1, inplace=True)

        features_user_song.rename(columns={'ID':'track_id'},inplace=True)

        if (song_id in songs_clusters[songs_clusters["dataset"] == "Hot"]['track_id'].values.tolist()):
        #if songs_clusters.loc[songs_clusters['track_id'] == features_user_song['track_id'].iloc[0], 'dataset'].values == 'Hot':
            features_user_song['dataset'] = "Hot"
        else:
            features_user_song['dataset'] = "Not Hot"

        numerical = features_user_song.select_dtypes(include=np.number)

        scaler = load_scaler()
        kmeans_8 = load_kmean()

        user_song_audio_features_scaled_np = scaler.transform(numerical)
        user_song_audio_features_scaled_df = pd.DataFrame(user_song_audio_features_scaled_np, columns = numerical.columns)

        user_song_cluster = kmeans_8.predict(user_song_audio_features_scaled_df)[0]

        if features_user_song['dataset'].iloc[0] == 'Hot':
            hot_select_same_cluster(songs_clusters,user_song_cluster)
        else:
            not_hot_select_same_cluster(songs_clusters,user_song_cluster)
        
        print()
        
        user_input = ''
        while (user_input.lower() not in ['yes','no']):
            user_input = input('Do you want another recommendation? (yes/no):\n')
        
        print()