
# https://carpentry.library.ucsb.edu/2021-08-23-ucsb-python-online/09-working-with-sql/index.html

import csv
import sqlite3
import pandas as pd
from os import listdir
from os.path import isfile, join
import lyricsgenius

def retrieve_artist_trackID(path):
    con = sqlite3.connect(path)
    df = pd.read_sql_query('SELECT artist_name,track_id FROM songs', con)
    df['artist_name'] = df['artist_name'].str.lower().str.strip() #turn all artist names to lowercase and get rid of punctuation
    df.to_csv('MSD_artists_trackIDS.csv', index=False)

def retrieve_unique_artists(path):
    con = sqlite3.connect(path)
    df = pd.read_sql_query('SELECT artist_name FROM songs', con)
    df.drop_duplicates(inplace= True)
    df['artist_name'] = df['artist_name'].str.lower().str.strip() 
    df.to_csv('MSD_artists_unique.csv', index=False)

def retrieve_artist_lyrics(path):
    genius = lyricsgenius.Genius("2ZvGLdJrJsZyhm1EU_35tcCHz89eyp4TfBRyybo9JIqUDTR71nH6jXs9DtQdB-D2")
    artist_lyrics = pd.DataFrame(columns=['artist','lyric'])
    no_lyrics = pd.DataFrame(columns=['artist', 'song'])

    df = pd.read_csv(path, index_col=0)
    num_rows = len(df)
    for row in range(num_rows):
        row = df.iloc[row].str.lower().str.strip() 
        track = row['track title']
        artist = row['artist name']
        
        song = genius.search_song(track, artist) #Song, Artist Pair
        if(song == None):
            new_row = pd.DataFrame([{'artist':artist, 'song':track}])
            no_lyrics = pd.concat([no_lyrics, new_row], ignore_index=True)

        else:
            new_row = pd.DataFrame([{'artist':artist, 'lyric': song.lyrics}])
            artist_lyrics = pd.concat([artist_lyrics, new_row], ignore_index=True)

    artist_lyrics['lyric'] = artist_lyrics['lyric'].str.lower().str.strip() 
    artist_lyrics.to_csv('artist_lyrics.csv', index=False)
    no_lyrics.to_csv('artist_no_lyrics.csv', index=False)
    print("there are ", len(no_lyrics), " songs that don't have lyrics")
    print("there are ", len(artist_lyrics), " songs that have lyrics")
 

def artist_intersection(msd, fma):
    msd_df = pd.read_csv(msd)
    msd_df.rename(columns={'artist_name': 'name'}, inplace=True) #rename column to merge
    fma_df = pd.read_csv(fma)
    df_merged = pd.concat([msd_df, fma_df], ignore_index=True)
    duplicates = df_merged[df_merged.duplicated()]
    duplicates.to_csv('MSD_FMA_artists.csv', index=False)

def merge_famous_artists(path):
    #https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    mainframe = pd.DataFrame(columns=['artist','lyric'])
    for file in onlyfiles:
        file_path = path + "/" + file
        df = pd.read_csv(file_path, index_col=0)
        df.columns = df.columns.str.lower()
        df.drop(columns=['title','album','year','date'], inplace=True)
        df['artist'] = df['artist'].str.lower().str.strip() 
        df['lyric'] = df['lyric'].str.lower().str.strip()
        mainframe = pd.concat([mainframe, df], ignore_index= True)
    mainframe.to_csv('famous_artist_lyrics.csv', index=False)

def create_test_train(path):
    train = pd.DataFrame(columns=['artist','lyric'])
    test = pd.DataFrame(columns=['artist','lyric'])
    df = pd.read_csv(path)
    unique = df['artist'].unique()
    for artist in unique:
        artist_row = df[df['artist'] == artist]
        train = pd.concat([train, artist_row.head(15)], ignore_index= True)
        test = pd.concat([test, artist_row.tail(5)])

    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index= False)
            
if __name__ == "__main__":
    #path variables
    PATH_METADATA = './track_metadata.db'
    PATH_FMA = '../FMA/FMA_artists.csv'
    PATH_MSD = './MSD_artists_unique.csv'
    PATH_FMA_METADATA = '../FMA/tracks_medium.csv'
    PATH_FAMOUS_LYRICS = './famous_artist_lyrics.csv'

    # retrieve_unique_artists(PATH_METADATA)
    # retrieve_artist_trackID(PATH_METADATA)
    # artist_intersection(PATH_MSD, PATH_FMA)
    # merge_famous_artists(PATH_FAMOUS_LYRICS)
    create_test_train(PATH_FAMOUS_LYRICS)