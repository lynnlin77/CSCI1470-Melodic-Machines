import utils

def create_metadata(tracks_path, subset='small', columns = [('track', 'title'), ('track', 'genre_top'), ('artist', 'name')]):
    tracks = utils.load(tracks_path)
    subset = tracks[tracks['set', 'subset'] <= subset]
    result = subset[columns]
    result.columns = ['_'.join(tuple_c) for tuple_c in columns]
    result.to_csv('metadata.csv', index=True)

if __name__ == '__main__':
    tracks_path = 'tracks.csv'  # Adjust the path as necessary
    create_metadata(tracks_path)