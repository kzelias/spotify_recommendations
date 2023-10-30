import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import faiss
import joblib
import gdown


def download_data(table_name):
    # загрузка данных
    print("__Data loading...")
    url = "https://drive.google.com/uc?id=1UgDwPShtfqq9QAk5GI5IX-2hhyErVbze"
    gdown.download(url, table_name)


def dict_create(df_full):
    print("__Dict creating...")
    print(f"Shape: {df_full.shape}")
    print(f"nunique: {df_full.nunique()}")
    df_tracks = df_full.drop(columns=["playlist_id"])
    df_unique = df_tracks.drop_duplicates(["track_id"])
    df_unique = df_unique.set_index("track_id")
    unique = df_unique.to_dict()
    id_to_track = unique["track_uri"]
    track_to_id = dict((v, k) for k, v in id_to_track.items())
    joblib.dump(id_to_track, "binary/id_to_track.pickle")
    joblib.dump(track_to_id, "binary/track_to_id.pickle")
    print("dicts saved to binary/")


def svd_create(df_full, dimension):
    # SVD
    print("__Matrix creating...")
    df = df_full.drop(columns=["track_uri"])
    n_track = df["track_id"].unique().shape[0]
    n_playlist = df["playlist_id"].unique().shape[0]
    print(f"Tracks: {n_track}, playlists: {n_playlist}")
    # создаём tracks-playlists матрицу
    ratio = np.zeros((n_track, n_playlist))
    for line in df.itertuples():
        ratio[line[2], line[1]] = 1

    # вычисление svd
    print("__SVD running...")
    u, s, vt = svds(ratio, k=dimension)
    # размерности выходной матрицы
    print(f"Shape: {u.shape}")
    return u


def index_create(u, dimension):
    # faiss
    print("__Index creating...")
    index = faiss.IndexFlat(dimension)
    index.add(u)
    print(index.ntotal)

    # проверка что для любого вектора ближайший он сам
    for vec_id in range(5):
        vec_0 = index.reconstruct_batch(vec_id)
        D, I = index.search(vec_0, k=1)
        assert I[0] == vec_id, "created index is broken"

    # запись
    faiss.write_index(index, "binary/index_spotify_d100.bin")
    print("index saved to binary/")


def main():
    table_name = "rec_test_assignment_playlist2track.csv"
    download_data(table_name)

    df_full = pd.read_csv(table_name)

    dict_create(df_full)

    dimension = 100
    u = svd_create(df_full, dimension)

    index_create(u, dimension)

    print("DONE")


if __name__ == "__main__":
    main()
