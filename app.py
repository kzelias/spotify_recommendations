from fastapi import FastAPI, HTTPException
import faiss
import joblib
import logging

# загрузка индекса и словарей ids-tracks
INDEX = faiss.read_index("binary/index_spotify_d100.bin")
ID_TO_TRACK = joblib.load("binary/id_to_track.pickle")
TRACK_TO_ID = joblib.load("binary/track_to_id.pickle")

logger = logging.getLogger("uvicorn.error")
app = FastAPI()


@app.get("/health")
def read_root():
    return "alive"


@app.get("/recommend")
def get_recommend(id: str, k: int = 5):
    logger.info("________________NEW__REQUEST________________")

    # проверка есть ли данный track uri в базе
    if id in TRACK_TO_ID:
        track_id = TRACK_TO_ID[id]
    else:
        logger.info("There is no track uri")
        raise HTTPException(status_code=400, detail="There is no track uri")

    # поиск похожих векторов
    vec = INDEX.reconstruct_batch(track_id)
    indices = INDEX.search(vec, k + 1)[1]

    logger.info(f"track_uri: {id}, track_id: {track_id}")
    logger.info(f"Similar tracks indices: {indices[0][1:]}")

    # exception, когда поиск вернул вместо индекса -1 - не нашел рекомендаций
    if indices[0][1] == -1:
        raise HTTPException(status_code=400, detail="There are no recommendations")

    # компоновка результата
    result = []
    for inx in indices[0][1:]:  # without first
        if inx == -1:
            # добавляет строку вместо uri, когда 
            result.append("No track to recommend")
        else:
            result.append(ID_TO_TRACK[inx])

    return result
