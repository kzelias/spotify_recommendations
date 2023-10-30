## Сервис музыкальных рекомендаций.  


### Быстрый запуск:
Запустить команду в основной директории репозитория:  
- `docker compose -f docker/compose.fast.yaml up`  
(при сборке заранее созданный index и словари ids-uris скачаются с google drive)  
  
### Запуск с созданием index вручную:  
##### Настройка окружения
- `python3 -m venv .venv`  
- `source .venv/bin/activate`  
- `pip install -r requirements_for_index.txt`  
##### Создание индекса и сборка  
- `python3 create_index.py`, либо через `index_creation.ipynb`  
(данные скачаются с google drive. index сохранится с именем index_spotify_d100.bin, также сохраняются словари ids-uris)  

Запустить команду в основной директории репозитория:  
- `docker compose -f docker/compose.yaml up`  
  
### Проверка работоспособности
1. По ссылке:  
http://127.0.0.1:8000/recommend?id=spotify:track:3SuzCeGoNOWmbWOoDMou0B&k=3
2. Через OpenAPI docs:  
http://127.0.0.1:8000/docs#/  
Пример `spotify:track:1r0faljjM2b876iNoaDUh5`

### Примечания:  
1. Для построения векторных представлений выбран SVD, его конкретная реализация scipy.sparse.svds для разряженных матриц.  
2. Для построения ann индекса и поиска по нему выбрана библиотека faiss, исходя из последних коммитов, она больше поддерживается и чаще обновляется, чем аналоги в виде annoy, nmslib. По бенчмаркам nmslib быстре faiss, но faiss имеет возможность запускаться на GPU.  
