FROM python:3.10.13-slim

WORKDIR /app

COPY . . 

RUN python3 -m pip install --upgrade pip && python3 -m pip install -U setuptools

RUN python3 -m pip install --no-cache-dir -r requirements.txt

CMD uvicorn app:app