
FROM python:3.8-slim

EXPOSE 8000

WORKDIR /dashboard
COPY / /dashboard

RUN apt-get update 
RUN pip install --no-cache-dir -r requirements.txt
RUN export test=1

CMD gunicorn --workers 2 --bind 0.0.0.0:8000 app:server