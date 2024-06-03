FROM python:3.10.2-slim-buster

RUN apt-get update && \
apt-get install openjdk-11-jdk scala maven unzip gcc python3-dev -y

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY dags/ dags/
COPY model/ model/

EXPOSE 8000

ENTRYPOINT [ "python", "dags/main.py" ]