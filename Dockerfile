FROM python:3.10.2-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY dags/main.py .

EXPOSE 5000

ENTRYPOINT [ "python", "dags/main.py" ]