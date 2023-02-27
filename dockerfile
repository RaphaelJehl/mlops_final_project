FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app.py .
COPY model.py .

EXPOSE 5000

CMD ["python", "-m", "app.py"]