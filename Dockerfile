FROM python:3.8.13-slim-buster
COPY . /app
WORKDIR /app
RUN apt update && apt install -y default-jdk
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -c "import nltk; nltk.download('punkt')"
EXPOSE 8000
CMD uvicorn app:app --host 0.0.0.0 --port 8000
