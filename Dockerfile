FROM python:3.8.13-slim-buster
COPY . /app
WORKDIR /app
RUN apt update && apt install -y default-jdk
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -c "import nltk; nltk.download('punkt')"
ARG API_PORT=8000
ENV ENV_API_PORT=$API_PORT
EXPOSE $ENV_API_PORT
CMD uvicorn app:app --host 0.0.0.0 --port $ENV_API_PORT
