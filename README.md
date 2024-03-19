# 📰 fake news detection

automate news authenticity prediction with machine learning model

## initialization

1. install [python 3.8](https://www.python.org/downloads/)

2. install [java 8](https://www.oracle.com/java/technologies/downloads/)

3. run below to grant helper script execution

```sh
chmod +x run.sh
```

4. run the following to initialize project
```sh
./run.sh init
```

each of the following steps uses a YAML configuration file stored in `config` folder

## data cleaning

processed version of dataset is saved to `data` directory with `prep` suffix by default

```sh
./run.sh clean
```


## data visualization

all visualizations are saved to `visualization/outputs` directory by default

```sh
./run.sh viz
```

## model training

requires 8 GB of memory by default which is configurable at driver_memory in `config/modeling.yaml` file

```sh
./run.sh model
```


## parameter tuning

- best output model is saved to `modeling/outputs` directory by default
- hyper parameter performance summary is also stored to `modeling/outputs` directory as a CSV file
- tuning part requires 8 GB of memory which is configurable at driver_memory in `config/modeling.yaml` file

```sh
./run.sh tune
```

## inference

1. run below to launch model prediction api server

```sh
./run.sh api
```
- api server will be running on http://localhost:8000
- `/predict` endpoint processes texts and performs prediction
- first request to `/predict` might be slow due to spark model deserialization

2. open `app/frontend.html` file in browser

3. type or paste text in the web page to get model prediction 
