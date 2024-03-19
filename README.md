# fake news detection

Make sure to be inside project directory in your terminal

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

requires 8 GB of RAM by default which is configurable at driver_memory in `config/modeling.yaml` file

```sh
./run.sh model
```


## parameter tuning

- best output model is saved to `modeling/outputs` directory by default
- hyper parameter performance summary is also stored to `modeling/outputs` directory as a CSV file
- tuning part requires 8 GB of RAM which is configurable at driver_memory in `config/modeling.yaml` file

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

## References
- [fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [introduction-to-hydra-cc-a-powerful-framework-to-configure-your-data-science-projects](https://towardsdatascience.com/introduction-to-hydra-cc-a-powerful-framework-to-configure-your-data-science-projects-ed65713a53c6)
- [concatenate-two-pyspark-dataframes](https://stackoverflow.com/questions/37332434/concatenate-two-pyspark-dataframes)
- [spark-documentation](https://spark.apache.org/docs/3.1.1/)
- [logging-with-hydra](https://hydra.cc/docs/tutorials/basic/running_your_app/logging/)
- [how-can-i-add-a-blank-directory-to-a-git-repository](https://stackoverflow.com/questions/115983/how-can-i-add-a-blank-directory-to-a-git-repository)
- [what-are-the-differences-between-gitignore-and-gitkeep](https://stackoverflow.com/questions/7229885/what-are-the-differences-between-gitignore-and-gitkeep)
- [pyspark-java-lang-outofmemoryerror](https://stackoverflow.com/questions/32336915/pyspark-java-lang-outofmemoryerror-java-heap-space)
- [countvectorizer-hashingtf](https://towardsdatascience.com/countvectorizer-hashingtf-e66f169e2d4e)
- [pyspark-mllib-feature-extraction](https://spark.apache.org/docs/1.4.1/mllib-feature-extraction.html)
- [hydra-compose-api](https://hydra.cc/docs/advanced/compose_api/)
- [pandas-dataframe-to-json](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html)
