# fake news detection

Make sure to be inside project directory in your terminal

**Initialization**
1. Install [Python 3.8](https://www.python.org/downloads/)
2. Install [Java 8](https://www.oracle.com/java/technologies/downloads/)
3. Run below to grant execute permission to helper script

```sh
chmod +x run.sh
```

4. Run the following to initialize project
```sh
./run.sh init
```

<br/>

Each of the following steps uses a YAML configuration file stored in `config` folder

**Data Cleaning**

```sh
./run.sh clean
```

By default, processed version of dataset is saved to `data` directory with `prep` suffix

<br/>

**Data Visualization**

```sh
./run.sh viz
```

By default, all visualizations are saved to `visualization/outputs` directory

<br/>

**Model Training**

```sh
./run.sh model
```

This part requires 8 GB of RAM by default but the limit is configurable by editing driver_memory in `config/modeling.yaml` file

<br/>

**Parameter Tuning**

```sh
./run.sh tune
```
- By default, best output model is saved to `modeling/outputs` directory. Hyper parameter performance summary is also stored to `modeling/outputs` directory as a CSV file
- Tuning part also require 8 GB of RAM by default but the limit is configurable by editing driver_memory in `config/modeling.yaml` file

<br/>

**Inference**

1. Run below to launch model prediction API server
```sh
./run.sh api
```
- By default, API server is running on `http://localhost:8000` where endpoint `/predict` processes texts and performs prediction
- First request to `/predict` endpoint might be slow due to spark model deserialization

2. Open `app/frontend.html` file in browser
3. Type or paste text in the web page to get prediction from model

<br/>

**Final Output**
![image](https://github.com/ppkgtmm/fake-news-detection/blob/main/images/ui-fake-news.png?raw=true)

<br/>

**References**
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
