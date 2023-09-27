# fake news detection

Repo created to store source code of fake news detection end-to-end machine learning project which involves work from data cleaning, data visualization, model training, parameter tuning to model inference

<p align="center">
<img width="800" alt="image" src="https://github.com/ppkgtmm/fake-news-detection/blob/main/images/diagram.png?raw=true"> 
</p>

## Data cleaning
Primary focus of data cleaning step was to make model better generalize on new data by not only removing special characters and html new line tag but also normalizing high cardinality values such as urls, email addresses, hashtags, social media mentions and numbers. Expansion of modal verbs was also conducted to standardize such words and emphasize negation if any was present in the sentence

## Data visualization
Firstly, distribution of labels were visualized to see if there is any class imbalance. To get information about topics that news inputs are related to, visualization of new subject distribution was done. Afterwards, common words in news inputs, distribution of word count as well as average word length per news were observed through world cloud and histogram visualizations respectively. Findings from data visualization step can be found [here](https://github.com/ppkgtmm/fake-news-detection/blob/main/visualization/README.md)

## Modeling and tuning
In modeling part, 2 algorithms were used for classification namely Logistic Regression and Naive Bayes. Both algorithms' predictive power were compared using ROC AUC score and the [result](https://github.com/ppkgtmm/fake-news-detection/blob/main/outputs/2023-09-17/18-37-35/model.log) is shown below. The reason behind using ROC AUC score as a model performance comparison metric was that the news dataset is not imbalanced and high ROC AUC score also helps to ensure model is good at separating between real and fake news. Finally based on the ROC AUC score, Logistic Regression algorithm was selected and further tuned with grid search approach

```txt
[2023-09-17 18:37:35,613][__main__][INFO] - Config param validation successful
[2023-09-17 18:37:35,613][__main__][INFO] - Begin modeling process
[2023-09-17 18:38:19,768][modeling.training][INFO] - Logistic regression validation AUC score : 0.9840257205725704
[2023-09-17 18:38:34,351][modeling.training][INFO] - Multinomial NB validation AUC score : 0.945421586414332
[2023-09-17 18:38:34,352][__main__][INFO] - End modeling process
```

## Model inference
A couple of news inputs containing fake news followed by real news were sent for prediction. The prediction results are shown in the second screenshot below; the first item of probability values corresponds to the REAL news class while the other corresponds to the FAKE news class

![image](https://github.com/ppkgtmm/fake-news-detection/blob/main/images/api-input.png?raw=true)
![image](https://github.com/ppkgtmm/fake-news-detection/blob/main/images/api-output.png?raw=true)

Simple front end for interacting with machine learning model was also implemented

![image](https://github.com/ppkgtmm/fake-news-detection/blob/main/images/ui-real-news.png?raw=true)
![image](https://github.com/ppkgtmm/fake-news-detection/blob/main/images/ui-fake-news.png?raw=true)

## Usage

Make sure to be inside project directory in your terminal

#### Initialization
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

Each of the following steps uses a YAML configuration file stored in `config` folder

#### Data cleaning

```sh
./run.sh clean
```

By default, processed version of dataset is saved to `data` directory with `prep` suffix


#### Data visualization

```sh
./run.sh viz
```

By default, all visualizations are saved to `visualization/outputs` directory

#### Model training

```sh
./run.sh model
```

This part requires 8 GB of RAM by default but the limit is configurable by editing driver_memory in `config/modeling.yaml` file

#### Parameter tuning

```sh
./run.sh tune
```
- By default, best output model is saved to `modeling/outputs` directory. Hyper parameter performance summary is also stored to `modeling/outputs` directory as a CSV file
- Tuning part also require 8 GB of RAM by default but the limit is configurable by editing driver_memory in `config/modeling.yaml` file
  
#### Inference

1. Run below to launch model prediction API server
```sh
./run.sh api
```
- By default, API server is running on `http://localhost:8000` where endpoint `/predict` processes texts and performs prediction
- First request to `/predict` endpoint might be slow due to spark model deserialization

2. Open `app/frontend.html` file in browser
3. Type or paste text in the web page to get prediction from model

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
