import os
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import split
from hydra.utils import get_original_cwd
from pyspark.sql.functions import lit


def get_feature_name(feature_current_name, prefix):
    feature_name = feature_current_name.split("_", 1)[-1]
    return "{}_{}".format(prefix, feature_name)


def create_spark_session(app_name):
    return (
        SparkSession.builder.config("spark.driver.memory", "8g")
        .appName(app_name)
        .getOrCreate()
    )


def combine_data(data):
    assert len(data) > 1
    acc = data[0]
    for item in data[1:]:
        acc = acc.union(item)
    return acc


def explode_text(dataset, text_feature_name):
    new_column = get_feature_name(text_feature_name, "exploded")
    return (
        dataset.withColumn(new_column, split(dataset[text_feature_name], " ")),
        new_column,
    )


def read_and_transform_datasets(session, config):
    dfs = []
    target_var = config.variables.target_var
    current_path = get_original_cwd()

    for idx, in_path in enumerate(config.dataset.input_paths):
        full_in_path = os.path.join(current_path, in_path)
        header = config.dataset.headers[idx]
        target_val = config.variables.target_vals[idx]
        df = session.read.csv(full_in_path, header=header, inferSchema=True)
        df = df.withColumn(target_var, lit(target_val)).fillna("")
        dfs.append(df)

    return dfs


def get_train_test_set(session, config):
    dfs = read_and_transform_datasets(session, config)
    combined = combine_data(dfs)

    data, text_col_name = explode_text(combined, config.variables.text_var)

    train_split = config.splitting.train_split
    train_set, test_set = data.randomSplit(
        [train_split, 1 - train_split], seed=config.splitting.seed
    )

    return train_set, test_set, text_col_name


def get_pipeline(text_col_name, model, features_name):
    vectorizer = CountVectorizer(inputCol=text_col_name, outputCol=features_name)
    return Pipeline(stages=[vectorizer, model])


def evaluate(model, test_set, target_name):
    test_pred = model.transform(test_set)

    evaluator = BinaryClassificationEvaluator(
        labelCol=target_name, rawPredictionCol="prediction", metricName="areaUnderROC"
    )

    return evaluator.evaluate(test_pred)
