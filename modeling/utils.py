import os
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, VectorAssembler
from pyspark.sql.functions import split
from hydra.utils import get_original_cwd
from pyspark.sql.functions import lit


def create_spark_session(app_name):
    return SparkSession.builder.appName(app_name).getOrCreate()


def combine_data(data):
    assert len(data) > 1
    acc = data[0]
    for item in data[1:]:
        acc = acc.union(item)
    return acc


def explode_text(dataset, text_features):
    acc = dataset
    result_column = []

    for text_feature in text_features:
        new_column = "exploded_{}".format(text_feature)
        acc = acc.withColumn(new_column, split(acc[text_feature], " "))
        result_column.append(new_column)

    return acc, result_column


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

    data, text_col_name = explode_text(combined, config.variables.text_vars)

    train_split = float(config.splitting.train_split)
    train_set, test_set = data.randomSplit(
        [train_split, 1 - train_split], seed=config.splitting.seed
    )

    return train_set, test_set, text_col_name


def get_count_vectorizer(text_features):
    vectorizers = []
    result_column = []

    for text_feature in text_features:
        output_col = "vectorized_{}".format(text_feature)
        result_column.append(output_col)
        count_vectorizer = CountVectorizer(inputCol=text_feature, outputCol=output_col)
        vectorizers.append(count_vectorizer)

    return vectorizers, result_column


def get_pipeline(text_col_name, model, features):
    count_vectorizers, vector_names = get_count_vectorizer(text_col_name)
    assembler = VectorAssembler(inputCols=vector_names, outputCol=features)

    return Pipeline(stages=count_vectorizers + [assembler, model])


def evaluate(model, test_set, target):
    test_pred = model.transform(test_set)

    evaluator = BinaryClassificationEvaluator(
        labelCol=target, rawPredictionCol="prediction", metricName="areaUnderROC"
    )

    return evaluator.evaluate(test_pred)
