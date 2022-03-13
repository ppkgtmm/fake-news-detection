import os
import logging
from hydra.utils import get_original_cwd
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, udf, split
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import CountVectorizer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

log = logging.getLogger(__name__)


# def explode_text(text):
#     if text is None:
#         return [""]
#     return text.split()
#
#
# explode_udf = udf(explode_text, ArrayType(StringType()))


def create_spark_session():
    return SparkSession.builder.appName("app").getOrCreate()


def read_prepare_data(session, config):
    dfs = []
    target_var = config.variables.target_var
    current_path = get_original_cwd()

    for idx, in_path in enumerate(config.dataset.input_paths):
        full_in_path = os.path.join(current_path, in_path)
        header = config.dataset.headers[idx]
        target_val = config.variables.target_vals[idx]
        df = session.read.csv(full_in_path, header=header, inferSchema=True).withColumn(
            target_var, lit(target_val)
        )
        dfs.append(df)

    return dfs


def combine_data(data):
    assert len(data) > 1
    acc = data[0]
    for item in data[1:]:
        acc = acc.union(item)
    return acc


def run_explode_text(dataset, text_features):

    acc = dataset
    result_column = []

    for text_feature in text_features:
        new_column = "exploded_{}".format(text_feature)
        acc = acc.withColumn(new_column, split(acc[text_feature], " "))
        result_column.append(new_column)

    return acc, result_column


def get_count_vectorizer(text_features):

    vectorizers = []
    result_column = []

    for text_feature in text_features:
        output_col = "vectorized_{}".format(text_feature)
        result_column.append(output_col)
        count_vectorizer = CountVectorizer(inputCol=text_feature, outputCol=output_col)
        vectorizers.append(count_vectorizer)

    return vectorizers, result_column


def build_model(config):

    features = config.variables.feature_var
    target = config.variables.target_var

    spark = create_spark_session()

    dfs = read_prepare_data(spark, config)
    combined = combine_data(dfs)

    train_set, test_set = combined.randomSplit([0.85, 0.15], seed=config.splitting.seed)

    train, text_col_name = run_explode_text(train_set, config.variables.text_vars)
    test, _ = run_explode_text(test_set, config.variables.text_vars)

    count_vectorizers, vector_names = get_count_vectorizer(text_col_name)
    assembler = VectorAssembler(inputCols=vector_names, outputCol=features)

    lr = LogisticRegression(labelCol=target, featuresCol="features", maxIter=10)

    pipeline = Pipeline(stages=count_vectorizers + [assembler, lr])

    model = pipeline.fit(train)
    test_pred = model.transform(test)
    evaluator = BinaryClassificationEvaluator(
        labelCol=target, rawPredictionCol="prediction", metricName="areaUnderROC"
    )

    log.info("Test AUC score : {}".format(evaluator.evaluate(test_pred)))
