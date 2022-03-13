import os
import logging
from hydra.utils import get_original_cwd
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from modeling.utils import get_pipeline, evaluate, combine_data, explode_text


log = logging.getLogger(__name__)


def create_spark_session(config):
    return SparkSession.builder.appName(config.spark.app_name).getOrCreate()


def read_prepare_data(session, config):
    dfs = []
    target_var = config.variables.target_var
    current_path = get_original_cwd()

    for idx, in_path in enumerate(config.dataset.input_paths):
        full_in_path = os.path.join(current_path, in_path)
        header = config.dataset.headers[idx]
        target_val = config.variables.target_vals[idx]
        df = (
            session.read.csv(full_in_path, header=header, inferSchema=True)
            .withColumn(target_var, lit(target_val))
            .fillna("")
        )
        dfs.append(df)

    return dfs


def do_modeling(config):

    features = config.variables.feature_var
    target = config.variables.target_var

    spark = create_spark_session(config)

    dfs = read_prepare_data(spark, config)
    combined = combine_data(dfs)

    train_set, test_set = combined.randomSplit([0.85, 0.15], seed=config.splitting.seed)

    train, text_col_name = explode_text(train_set, config.variables.text_vars)
    test, _ = explode_text(test_set, config.variables.text_vars)

    lr = LogisticRegression(labelCol=target, featuresCol=features, maxIter=10)
    lr_pipeline = get_pipeline(text_col_name, lr, features)

    model_lr = lr_pipeline.fit(train)
    log.info(
        "Logistic regression test AUC score : {}".format(
            evaluate(model_lr, test, target)
        )
    )

    nb = NaiveBayes(labelCol=target, featuresCol=features)
    nb_pipeline = get_pipeline(text_col_name, nb, features)

    model_nb = nb_pipeline.fit(train)
    log.info(
        "Multinomial NB test AUC score : {}".format(evaluate(model_nb, test, target))
    )
