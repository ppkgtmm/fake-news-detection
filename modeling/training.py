import logging
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from modeling.utils import (
    create_spark_session,
    get_train_test_set,
    get_pipeline,
    evaluate,
)

log = logging.getLogger(__name__)


def do_modeling(config):
    features = config.variables.feature_var
    target = config.variables.target_var

    spark = create_spark_session(config.spark.app_name)

    train, test, text_col_name = get_train_test_set(spark, config)

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
