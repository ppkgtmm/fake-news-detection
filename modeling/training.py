import logging
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from modeling.utilities import (
    create_spark_session,
    get_train_test_set,
    get_pipeline,
    evaluate,
)

log = logging.getLogger(__name__)


def do_modeling(config):
    features = config.variables.feature_var
    target = config.variables.target_var

    spark = create_spark_session(config.spark.app_name, config.spark.driver_memory)

    train, _, text_col_name = get_train_test_set(spark, config)

    one_over_folds = 1 / config.cv.n_folds
    train, val = train.randomSplit(
        [1 - one_over_folds, one_over_folds], seed=config.splitting.seed
    )

    lr = LogisticRegression(labelCol=target, featuresCol=features, maxIter=10)
    lr_pipeline = get_pipeline(text_col_name, lr, features)

    model_lr = lr_pipeline.fit(train)
    log.info(
        "Logistic regression validation AUC score : {}".format(
            evaluate(model_lr, val, target)
        )
    )

    nb = NaiveBayes(labelCol=target, featuresCol=features)
    nb_pipeline = get_pipeline(text_col_name, nb, features)

    model_nb = nb_pipeline.fit(train)
    log.info(
        "Multinomial NB validation AUC score : {}".format(
            evaluate(model_nb, val, target)
        )
    )
