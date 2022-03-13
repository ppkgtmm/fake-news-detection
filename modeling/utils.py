from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, VectorAssembler
from pyspark.sql.functions import split


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
