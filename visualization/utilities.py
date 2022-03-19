import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS


def get_save_path(out_folder, *paths):
    return os.path.join(out_folder, *paths)


def word_counter(news):
    return len(news.split())


def get_avg_word_len(news):
    return np.average([len(word) for word in news.split()])


def plot_label_distribution(data, save_path, config):
    target_var = config.variables.target_var
    label_dist = data[target_var].value_counts(normalize=True).reset_index()

    plt.figure(figsize=(6, 4))
    ax = plt.bar(
        y=label_dist[target_var],
        x=label_dist["index"],
        color=config.style.target_colors,
        width=config.visualizations.bar_width,
    )
    ax.bar_label(ax.containers[0], fmt="%.2f")
    ax.get_legend().remove()
    plt.xlabel("Is fake news ?")
    plt.ylabel("Proportion")
    plt.title("Label distribution of news")
    plt.xticks(rotation=0)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_word_cloud(texts, save_path):
    word_cloud = WordCloud(stopwords=STOPWORDS, width=1600, height=800).generate(
        " ".join(texts)
    )
    word_cloud.to_file(save_path)


def plot_word_count_distribution(texts, save_path, bins="auto"):
    plt.figure(figsize=(8, 5))
    sns.histplot(texts.apply(word_counter), stat="density", bins=bins)
    plt.xlabel("Word count interval")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_avg_word_len_distribution(texts, save_path, bins="auto"):
    plt.figure(figsize=(8, 5))
    sns.histplot(texts.apply(get_avg_word_len), stat="density", bins=bins)
    plt.xlabel("Average word length interval")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_category_distribution(category, column_name, save_path):
    var_name_cap = str(column_name).capitalize()
    cat_dist = category.value_counts(normalize=True).reset_index()
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(y=cat_dist.columns[-1], x="index", data=cat_dist)
    ax.bar_label(ax.containers[0], fmt="%.2f")
    plt.xlabel(var_name_cap)
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
