import itertools
import matplotlib.pyplot as plt
import seaborn as sns

def write_graphs(dataset):
    columns = dataset.columns[:9]
    plt.subplots(figsize=(18, 15))
    length = len(columns)
    for i, j in itertools.zip_longest(columns, range(length)):
        plt.subplot((length / 2), 3, j + 1)
        plt.subplots_adjust(wspace=0.2, hspace=0.5)
        dataset[i].hist(bins=20, edgecolor='black')
        plt.title(i)
    plt.show()


def create_loss_results(results):
    plt.plot(results.history['accuracy'], label="accuracy")
    plt.plot(results.history['val_accuracy'], label="validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def write_distplot_column(data, bins, title):
    sns.distplot(data, bins=bins)
    plt.title(title)
