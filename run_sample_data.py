import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from k_mean_clustering import k_means_model
from pca import pca_model


def seeds_data(logger):
    cols = [
        "area",
        "perimeter",
        "compactness",
        "length",
        "width",
        "asymmetry",
        "groove",
        "class",
    ]
    df = pd.read_csv("data/seeds_dataset.txt", names=cols, sep="\s+")  # s: space
    print("--- Seeds data info: ---")
    print(df.info())
    print("--- Seeds data statistics: ---")
    print(df.describe().to_string())
    N_class = len(set(df["class"]))

    # visualization
    for i in range(len(cols) - 1):
        for j in range(i + 1, len(cols) - 1):
            x_label = cols[i]
            y_lable = cols[j]
            sns.scatterplot(x=x_label, y=y_lable, data=df, hue="class")
            plt.show()

    K_MEANS_SCORES = pd.DataFrame(
        index=["scaled", "not_scaled"],
        columns=["silhouette_score", "calinski_harabasz_score"],
    )
    logger.info("Starting with K-means model - not scaled data...")
    X = df[cols[:-1]].values
    clustering_class = k_means_model(df, X, "class", N_class, logger, scaled=False)
    silhouette_score, calinski_harabasz_score = clustering_class.fit_model()
    # The following x, y are only for visualization only. The model is fit on all variables except for class
    x = "compactness"
    y = "asymmetry"
    clustering_class.plot_expect_vs_original(x, y)
    clustering_class.prediction(x, y)
    print(
        f"Based on not scaled whole dataset, silhouette_score = {silhouette_score}, and calinski_harabasz_score = {calinski_harabasz_score}"
    )
    K_MEANS_SCORES.loc["not_scaled", "silhouette_score"] = silhouette_score
    K_MEANS_SCORES.loc[
        "not_scaled", "calinski_harabasz_score"
    ] = calinski_harabasz_score

    logger.info("Starting with K-means model - scaled data...")
    X = df[cols[:-1]].values
    clustering_class = k_means_model(df, X, "class", N_class, logger, scaled=True)
    silhouette_score, calinski_harabasz_score = clustering_class.fit_model()
    logger.info(f"Plotting the expectation vs original - x-axis={x}, y-axis={y}...")
    clustering_class.plot_expect_vs_original(x, y)
    clustering_class.prediction(x, y)
    print(
        f"Based on scaled whole dataset, silhouette_score = {silhouette_score}, and calinski_harabasz_score = {calinski_harabasz_score}"
    )
    K_MEANS_SCORES.loc["scaled", "silhouette_score"] = silhouette_score
    K_MEANS_SCORES.loc["scaled", "calinski_harabasz_score"] = calinski_harabasz_score

    print("Not scaled dataset vs scaled dataset: \n", K_MEANS_SCORES.to_string())

    logger.info("Starting with PCA model - not scaled data...")
    not_scaled_pca = pca_model(df, X, "class", N_class, logger, scaled=False)
    not_scaled_pca.fit_model()
    not_scaled_pca.plot_number_of_components()
    not_scaled_pca.prediction()
    logger.info("Starting with PCA model - scaled data...")
    scaled_pca = pca_model(df, X, "class", N_class, logger, scaled=True)
    scaled_pca.fit_model()
    scaled_pca.plot_number_of_components()
    scaled_pca.prediction()
