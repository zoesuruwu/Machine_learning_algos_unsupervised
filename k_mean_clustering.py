import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class k_means_model:
    _cluster_df = None

    def __init__(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        classify: str,
        N_clusters: int,
        logger,
        scaled: bool,
    ):
        self._df = df
        self._X = X
        self._classify = classify
        self._N_clusters = N_clusters
        self._logger = logger
        self._scaled = scaled

    def fit_model(self):
        if self._scaled:
            self._X = StandardScaler().fit_transform(self._X)

        kmeans = KMeans(n_clusters=self._N_clusters).fit(self._X)
        # kmeans.labels_ is the expected class
        k_means_model._cluster_df = pd.DataFrame(
            np.hstack((self._X, kmeans.labels_.reshape(-1, 1))),
            columns=self._df.columns,
        )
        silhouette_score = metrics.silhouette_score(
            self._X, kmeans.labels_, metric="euclidean"
        )
        calinski_harabasz_score = metrics.calinski_harabasz_score(
            self._X, kmeans.labels_
        )
        return silhouette_score, calinski_harabasz_score

    def plot_expect_vs_original(self, x_axis: str, y_axis: str):
        fig, axes = plt.subplots(1, 2, figsize=(24, 6))
        sns.scatterplot(
            x=x_axis,
            y=y_axis,
            hue=self._classify,
            data=k_means_model._cluster_df,
            ax=axes[0],
        )
        sns.scatterplot(
            x=x_axis, y=y_axis, hue=self._classify, data=self._df, ax=axes[1]
        )
        # Set titles for the subplots
        axes[0].set_title(
            f"Whole dataset - Expection of K-means: scaled = ({self._scaled}), k= {self._N_clusters}"
        )
        axes[1].set_title(
            f"Whole dataset - Original: scaled = ({self._scaled}), k= {self._N_clusters}"
        )
        plt.show()

    def prediction(self, x_axis: str, y_axis: str):
        X_train, X_test, y_train, y_test = train_test_split(
            self._X, self._df[self._classify], test_size=0.2, random_state=42
        )
        kmeans = KMeans(n_clusters=self._N_clusters).fit(X_train)
        predictions = kmeans.predict(X_test)
        fig, axes = plt.subplots(1, 2, figsize=(24, 6))
        test_data = pd.DataFrame(
            np.hstack((X_test, y_test.values.reshape(-1, 1))), columns=self._df.columns
        )
        pred_data = pd.DataFrame(
            np.hstack((X_test, predictions.reshape(-1, 1))), columns=self._df.columns
        )
        sns.scatterplot(
            x=x_axis, y=y_axis, hue=self._classify, data=pred_data, ax=axes[0]
        )
        sns.scatterplot(
            x=x_axis, y=y_axis, hue=self._classify, data=test_data, ax=axes[1]
        )
        # Set titles for the subplots
        axes[0].set_title(
            f"Test dataset - Expectation(labeled) of K-means: scaled = ({self._scaled}), k= {self._N_clusters}"
        )
        axes[1].set_title(
            f"Test dataset - Original calss: scaled = ({self._scaled}), k= {self._N_clusters}"
        )
        plt.show()
