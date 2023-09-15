import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class pca_model:
    def __init__(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        classify: str,
        N_comp: int,
        logger,
        scaled: bool,
    ):
        self._df = df
        self._X = X
        self._classify = classify
        self._N_comp = N_comp
        self._logger = logger
        self._scaled = scaled

    def fit_model(self):
        if self._scaled:
            # Scale the data
            self._X = StandardScaler().fit_transform(self._df)
        pca = PCA(self._N_comp).fit(self._X)
        pc = pca.transform(self._X)
        pc1 = pc[:, 0]
        pc2 = pc[:, 1]

        # Plot principal components
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        axes[0].scatter(
            pc1,
            pc2,
            c=self._df[self._classify],
            alpha=0.5,
        )
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        # plot the variance per PC
        var = pca.explained_variance_  # percentage of variance explained
        labels = ["PC" + str(i + 1) for i in range(self._N_comp)]
        axes[1].bar(labels, var)
        axes[1].set_xlabel("Pricipal Component")
        axes[1].set_ylabel("Proportion of Variance Explained")
        plt.tight_layout()
        plt.show()

    def prediction(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self._X, self._df[self._classify], test_size=0.2, random_state=42
        )
        pca = PCA(n_components=self._N_comp)  # Specify the desired number of components
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        # choosing logistic regression as classifier
        classifier = LogisticRegression(multi_class="multinomial", solver="lbfgs")
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = classification_report(y_test, predictions)
        print(
            f"Accuracy report - scaled = ({self._scaled}) with Nr of components = {self._N_comp}:"
        )
        print(accuracy)

    def plot_number_of_components(self):
        pca = PCA().fit(self._X)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("number of components")
        plt.ylabel("cumulative explained variance")
        plt.show()
