import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

def t_sne(vectors, data):

    warnings.filterwarnings("ignore")

    t_sne_2 = TSNE(n_components=2)
    t_sne_3 = TSNE(n_components=3)

    t_sne_2_prediction = t_sne_2.fit_transform(np.array(vectors))
    t_sne_3_prediction = t_sne_3.fit_transform(np.array(vectors))


    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax2 = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2, projection='3d')

    i = 0
    while i < len(data) - 1:
        for j in range(i + 1, len(data)):
            if data['author_email'][i] != data['author_email'][j] or j == len(data) - 1:
                ax2.scatter(t_sne_2_prediction[i:j, 0], t_sne_2_prediction[i:j, 1], s=1, label=data['author_email'][i])
                ax3.scatter(t_sne_3_prediction[i:j, 0], t_sne_3_prediction[i:j, 1], t_sne_3_prediction[i:j, 2], s=1, label=data['author_email'][i])
                i = j

    plt.legend(loc=5, bbox_to_anchor=(2, 0.5), markerscale = 5)
    fig.suptitle("2D and 3D Visualization using t-SNE")
    plt.show()

    return t_sne_2_prediction, t_sne_3_prediction


def pca(vectors, data):

    warnings.filterwarnings("ignore")

    pca2 = PCA(n_components=2)
    pca3 = PCA(n_components=3)

    pca_2_prediction = pca2.fit_transform(vectors)
    pca_3_prediction = pca3.fit_transform(vectors)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax2 = fig.add_subplot(1, 2, 1)
    ax3 = fig.add_subplot(1, 2, 2, projection='3d')

    i = 0
    while i < len(data) - 1:
        for j in range(i + 1, len(data)):
            if data['author_email'][i] != data['author_email'][j] or j == len(data) - 1:
                ax2.scatter(pca_2_prediction[i:j, 0], pca_2_prediction[i:j, 1], s=1, label=data['author_email'][i])
                ax3.scatter(pca_3_prediction[i:j, 0], pca_3_prediction[i:j, 1], pca_3_prediction[i:j, 2], s=1, label=data['author_email'][i])
                i = j
    
    plt.legend(loc=5, bbox_to_anchor=(2, 0.5), markerscale = 5)
    fig.suptitle("2D and 3D Visualization using PCA")
    plt.show()

    return pca_2_prediction, pca_3_prediction