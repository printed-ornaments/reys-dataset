import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from .utils.metrics import Scores
from .utils.logger import print_info


class FleuronsClusterEval:
    def __init__(self, num_classes, num_clusters=None):
        """
        Clustering Evaluation for Fleurons block dataset.

        Args:
            num_classes (int): Number of classes.
            num_clusters (int, optional): Number of clusters. Defaults to None.
        """
        self.scores = Scores(num_classes, num_clusters)

    def load_from_tsv(self, results, idx="id", cluster="cluster"):
        """
        Load results from tsv file.

        Args:
            results (tsv): TSV file. Default format is 'file_name\tid\tcluster'.
            idx (str, optional): Header of image id. Defaults to 'id'.
            cluster (str, optional): Header of cluster index. Defaults to 'cluster'.
        """
        print_info("Loading data...")
        data = pd.read_csv(results, sep="\t", header=0)
        self.cluster_assignments = data[cluster].to_numpy()
        self.true_labels, _ = pd.factorize(data[idx])
        self.scores.update(self.true_labels, self.cluster_assignments)
        print_info("Done.")

    def load_numpy(self, true_labels, cluster_assignments):
        """
        Load results from numpy arrays.

        Args:
            true_labels (numpy.ndarray): Ground truth labels of images.
            cluster_assignments (numpy.ndarray): Cluster assignments of images.
        """
        print_info("Loading data...")
        self.cluster_assignments = cluster_assignments
        self.true_labels = true_labels
        self.scores.update(self.true_labels, self.cluster_assignments)
        print_info("Done.")

    def evaluate(self):
        """
        Compute all metrics.
        """
        print_info("Running evaluation...")
        self.acc = self.global_accuracy()
        self.avg_acc = self.average_accuracy()
        self.nmi = self.nmi_score()
        print_info("Done.")

    def summarize(self):
        """
        Print evaluation summary.
        """
        if hasattr(self, "acc"):
            print(f"Global Accuracy:\t{self.acc:.3f}")
        if hasattr(self, "avg_acc"):
            print(f"Average Accuracy:\t{self.avg_acc:.3f}")
        if hasattr(self, "acc"):
            print(f"Normalized Mutual Info. Score:\t{self.nmi:.3f}")

    def global_accuracy(self):
        """
        Compute global accuracy.

        Returns:
            float: Global accuracy.
        """
        matrix = self.scores.compute_confusion_matrix()
        return np.diag(matrix).sum() / matrix.sum()

    def average_accuracy(self):
        """
        Compute average class-wise accuracy.

        Returns:
            float: Average accuracy.
        """
        matrix = self.scores.compute_confusion_matrix()
        with np.errstate(divide="ignore", invalid="ignore"):
            acc_by_class = np.diag(matrix) / matrix.sum(axis=1)
        return np.mean(np.nan_to_num(acc_by_class))

    def nmi_score(self):
        """
        Compute the normalized mutual information (NMI) for clustering.

        Returns:
            float: Normalized Mutual Information.
        """
        return nmi_score(self.true_labels, self.cluster_assignments)
