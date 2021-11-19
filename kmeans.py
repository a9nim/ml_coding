import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from sklearn.datasets import load_iris

class KMeans:
	"""K-Means Clustering.

	Attributes:
	  n_clusters (int): Number of clusters.
	  max_iter (int): Maximum number of iterations.
	  tol (float): Tolerance of difference in cluster centers of two consecutive
	  	iterations to declare convergence.
	  cluster_centers (ndarray of shape (n_clusters, _n_features)):
	  	`_n_features` is obtained from `fit()`.
	  labels (ndarray of shape (_n_samples,)): `_n_samples` is obtained from
	  	`fit()`.
	"""

	def __init__(self, n_clusters=8, max_iter=300, tol=0.0001):
		"""Initializes KMeans."""
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.tol = tol
		self.cluster_centers = None
		self.labels = None

		# Internal attributes
		#
		# Attributes:
		#   _n_samples (int): Number of samples.
		#   _n_features (int): Number of features.
		#   _n_iter (int): Number of iterations run.
		#   _sample_cluster_distances (ndarray of shape
		#	  (_n_samples, n_clusters)): Distance to cluster centers for each
		#	  sample.
		self._n_samples = 0
		self._n_features = 0
		self._n_iter = 0
		self._sample_cluster_distances = None

	def fit(self, X):
		"""Computes the K-Means clustering.

		Args:
		  X (ndarray of shape (n_samples, n_features)): Training samples.
		"""
		# Initialize the cluster centers.
		self._n_samples = X.shape[0]
		self._n_features = X.shape[1]
		self._n_iter = 0
		prev_cluster_centers = np.zeros(shape=(self.n_clusters,
			self._n_features))
		rng = np.random.default_rng()
		self.cluster_centers = X[rng.choice(self._n_samples,
			size=self.n_clusters, replace=False)]
		self._sample_cluster_distances = self._get_cluster_distances(X,
			self.cluster_centers)
		self.labels = np.argmin(self._sample_cluster_distances, axis=1)

		# Iteratively run clustering.
		while self._n_iter < self.max_iter and not self._converged(
			prev_cluster_centers, self.cluster_centers, self.tol):
			prev_cluster_centers = deepcopy(self.cluster_centers)
			self.cluster_centers = self._calc_cluster_centers(X, self.labels,
				self.n_clusters)
			self._sample_cluster_distances = self._get_cluster_distances(X,
				self.cluster_centers)
			self.labels = np.argmin(self._sample_cluster_distances, axis=1)
			self._n_iter += 1

	def predict(self, X):
		"""Predicts the cluster index for each sample.

		Args:
		  X (ndarray of shape (_n_samples, n_features)): Test samples.

		Returns:
		  (ndarray of shape (_n_samples,)): Index of the cluster each sample
		  	belongs to.
		"""
		cluster_distances = _get_cluster_distances(X, self.cluster_centers)
		labels = np.argmin(cluster_distances, axis=1)
		return labels
			

	@staticmethod
	def _converged(prev_cluster_centers, new_cluster_centers, tol):
		"""Compares the previous and new cluster centers to check convergence.

		Args:	
		  prev_cluster_centers (ndarray of shape (n_clusters, _n_features)):
		  	cluster centers from the previous iteration.
		  new_cluster_centers (ndarray of shape (n_clusters, _n_features)):
		  	cluster centers of the current iteration.
		  tol (float): tolerance of difference.

		Returns:
		  (bool): whether the cluster centers have converged.
		"""
		return np.linalg.norm(prev_cluster_centers - new_cluster_centers) <= tol

	@staticmethod
	def _get_cluster_distances(X, cluster_centers):
		"""Computes the distance to cluster centers for each sample.

		Args:
		  X (ndarray of shape (_n_samples, _n_features)): Training samples.
		  cluster_centers (ndarray of shape (n_clusters, _n_features)):
		  	Cluster centers.

		Returns:
		  (ndarray of shape (_n_samples, n_clusters)): Distance to cluster
		  	centers.
		"""
		cluster_distances = np.zeros(shape=(X.shape[0],
			cluster_centers.shape[0]))
		for i in range(cluster_centers.shape[0]):
			cluster_distances[:, i] = np.linalg.norm(X - cluster_centers[i],
				axis=1);
		return cluster_distances

	@staticmethod
	def _calc_cluster_centers(X, labels, n_clusters):
		"""Computes the cluster centers.

		Args:
		  X (ndarray of shape (_n_samples, _n_features)): Training samples.
		  labels (ndarray of shape (_n_samples,)): Cluster assignment for each
		    sample.
		  n_clusters (int): Number of clusters.

		Returns:
		  ndarray of shape (n_clusters, _n_features).
		"""
		new_cluster_centers = np.zeros(shape=(n_clusters, X.shape[1]))
		for i in range(n_clusters):
			new_cluster_centers[i] = np.mean(X[labels == i], axis=0)
		return new_cluster_centers

def main():
	# Fit Iris Flower Dataset.
	iris_dataset = load_iris()
	X = iris_dataset.data
	km = KMeans(n_clusters=3)
	km.fit(X)

	# Plot the data
	fig, ax = plt.subplots()
	colors = ['red', 'green', 'blue']
	for i in range(km._n_samples):
		ax.scatter(X[i, 0], X[i, 1], color=colors[km.labels[i]])
	for i in range(km.n_clusters):
		ax.scatter(km.cluster_centers[i][0], km.cluster_centers[i][1],
			s=128, marker='*', color='purple')
	plt.show()


if __name__=="__main__":
	main()
