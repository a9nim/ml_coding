import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes

class LinearRegression:
	"""Runs linear regression.

	Attributes:
  	  n_features (int): Number of features.
      n_samples (int): Number of samples.
      coef (ndarray of shape (_n_features,), float): Coefficients for the linear
        regression.
      intercept (float): The bias term in the linear model.
	"""

	def __init__(self, option='normal_equation'):
		"""Initializes Linear Regression.

		Args:
	  	option (string): Possble options are {"normal_equation",
	    "gradient_descent"}.

		Internal attributes:
	  	_option (string): Linear model option.
	  	_weights (ndarray of shape (n_features + 1, 1), float): Linear model
	  	  weights.
	  	_n_iter (int): Number of iterations for gradient descent.
	  	_batch_size (int): Batch size for gradient descent.
	  	_learning_rate (float): Learning rate for gradient descent.
	  	_mu (ndarray of shape (n_features,), float): Mean of the training data
	  	  for normalization.
	  	_sigma (ndarray of shape (n_features,), float): Standard deviation of
	  	  the training data for normalization.
		"""
		self.n_features = 0
		self.n_samples = 0
		self.coef = None
		self.intercept = 0.0
		self._option = option
		self._weights = None
		self._n_iter = 1000
		self._batch_size = 8
		self._learning_rate = 0.001
		self._mu = None
		self._sigma = None

	def fit(self, X, y):
		"""Computes the linear regression model.
	
		Args:
		X (ndarray of shape (n_samples, n_features), float): Training data.
		y (ndarray of shape (n_samples,), float): Ground-truth data.
		"""		
		X, self._mu, self._sigma = self._normalize_input(X)
		self.n_features = X.shape[1]
		self.n_samples = X.shape[0]
		X_b = np.hstack((np.ones(shape=(self.n_samples, 1)), X))
		y = np.expand_dims(y, axis=1)
		if self._option == 'normal_equation':
			# Normal equation:
			#   theta = inv(X.T * X) * (X.T * y)
			self._weights = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T.dot(y))
		elif self._option == 'gradient_descent':
			# Loss function:
			#   L(theta) = 1/2 * (X * theta - Y).T * (X * theta - Y)
			# Gradient descent (denominator layout):
			# 	dL/dtheta = X.T * (X * theta - Y)
			rng = np.random.default_rng()
			self._weights = rng.random(size=(self.n_features + 1, 1))
			for i in range(self._n_iter):
				# Shuffle the training data.
				X_b_y = np.hstack((X_b, y))
				rng.shuffle(X_b_y, axis=0)
				for j in np.arange(start=0, stop=self.n_samples, step=self._batch_size):
					X_b_batch = X_b_y[j:j + self._batch_size, :-1]
					y_batch = X_b_y[j:j + self._batch_size, [-1]]
					gradient = X_b_batch.T.dot(X_b_batch.dot(self._weights) - y_batch)
					self._weights -= self._learning_rate * gradient
		self.coef = self._weights[1:, 0]
		self.intercept = self._weights[0, 0]

	def predict(self, X):
		"""Predicts the value for each sample.

		Args:
		X (ndarray of shape (n_samples, n_features), float): Test data.

		Returns:
		(ndarray of shape (n_samples,), float): Prediction data.
		"""
		X, _, _ = self._normalize_input(X, self._mu, self._sigma)
		n_samples = X.shape[0]
		X_b = np.hstack((np.ones(shape=(n_samples, 1)), X))
		y_predict = X_b.dot(self._weights)
		return y_predict[:, 0]

	@staticmethod
	def _normalize_input(X, mu=None, sigma=None):
		"""Normalizes the input.

		Args:
		  X (ndarray of shape (n_samples, n_features), float): Training or
		    prediction data.
		  mu (ndarray of shape (n_features,), float): Mean of the training or
		    prediction data. If None, use the mean of the input `X`.
		  sigma (ndarray of shape (n_features,), float): Standard deviation of
		    the training or prediction data. If None, use the standard deviation
		    of the input `X`.

		Returns:
		  A tuple of the following elements:
		  ndarray of shape (n_samples, n_features): Normalized input values in
		    the range [-1, 1].
		  float: `mu` if provided, otherwise the mean of the input.
		  float: `sigma` if provided, otherwise the standard deviation of the
		  	input.
		"""
		mu = np.mean(X, axis=0) if mu is None else mu
		sigma = np.std(X, axis=0) if sigma is None else sigma
		X = (X - mu) / sigma
		return X, mu, sigma

def main():
    # Fit Diabetes Dataset.
    X, y = load_diabetes(return_X_y=True)
    X = X[:,:2]
    X_train = X[:-40]
    y_train = y[:-40]
    X_test = X[-40:]
    y_test = y[-40:]
    lr = LinearRegression(option='gradient_descent')
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    print('Number of samples: \n', lr.n_samples)
    print('Number of features: \n', lr.n_features)
    print('Coeffcient: \n', lr.coef)
    print('Intercept: \n', lr.intercept)

    # Plot the data.
    X_test, _, _ = lr._normalize_input(X_test)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.scatter3D(X_test[:, 0], X_test[:, 1], y_test, color='black')
    surface_0, surface_1 = np.meshgrid(np.linspace(start=-1, stop=1, num=3),
    	np.linspace(start=-1, stop=1, num=3))
    ax.plot_surface(surface_0, surface_1, lr.predict(np.column_stack(
    	(surface_0.ravel(), surface_1.ravel()))).reshape(surface_0.shape),
    	color='red', alpha=0.3)
    plt.show()

if __name__ == '__main__':
	main()
