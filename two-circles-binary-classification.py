from cProfile import label
from sklearn.datasets import make_circles


from sklearn.datasets import make_circles
from numpy import where
from matplotlib import pyplot

X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)

for i in range(2):
    samples_ix = where(y == i)
    pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))

pyplot.legend()
pyplot.show()