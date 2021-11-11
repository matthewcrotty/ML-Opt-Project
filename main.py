import numpy as np
from hmmlearn import hmm
# https://hmmlearn.readthedocs.io/en/latest/tutorial.html#available-models

model = hmm.GaussianHMM(n_components=10, covariance_type="full")

# 10 Hmm models for each class
# run data point through all models to find best fit