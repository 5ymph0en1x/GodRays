import numpy as np

def zscore(array, value):
	zscore = (value - np.mean(array)) / np.std(array)
	return zscore