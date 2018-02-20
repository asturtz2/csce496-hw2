import numpy as np
def split_data(data, labels, proportion):

	size = data.shape[0]
	np.random.seed(42)
	s = np.random.permutation(size)
	split_idx = int(proportion * size)
	#data = (data//255)
	labels2 = correctLabel(labels)
	return data[s[:split_idx]], data[s[split_idx:]], labels2[s[:split_idx]], labels2[s[split_idx:]]

def correctLabel(labels):
	m = len(labels)
	labels2 = np.zeros((m,10))
	for i in range(0,m-1):
		labels2[i,int(labels[i])]=1
	return labels2
