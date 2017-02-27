import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

init_op = tf.initialize_all_variables()

def D(X, Z):
	#print(X)
	#print(Z)
	sub = X[:,:,tf.newaxis] - tf.transpose(Z)
	matD = tf.matmul(tf.transpose(sub, perm=[0,2,1]), sub)
	return tf.matrix_diag_part(matD)

def find_knn(diff, k, targs):
	# k = 5
	#print(tData)
	#print(vData)
	#print(sess.run(diff))

	top_k = tf.nn.top_k(-diff, k)
	idx = top_k.indices
	idx_np = idx.eval()
	#print(idx_np)

	#print(sum(targs[idx_np])/k)
	
	resp = np.zeros(80)	
	#print(resp.shape)
	
	resp[idx_np] = 1.0 /k

	#print(resp)

	return resp

	# resp = tf.Variable(tf.zeros_like(diff))
	# upd = tf.ones([1,5], tf.float64)
	# upd = upd + 1/k
	# print(sess.run(idx))
	# print("RESP")
	# print(resp)
	# print("UPD")
	# print(upd)
	# print("IDX")
	# print(idx)
	# resp = tf.scatter_update(resp, idx, upd)

np.random.seed(521)

with tf.Session() as sess:
	Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
	Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
	+ 0.5 * np.random.randn(100 , 1)
	
	randIdx = np.arange(100)
	np.random.shuffle(randIdx)
	
	trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
	validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
	testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]
	
	#sess.run(init_op)
	#print(Data[randIdx[10:30]])
	#print(Data[randIdx[:10]])
	#print(sess.run(D(Data[randIdx[10:30]], Data[randIdx[:10]] )))
	#print(trainData.shape)
	#print(validData.shape
	
	X = np.linspace(0.0, 11.0, num = 1000)[:, np.newaxis]

	y = np.zeros(1000)
	j = 0
	i = 0
	k = 1
	# print(Data[0])
	# print(X)
	for k in [1,3,5,50]:
		j = 0
		print("k = " + str(k))
		diff = D(X, trainData).eval()
		for row in diff:
			resp = find_knn(row, k, trainTarget)
			y[j] = np.dot(trainTarget.transpose(), resp)
			j += 1
		plt.figure(i)
		plt.plot(X, y, Data, Target, '.')	
		plt.title('k = ' + str(k))
		plt.axis([0,11,-2,11])
		plt.show()
		i += 1
	'''
	for k in [1,3,5,50]:
		print("K = " + str(k))
		i = 0
		loss = 0
		diff = D(trainData, trainData).eval()
		for row in diff:
			resp = find_knn(row, k, trainTarget)
			y_hat = np.dot(trainTarget.transpose(), resp)
			# print(y_hat)
			loss += (y_hat - trainTarget[i]) * (y_hat-trainTarget[i]) * 2 / trainTarget.shape[0]
			i += 1
		print("Training Loss = " + str(loss))

		i = 0
		loss = 0
		diff = D(validData, trainData).eval()
		#print(validData)
		#print(trainData)
		#print(diff)
		for row in diff:
			resp = find_knn(row, k, trainTarget)
			y_hat = np.dot(trainTarget.transpose(), resp)
			# print(y_hat)
			loss += (y_hat - validTarget[i]) * (y_hat-validTarget[i]) * 2 / validTarget.shape[0]
			i += 1
		print("Validation Loss = " + str(loss))

		i = 0
		loss = 0
		diff = D(testData, trainData).eval()
		for row in diff:
			resp = find_knn(row, k, trainTarget)
			y_hat = np.dot(trainTarget.transpose(), resp)
			# print(y_hat)
			loss += (y_hat - testTarget[i]) * (y_hat-testTarget[i]) * 2 / testTarget.shape[0]
			i += 1
		print("Test Loss = " + str(loss))
	'''
	
	#print(D(validData,trainData))
	#idx = tf.nn.top_k(-D(Data[randIdx[10:30]], Data[randIdx[:10]]), 5).indices
	#print(sess.run(idx))
	#print(trainTarget[:10])
	#print(sess.run(tf.gather(trainTarget[:10], idx)))
	
