import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

with np.load("notMNIST.npz") as data :
	Data, Target = data ["images"], data["labels"]
	posClass = 2
	negClass = 9
	dataIndx = (Target==posClass) + (Target==negClass)
	Data = Data[dataIndx]/255.
	Target = Target[dataIndx].reshape(-1, 1)
	Target[Target==posClass] = 1
	Target[Target==negClass] = 0
	np.random.seed(521)
	randIndx = np.arange(len(Data))
	np.random.shuffle(randIndx)
	Data, Target = Data[randIndx], Target[randIndx]
	trainData, trainTarget = Data[:3500], Target[:3500]
	validData, validTarget = Data[3500:3600], Target[3500:3600]
	testData, testTarget = Data[3600:], Target[3600:]


def SGD_Optimizer (x, t, W, hp_lambda, hp_eta, hp_mbs):
	X = tf.matmul(tf.cast(x, tf.float32),W)
	L = tf.matmul(tf.cast(x, tf.float32), tf.sigmoid(X) - t, True, False) / hp_mbs + hp_lambda * tf.abs(W) / hp_mbs
	W = W - hp_eta * L
	return W

hyperparameters = {}
hyperparameters['lambda'] = 0.01
hyperparameters['mini_batch_size'] = 500
hyperparameters['num_epochs'] = 50
hyperparameters['eta'] = 1

W = tf.Variable(tf.random_uniform([28**2,1]))
# b = tf.Variable(tf.zeros([1,1], tf.float64))

# print(type(trainTarget[0,0]))
# print(type(X[0,0]))

# print L_D
trainData = np.reshape(trainData, [3500, 28**2])
validData = np.reshape(validData, [100, 28**2])

x = tf.placeholder(tf.float32, [None, 784])
x__ = x
y = tf.placeholder(tf.float32, [None,1])
learning_rate = tf.placeholder(tf.float32, [])
hp_mbs = tf.placeholder(tf.float32, [])
hp_lambda = tf.placeholder(tf.float32, [])

pred = tf.matmul(x__, W)
dL_dW = tf.matmul(x__, tf.sigmoid(pred) - y, True, False) / hp_mbs + hp_lambda * tf.abs(W) / hp_mbs

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y)) + hp_lambda * tf.matmul(W,W,True,False) / 2

adam_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	init_op = tf.initialize_all_variables()
	sess.run(init_op)
	for i in range(hyperparameters['num_epochs']):
		randIndx = np.arange(3500)
		np.random.shuffle(randIndx)
	
		for j in range(3500/hyperparameters['mini_batch_size']):
	 		# print "Epoch: %d, Batch: %d" % (i, j)
			# print(W)
			# print(trainData)
			miniData = trainData[randIndx[j*hyperparameters['mini_batch_size']:(j+1)*hyperparameters['mini_batch_size']]]
			miniTarget = trainTarget[randIndx[j*hyperparameters['mini_batch_size']:(j+1)*hyperparameters['mini_batch_size']]]
			# print miniData
			# print miniTarget
			# L_D = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(miniTarget, tf.matmul(miniData, W))) / 500
			# L_W = hyperparameters['lambda'] * tf.matmul(W,W,True,False) / 2
	
			# loss = L_D + L_W
			W = SGD_Optimizer (miniData, miniTarget, W, hyperparameters['lambda'], hyperparameters['eta'], hyperparameters['mini_batch_size'])
			# W = Adam_Optimizer (miniData, miniTarget, W, hyperparameters['lambda'], hyperparameters['eta'], hyperparameters['mini_batch_size'], L_D + L_W)
			
			# print miniData
			# X = tf.matmul(miniData,W) + b
			# print 'X', X
	
			# print 'miniData', miniData
			# dL_dW = tf.matmul(miniData, tf.sigmoid(X) - miniTarget, True, False) / 500 + hyperparameters['lambda'] * tf.abs(W) / 500
			# print dL_dW

			# Adam Optimizer

			# _, c = sess.run([adam_op, loss], feed_dict={x:miniData, y:miniTarget, learning_rate:hyperparameters['eta'], 
			# 	hp_lambda:hyperparameters['lambda']})

			# SGD
			
			# dW, c = sess.run([dL_dW, loss], feed_dict={x:miniData, y:miniTarget, learning_rate:hyperparameters['eta'], 
			# 	hp_lambda:hyperparameters['lambda'], hp_mbs:hyperparameters['mini_batch_size']})
			
			# print c
			# print sess.run(tf.transpose(W))
			# W = W - dW *hyperparameters['eta']
			# print tf.trainable_variables()
			# print c
		
		# validError = sess.run(loss, feed_dict={x:validData, y:validTarget, hp_lambda:hyperparameters['lambda']})
		# trainError = sess.run(loss, feed_dict={x:trainData, y:trainTarget, hp_lambda:hyperparameters['lambda']})
		X = tf.matmul(tf.cast(validData, tf.float32),W)
		validError = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(X, tf.cast(validTarget, tf.float32))) + hyperparameters['lambda'] * tf.matmul(W,W,True,False) / 2
		y_hat = tf.cast(tf.sigmoid(X) >= 0.5, tf.int32)
		validAcc = tf.matmul((1-y_hat), tf.cast(1-validTarget, tf.int32), True, False) + tf.matmul(y_hat,tf.cast(validTarget, tf.int32), True, False)
	
		with tf.Session() as sess:
			sess.run(init_op)
	#		print "Valid Error (Epoch %d):" % i, validError
			print "Valid Error: ", sess.run(validError)
			print "Valid Accuracy: ", sess.run(tf.cast(validAcc, tf.float64)/100)

