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

hyperparameters = {}
hyperparameters['lambda'] = 0.001
hyperparameters['mini_batch_size'] = 500
hyperparameters['num_epochs'] = 50
hyperparameters['eta'] = 1

W = tf.Variable(tf.zeros([28**2,1], tf.float32))
b = tf.Variable(tf.zeros([1,1], tf.float32))

trainData = np.reshape(trainData, [3500, 28**2])
validData = np.reshape(validData, [100, 28**2])

x = tf.placeholder(tf.float32, [None, 784])
x__ = x
y = tf.placeholder(tf.float32, [None,1])
learning_rate = tf.placeholder(tf.float32, [])
hp_mbs = tf.placeholder(tf.float32, [])
hp_lambda = tf.placeholder(tf.float32, [])

pred = tf.matmul(x__, W) + b
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y)) + hp_lambda * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b)) / 2
accuracy = tf.reduce_mean((1.0-tf.cast(pred>=0.5, tf.float32))*(1.0-y) + tf.cast(pred>=0.5, tf.float32)*y)
adam_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	init_op = tf.initialize_all_variables()
	sess.run(init_op)
	for i in range(hyperparameters['num_epochs']):
		randIndx = np.arange(3500)
		np.random.shuffle(randIndx)
	
		for j in range(3500/hyperparameters['mini_batch_size']):
			miniData = trainData[randIndx[j*hyperparameters['mini_batch_size']:(j+1)*hyperparameters['mini_batch_size']]]
			miniTarget = trainTarget[randIndx[j*hyperparameters['mini_batch_size']:(j+1)*hyperparameters['mini_batch_size']]]

			# Adam Optimizer

			_, c = sess.run([adam_op, loss], feed_dict={x:miniData, y:miniTarget, learning_rate:hyperparameters['eta'], 
				hp_lambda:hyperparameters['lambda']})
		
		validError = sess.run(loss, feed_dict={x:validData, y:validTarget, hp_lambda:hyperparameters['lambda']})
		trainError = sess.run(loss, feed_dict={x:trainData, y:trainTarget, hp_lambda:hyperparameters['lambda']})

		validAccuracy = sess.run(accuracy, feed_dict={x:validData, y:validTarget, hp_lambda:hyperparameters['lambda']})
		trainAccuracy = sess.run(accuracy, feed_dict={x:trainData, y:trainTarget, hp_lambda:hyperparameters['lambda']})

		print "\n--- Epoch %d ---" % (i)
		print "Valid Error: ", validError
		print "Training Error: ", trainError
		print "Valid Accuracy: ", validAccuracy
		print "training Accuracy: ", trainAccuracy

