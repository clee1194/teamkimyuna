import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

with np.load ("tinymnist.npz") as data :
	trainData, trainTarget = data ["x"], data["y"]
	validData, validTarget = data ["x_valid"], data ["y_valid"]
	testData, testTarget = data ["x_test"], data ["y_test"]

np.random.seed(521)

# print(trainData.shape)

W = tf.cast(tf.Variable(tf.zeros([64, 1])), tf.float64)
b = tf.cast(tf.Variable(tf.zeros([1, 1])), tf.float64)

W_0 = W
b_0 = b

mini_batch_size = 50
weight_decay_coeff = 1
learning_rate = 0.003

iterations_per_batch = trainData.shape[0] / mini_batch_size
# print(trainData.size)

#initialize the variable
init_op = tf.initialize_all_variables()
trainingPlot = {}
with tf.Session() as sess:
	for weight_decay_coeff in [0, 0.0001, 0.001, 0.01, 0.1]:
	#for learning_rate in [0.002, 0.003, 0.005, 0.0075]:
		print("REGULARIZATION = " + str(weight_decay_coeff))
		print("LEARNING_RATE = " + str(learning_rate))
		trainingPlot[learning_rate] = []
		for i in range(50):
			randIdx = np.arange(700)
			np.random.shuffle(randIdx)
			# print("Batch " + str(i))
			for j in range(iterations_per_batch):
				miniBatch = trainData[randIdx[j*mini_batch_size:(j+1)*mini_batch_size]]# trainData[j*mini_batch_size:(j+1)*mini_batch_size]
				miniBatch_target = trainTarget[randIdx[j*mini_batch_size:(j+1)*mini_batch_size]]#trainTarget[j*mini_batch_size:(j+1)*mini_batch_size]
				#print(miniBatch)
				# print(miniBatch_target)
				#print(tf.matmul(W,miniBatch, True, True))
				#print(miniBatch_target)
				trainingY = tf.matmul(W,miniBatch, True, True) + b
				trainingDiff = trainingY - tf.transpose(miniBatch_target)
				dL_dW = tf.matmul(trainingDiff, miniBatch) + weight_decay_coeff * tf.transpose(W)
				db_dW = tf.matmul(trainingDiff, tf.ones([1,50], tf.float64), False, True)
				# print(dL_dW)
				# print(db_dW)
				W = W - tf.transpose(dL_dW) * learning_rate
				b = b - tf.transpose(db_dW) * learning_rate
			validY = tf.matmul(W,validData,True, True) + b
			#print(validY)
			#print(validTarget.shape)
			validDiff = validY - tf.transpose(validTarget)
			validError = tf.matmul(validDiff,validDiff, False, True) 
			testY = tf.matmul(W,testData,True, True) + b
			testDiff = testY - tf.transpose(testTarget)
			testError = tf.matmul(testDiff, testDiff, False, True)
		sess.run(init_op)
		print(sess.run(validError))
		print(sess.run(testError))
		W = W_0
		b = b_0
		#print 
			
			# a = tf.Print(
