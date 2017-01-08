'''
datacreated time used = 2.309
Graph Created time used = 0.392
Epoch:  1 Completed. Accuracy:  0.595894607843
Epoch:  2 Completed. Accuracy:  0.833946078431
Epoch:  3 Completed. Accuracy:  0.894301470588
Epoch:  4 Completed. Accuracy:  0.927083333333
Epoch:  5 Completed. Accuracy:  0.963541666667
Epoch:  6 Completed. Accuracy:  0.980085784314
Epoch:  7 Completed. Accuracy:  0.980698529412
Epoch:  8 Completed. Accuracy:  0.98743872549
Epoch:  9 Completed. Accuracy:  0.988051470588
Epoch:  10 Completed. Accuracy:  0.989583333333
time used = 110.702
[Finished in 113.7s]
'''


import time
t=time.time()
import tensorflow as tf
import numpy as np
from random import shuffle
from tqdm import tqdm

#Generating Train Data
train_x=['{0:015b}'.format(i) for i in range(2**15)]
to=[]
shuffle(train_x)
ti=[]
for i in train_x:
	tl=list(i)
	tl=[[i] for i in tl]
	tp=sum([int(x[0]) for x in tl])
	ti.append(np.array(tl))
	tl=[0]*16
	tl[tp]=1
	to.append(tl)
train_x=np.array(ti)
train_y=np.array(to)
test_x=train_x[int(0.9*(len(train_x))):]
test_y=train_y[int(0.9*(len(train_x))):]
train_y=train_y[:int(0.9*(len(train_x)))]
train_x=train_x[:int(0.9*(len(train_x)))]


batch_size=32
num_batch=len(train_x)//batch_size
num_epoch=10
num_test_size=len(test_x)//batch_size

print("datacreated",'time used = {0:.3f}'.format(time.time()-t))
t=time.time()
#Model

data=tf.placeholder(tf.float32, [None, 15,1])
target=tf.placeholder(tf.float32, [None, 16])
hidden_units=32
cell=tf.nn.rnn_cell.LSTMCell(hidden_units, state_is_tuple=True)
#cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
#cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 4, state_is_tuple=True)
init_state=cell.zero_state(batch_size, tf.float32)
val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val=tf.transpose(val,perm=[1,0,2])
#print ([int(x) for x in val.get_shape()])
final_output_for_every_number_batch=tf.gather(val, (val.get_shape()[0])-1)
#print ([int(x) for x in final_output_for_every_number_batch.get_shape()])
weight = tf.Variable(tf.truncated_normal([hidden_units, int(target.get_shape()[1])], stddev=0.01))
bias = tf.Variable(tf.truncated_normal([int(target.get_shape()[1])], stddev=0.01))
prediction=tf.matmul(final_output_for_every_number_batch, weight)+bias
#print ([int(x) for x in prediction.get_shape()])
#print (int(target.get_shape()[1]))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,target))
optimizer=tf.train.AdamOptimizer()
minimize=optimizer.minimize(loss)
accurate=tf.equal(tf.argmax(prediction, 1), tf.argmax(target, 1))
accuracy=tf.reduce_mean(tf.cast(accurate, tf.float32))

print("Graph Created",'time used = {0:.3f}'.format(time.time()-t))
t=time.time()

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

for i in range(num_epoch):
	ptr=0
	for j in range(num_batch):
		x_input=train_x[ptr:ptr+batch_size]
		y_input=train_y[ptr:ptr+batch_size]
		ptr+=batch_size
		sess.run(minimize, feed_dict={data:x_input, target:y_input})
	p=0
	ptr=0
	for j in range(num_test_size):
		x_input=test_x[ptr:ptr+batch_size]
		y_input=test_y[ptr:ptr+batch_size]
		ptr+=batch_size
		q=sess.run(accuracy, feed_dict={data:x_input, target:y_input})
		p+=q
	print ("Epoch: ", i+1, "Completed. Accuracy: ", p/(num_test_size))
	#print ("Epoch: ", i+1, "Completed. Accuracy: ", sess.run(accuracy, feed_dict={data:test_x, target:test_y}))
sess.close()

print('time used = {0:.3f}'.format(time.time()-t))