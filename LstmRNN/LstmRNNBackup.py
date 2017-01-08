import time
t=time.time()
import tensorflow as tf
import numpy as np
from random import shuffle
from tqdm import tqdm
import sys
hidden_units=32
num_batch=25
num_steps=5
num_layers=4
num_epoch=10

data=open("tiny-shakespeare.txt", "r").read()
vocab=list(set(data))
vocab_size=len(vocab)
char_to_index=dict(zip(vocab, range(vocab_size)))
index_to_char=dict(zip(range(vocab_size), vocab))
data=[char_to_index[x] for x in data]

def build_graph(num_batch, num_steps, hidden_units, vocab_size):
	x=tf.placeholder(tf.int32, [num_batch, num_steps])
	y=tf.placeholder(tf.int32, [num_batch, num_steps])
	embeddings = tf.Variable(tf.random_uniform([vocab_size,hidden_units], -1.0, 1.0))
	rnn_inputs=tf.nn.embedding_lookup(embeddings, x)
	cell=tf.nn.rnn_cell.LSTMCell(hidden_units, state_is_tuple=True)
	cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=0.9, output_keep_prob=0.9)
	cell=tf.nn.rnn_cell.MultiRNNCell([cell]*num_layers, state_is_tuple=True)
	init_state=cell.zero_state(num_batch, tf.float32)
	val, final_state=tf.nn.dynamic_rnn(cell, rnn_inputs, dtype=tf.float32, initial_state=init_state)
	weight = tf.Variable(tf.truncated_normal([hidden_units, vocab_size], stddev=0.01))
	bias=tf.Variable(tf.truncated_normal([vocab_size], stddev=0.01))
	rnn_outputs=tf.reshape(val, [-1, hidden_units])
	logits=tf.matmul(rnn_outputs, weight)+bias
	predictions=tf.nn.softmax(logits)
	y_reshaped=tf.reshape(y, [-1])
	loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
	optimizer=tf.train.AdamOptimizer()
	minimizer=optimizer.minimize(loss)
	saver=tf.train.Saver()
	return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        loss = loss,
        minimizer = minimizer,
        predictions = predictions,
        saver=saver
    )

data=np.array(data)
seq_length=len(data)//num_batch
epoch_size=(seq_length-1)//num_steps
data=np.reshape(data[0:seq_length*num_batch], [num_batch, seq_length])

with tf.Graph().as_default():
	sess=tf.Session()
	g_train=build_graph(num_batch=num_batch, num_steps=num_steps,hidden_units=hidden_units, vocab_size=vocab_size)
	init=tf.initialize_all_variables()
	sess.run(init)
	for i in range(0):
		epoch_loss=0
		for j in tqdm(range(epoch_size)):
			x_input=data[:,j*num_steps:(j+1)*num_steps]
			y_input=data[:,j*num_steps+1:(j+1)*num_steps+1]
			_, loss=sess.run([g_train['minimizer'], g_train['loss']], feed_dict={g_train['x']:x_input, g_train['y']:y_input})
			epoch_loss+=loss
		print ("Epoch: ", i+1, "completed. Loss: ", epoch_loss)
	#g_train['saver'].save(sess, "charRNN.ckpt")
	sess.close()


#Sampling
with tf.Graph().as_default():
	Sample_len=200
	sess=tf.Session()
	#tf.reset_default_graph()
	g_test=build_graph(num_batch=1, num_steps=1,hidden_units=hidden_units, vocab_size=vocab_size)
	init=tf.initialize_all_variables()
	#sess.run(init)
	g_test['saver'].restore(sess, "/home/harshul/Documents/MachineLearning/tensorflow/LstmRNN/charRNN.ckpt")
	state=None
	ch=np.random.choice(range(vocab_size))
	ch=[[ch]]
	out_sample=[]
	for i in range(Sample_len):
		if state is not None:
			p, state=sess.run([g_test['predictions'], g_test['final_state']], feed_dict={g_test['x']:ch, g_test['init_state']:state})
		else:
			p, state=sess.run([g_test['predictions'], g_test['final_state']], feed_dict={g_test['x']:ch})
		ch=np.random.choice(vocab_size, p=p[0])
		out_sample.append(index_to_char[ch])
		ch=[[ch]]
	print (''.join(out_sample))
	sess.close()

print('time used = {0:.3f}'.format(time.time()-t))