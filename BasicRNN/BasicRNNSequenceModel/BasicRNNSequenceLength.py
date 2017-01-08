import time
t=time.time()
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm

text=open('text.txt', 'r').read()
uniqueChars=list(set(text))
text_size, vocab_size=len(text), len(uniqueChars)
char_to_index={ ch:i for i,ch in enumerate(uniqueChars)}
index_to_char={ i:ch for i,ch in enumerate(uniqueChars)}

hidden_size = 100
totalIteration=5000
seq_length=100

def one_hot(v):
	return np.eye(vocab_size)[v]

x=tf.placeholder(tf.float32, [None, vocab_size])
y_in=tf.placeholder(tf.float32, [None, vocab_size])
hStart=tf.placeholder(tf.float32, [1, hidden_size])

Wxh=tf.Variable(tf.random_normal([vocab_size, hidden_size], stddev=0.01))
Whh=tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.01))
Bh=tf.Variable(tf.random_normal([hidden_size], stddev=0.01))
Why=tf.Variable(tf.random_normal([hidden_size, vocab_size], stddev=0.01))
By=tf.Variable(tf.random_normal([vocab_size], stddev=0.01))


hState=hStart
y_outAll=[]
for i, q in enumerate(tf.split(0, seq_length, x)):
	if i==0:
		y_outAll=[]
	hState=tf.tanh(tf.matmul(q, Wxh)+tf.matmul(hState, Whh)+Bh)
	y_out=tf.matmul(hState, Why)+By
	y_outAll.append(y_out)

hLast=hState
output_softmax=tf.nn.softmax(y_outAll[-1])
output_softmax1=tf.argmax(output_softmax, 1)
outputs=tf.concat(0, y_outAll)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, y_in))

minimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
'''
grads_and_vars = minimizer.compute_gradients(loss)
grad_clipping = tf.constant(5.0, name="grad_clipping")
clipped_grads_and_vars = []
for grad, var in grads_and_vars:
	clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
	clipped_grads_and_vars.append((clipped_grad, var))

updates = minimizer.apply_gradients(clipped_grads_and_vars)
'''
def sampleNetwork():
	sample_length = 200
	start_ix      = random.randint(0, len(text) - seq_length)
	sample_seq_ix = np.array([char_to_index[ch] for ch in text[start_ix:start_ix+seq_length]])
	sample_prev_state_val=np.zeros([1, hidden_size])
	ixes=[]
	for t in range(sample_length):
		sample_input_vals = [one_hot(k) for k in sample_seq_ix]
		ix, sample_prev_state_val = sess.run([output_softmax, hLast], feed_dict={x: sample_input_vals, hStart: sample_prev_state_val})
		ix = np.random.choice(range(vocab_size), p=ix.ravel())
		ixes.append(ix)
		sample_seq_ix = np.array(np.concatenate((sample_seq_ix[1:] , [ix]), 0))
	txt = ''.join(index_to_char[ix] for ix in ixes)
	print('----\n %s \n----\n' % (txt,))

saver=tf.train.Saver()
sess=tf.Session()
init=tf.initialize_all_variables()
hStart_val=np.zeros([1, hidden_size])
#sess.run(init)
saver.restore(sess, "/home/harshul/Documents/MachineLearning/tensorflow/BasicRNN/mymodelcharRNNSequence.ckpt")
positionInText=0
z=len(text)
for i in range(totalIteration):
	t2=time.time()
	hStart_val=np.zeros([1, hidden_size])
	positionInText=0
	
	for positionInText in tqdm(range(z-1)[::seq_length]):
		if positionInText+seq_length+1>=z:
			break
		inputs=one_hot([char_to_index[ch] for ch in text[positionInText:positionInText+seq_length]])
		inputs=np.array(inputs)
		targets=one_hot([char_to_index[ch] for ch in text[positionInText+1:positionInText+seq_length+1]])
		targets=np.array(targets)
		_, hStart_val, loss_val= sess.run([minimizer, hLast, loss], feed_dict={x: inputs, y_in: targets, hStart:hStart_val})
		'''
		if positionInText%10000==0:
			print ("completed: ", positionInText, "of ", z, "loss: ", loss_val)
			sampleNetwork()
		'''
	print ("Iteration ", i+1, " completed")
	print('time used = {0:.3f}'.format(time.time()-t2))
	sampleNetwork()
	saver.save(sess, "/home/harshul/Documents/MachineLearning/tensorflow/BasicRNN/mymodelcharRNNSequence.ckpt")
print('time used = {0:.3f}'.format(time.time()-t))