# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 09:23:23 2017

@author: jiang_y
"""
#######################################################################
######################### Tensor Operation ############################
#######################################################################
# convert numpy array to tensor or vice versa
a1 = tf.convert_to_tensor(a)
with tf.Session() as sess:
    a=a1.eval()

# slice
tf.slice(a,[1,0,0,0],[1,32,32,3])       # tf.slice(input_, begin, size, name=None)

# reshape



# soft max entropy functions and input shape
# ref: https://www.tensorflow.org/api_docs/python/nn/classification?authuser=2

# logit.shape = [batch_size, num_classes]; label.shape = [batch_size, num_classes] (one-hot vector)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(soft_max_linear,labels)
loss = tf.reduce_mean(cross_entropy)

# # logit.shape = [batch_size, num_classes]; label.shape = [batch_size] (one-hot vector)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(soft_max_linear,labels)
loss = tf.reduce_mean(cross_entropy)


#######################################################################
########################### Build Graph ###############################
#######################################################################

with tf.Graph().as_default():
    
    ...
    
    with tf.name_scope('hidden') as scope:
      a = tf.constant(5, name='alpha')
      W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
      b = tf.Variable(tf.zeros([1]), name='biases')
      
    ...
    
    merged = tf.summary.merge_all()
    
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('/tmp/cifar10_my_ex/train_try',sess.graph)
        test_writer = tf.summary.FileWriter('/tmp/cifar10_my_ex/test_try')
        
        for i in range(300000):
            _ = sess.run([train_step],feed_dict={x:x,y:y})
                if (i%100==0): 
                    train_writer.add_summary(summary,i)


#######################################################################
########################### Add Summary ###############################
#######################################################################
sess = tf.InteractiveSession()
with tf.name_scope(layer_name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    tf.summary.histogram('histogram', var)
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('tmp/train',sess.graph)
test_writer = tf.summary.FileWriter('tmp/test')
tf.global_variables_initializer().run()


for i in range(n):
  if i % 10 == 0:  # Record summaries and test-set accuracy
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # Record train set summaries, and train
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()


#######################################################################
########################## Save Checkpoint ############################
#######################################################################
# ref: https://www.tensorflow.org/how_tos/variables/
# script to inspect checkpoint file: https://www.tensorflow.org/code/tensorflow/python/tools/inspect_checkpoint.py
   
###################### Saving check point file ########################
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  ..
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in file: %s" % save_path)
  
  
###################### Restoring check point file ########################
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model