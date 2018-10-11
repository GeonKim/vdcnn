
# coding: utf-8

# In[21]:
import os
os.getcwd()
os.chdir('/home/pirl/Downloads/공부')
##########################
# need to change to GPU
###########################
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')

import numpy as np
import tensorflow as tf
# Load data. Load your own data here
import csv
import numpy as np
samples = {}
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
FEATURE_LEN =1014
cdict = {}
for i,c in enumerate(ALPHABET):
    cdict[c] = i + 2
samples = {}
with open('news_data.csv') as f:
    reader = csv.DictReader(f,fieldnames=['class'],restkey='fields')
    for row in reader:
        label = row['class']
        if label not in samples:
            samples[label] = []
        sample = np.ones(FEATURE_LEN)
        count = 0
        for field in row['fields']:
            for char in field.lower():
                if char in cdict:
                    sample[count] = cdict[char]
                    count += 1
                if count >= FEATURE_LEN-1:
                    break
        samples[label].append(sample)
    samples_per_class = None
    classes = samples.keys()
    class_samples = []
    for c in classes:
        if samples_per_class is None:
            samples_per_class = len(samples[c])
        else:
            assert samples_per_class == len(samples[c])
        class_samples.append(samples[c])


# In[22]:


def build_onehot(input_):
    target = np.zeros(len(classes))
    target[input_] = 1
    return target
y= []
y_ = []
for i in range(len(classes)):
    for j in range(samples_per_class):
        target =build_onehot(i)
        y.append(target)
        y_.append(i)
y_ = np.array(y_)


# In[23]:


x = np.reshape(class_samples,(-1,FEATURE_LEN))
y = np.reshape(y,(-1,len(classes)))


# In[31]:


print("\nEvaluating...\n")
# Evaluation
# ==================================================
checkpoint_dir = '/home/pirl/Downloads/공부/runs_new/1539196181/checkpoints'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
print("checkpoint file: {}".format(checkpoint_file))


# In[25]:


import os
import time
import datetime
def batch_iter(x, y, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    # data = np.array(data)
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    print("num batches per epoch is: " + str(num_batches_per_epoch))
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
    else:
        x_shuffled = x
        y_shuffled = y
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        x_batch = x_shuffled[start_index:end_index]
        y_batch = y_shuffled[start_index:end_index]
        batch = list(zip(x_batch, y_batch))
        yield batch


# In[26]:


### test with meta graph 

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name(
            "dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name(
            "loss/predictions").outputs[0]

        # Generate batches for one epoch
       
        batches = batch_iter(x, y,128, 1)

        # Collect the predictions here
        all_predictions = []

        for test_batch in batches:
            x_batch, y_batch = zip(*test_batch)
            batch_predictions = sess.run(
                predictions, {input_x: x_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate(
                [all_predictions, batch_predictions])

# Print accuracy
correct_predictions = float(sum(np.equal(all_predictions,y_)))
print("Total number of test examples: {}".format(len(y_)))
print("Accuracy: {:g}".format(correct_predictions/float(len(y_))))

