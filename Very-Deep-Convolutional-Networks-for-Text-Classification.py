
# coding: utf-8
import os
os.chdir('/home/pirl/Downloads/AI_Project_Group2/vdcnn')
import time
import datetime
import tflearn
import tensorflow as tf
import numpy as np
from tensorflow import flags
import cnn_tool as tool
from konlpy.tag import Kkma

#samples = {}
#ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
#FEATURE_LEN =800
#cdict = {}
#for i,c in enumerate(ALPHABET):
#    cdict[c] = i + 2

#%%
#samples = {}
#
#with open('news_data.csv') as f:
#    reader = csv.DictReader(f,fieldnames=['class'],restkey='fields')
#    for row in reader:
#        label = row['class']
#        if label not in samples:
#            samples[label] = []
#        sample = np.ones(FEATURE_LEN)
#        count = 0
#        for field in row['fields']:
#            for char in field.lower():
#                if char in cdict:
#                    sample[count] = cdict[char]
#                    count += 1
#                if count >= FEATURE_LEN-1:
#                    break
#        samples[label].append(sample)
#    samples_per_class = None
#    classes = samples.keys()
#    class_samples = []
#
#    for c in classes:
#        if samples_per_class is None:
#            samples_per_class = len(samples[c])
#        else:
#            assert samples_per_class == len(samples[c])
#        class_samples.append(samples[c])
#
#def build_onehot(input_):
#    target = np.zeros(len(classes))
#    target[input_] = 1
#    return target
#
#y= []
#for i in range(len(classes)):
#    for j in range(samples_per_class):
#        target =build_onehot(i)
#        y.append(target)
#

#%%
#data loading
data_path = 'posco_news_data.csv' # csv 파일로 불러오기
#contents는 각 기사 스트링으로 바꿔 리스트에 넣은거, points는 클래스 0or 1
contents, points = tool.loading_rdata(data_path)
contents = tool.cut(contents)
#불용어 처리
kkma = Kkma()
stop_words = ['NNG', 'NNP', 'NNB', 'NR', 'NP', 'VV','VA','VX','VCP', 'VCN','MM','MAG','MAJ','XR']
test = []
test2 = []
def remover(doc):
    text = ''.join(c for c in doc if c.isalnum() or c in '+, ')
    text = ''.join([i for i in text if not i.isdigit()])
    return text
def tokenize(doc):
    return [t[0] for t in kkma.pos(remover(doc)) if t[1] in stop_words]

for i in range(len(contents)):
    test.append(tokenize(contents[i]))
    test2.append((' ').join(test[i]))
    if i%10==0:
        print("%d개의 기사 중 %d번째 기사 불용어 처리 중임돠~ ^오^"%(len(contents),len(test)))
contents = test2

#단어 to Vector로.
max_document_length = 800
x, vocabulary, vocab_size = tool.make_input(contents, max_document_length)
print('사전단어수 : %s' %(vocab_size))
y = tool.make_output(points, threshold = 0)

#tool.save_vocab('news_vocab.txt', contents, max_document_length)

 # divide dataset into train/test set
x_train, x_dev, y_train, y_dev = tool.divide(x, y, train_prop = 0.8)
#%%

#x = np.reshape(class_samples,(-1,FEATURE_LEN))
#y = np.reshape(y,(-1,len(classes)))


# In[ ]:
#
#
##np.random.seed(10)
#shuffle_indices = np.random.permutation(np.arange(len(y)))
#shuffle_indices
#x_shuffled = x[shuffle_indices]
#len(x_shuffled)
#y_shuffled = y[shuffle_indices]
#len(x_shuffled)
## Split train/test set
#x_train, x_test = x_shuffled[:int(len(x_shuffled)*0.8)], x_shuffled[int(len(x_shuffled)*0.8):]
#y_train, y_test = y_shuffled[:int(len(x_shuffled)*0.8)], y_shuffled[int(len(x_shuffled)*0.8):]
#print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))


# In[ ]:


def Convolutional_Block(input_, filter_num, train, scope):
    
    norm = tf.random_normal_initializer(stddev=0.05)#표.편 0.05로 정규분포에서 초기화
    filter_shape1 = [3, 1, input_.get_shape()[3], filter_num]
    with tf.variable_scope(scope):
        filter_1 = tf.get_variable('filter1', filter_shape1, initializer=norm)
        conv1 = tf.nn.conv2d(input_, filter_1, strides=[1, 1, filter_shape1[1], 1], padding="SAME")
        batch_normal1 =tflearn.layers.normalization.batch_normalization(conv1,trainable= train,scope = scope+"BN1")
        filter_shape2 = [3, 1, batch_normal1.get_shape()[3], filter_num]
        filter_2 = tf.get_variable('filter2', filter_shape2, initializer=norm)
        conv2 = tf.nn.conv2d(tf.nn.relu(batch_normal1), filter_2, strides=[1, 1, filter_shape2[1], 1], padding="SAME")
        batch_normal2 =tflearn.layers.normalization.batch_normalization(conv2,trainable=train,scope = scope+"BN2")
        pooled = tf.nn.max_pool(tf.nn.relu(batch_normal2),ksize=[1, 3, 1, 1],strides=[1, 2, 1, 1],padding='SAME',name="pool1")
        return pooled
    
def Conv(input_,filter_shape,strides,train,scope):
    norm = tf.random_normal_initializer(stddev=0.05)
    with tf.variable_scope(scope):
        filter_1 = tf.get_variable('filter1', filter_shape, initializer=norm)
        conv = tf.nn.conv2d(input_, filter_1, strides=strides, padding="SAME")
        batch_normal =tflearn.layers.normalization.batch_normalization(conv,trainable=train,scope = scope+"BN")
        return batch_normal
    
def linear(input, output_dim, scope=None, stddev=0.1):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        l2_loss = tf.nn.l2_loss(w)+tf.nn.l2_loss(b)
        return tf.matmul(input, w) + b,l2_loss
class CharCNN(object):
    """
    A CNN for text classification.
    based on the Very Deep Convolutional Networks for Natural Language Processing.
    """
    def __init__(self, num_classes=2, filter_size=3,
                 l2_reg_lambda=0.001, sequence_max_length=800, num_quantized_chars=71,embedding_size=16):
        '''
        num_classes : 클래스 갯수 (우린 두개)
        filter_size : 
        l2_reg_lambda : 
        sequence_max_length : 
        num_quantized_chars : 
        embedding_size : 
        '''
        
        self.input_x = tf.placeholder(tf.int32, [None,sequence_max_length], name="input_x") #dict 갯수니깐 int
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")#1.0 or 0.0
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") 
        self.training =  tf.placeholder(tf.int32, name="trainable")
        #컨볼루션 실행 여부 여기서 결정
        if self.training==1:
            TRAIN = True
        else:
            TRAIN = False
            
        l2_loss = tf.constant(0.0)
        with tf.device('/gpu:0'),tf.name_scope("embedding"):
            W0 = tf.Variable(tf.random_uniform([num_quantized_chars, embedding_size], -1.0, 1.0),name="W")
            self.embedded_characters = tf.nn.embedding_lookup(W0,self.input_x)
            self.embedded_characters_expanded = tf.expand_dims(self.embedded_characters,-1,name="embedding_input")
        with tf.name_scope("layer-0"): 
            filter_shape0 = [3, embedding_size, 1, 64]
            strides0 =[1, 1,embedding_size, 1]
            self.h0 = Conv(self.embedded_characters_expanded,filter_shape0,strides0,TRAIN,'layer_0')
        with tf.name_scope("layer-1-2-3-4-5-6-7-8"):
            self.h1 = Convolutional_Block(self.h0,64,TRAIN,'layer_1-2')
            self.h2 = Convolutional_Block(self.h1,128,TRAIN,'layer_3-4')
            self.h3 = Convolutional_Block(self.h2,256,TRAIN,'layer_5-6')
            self.h4 = Convolutional_Block(self.h3,512,TRAIN,'layer_7-8')
            self.h5 = tf.transpose(self.h4,[0,3,2,1])
            self.pooled = tf.nn.top_k(self.h5, k=8,name='k-maxpooling') 
            self.h6 = tf.reshape(self.pooled[0],(-1,512*8))
        with tf.name_scope("fc-1-2-3"):
            self.fc1_out,fc1_loss = linear(self.h6, 2048, scope='fc1', stddev=0.1)
            l2_loss += fc1_loss
            self.fc2_out,fc2_loss = linear(tf.nn.relu(self.fc1_out), 2048, scope='fc2', stddev=0.1)
            l2_loss += fc2_loss
            self.fc3_out,fc3_loss = linear(tf.nn.relu(self.fc2_out), num_classes, scope='fc3', stddev=0.1)
            l2_loss += fc3_loss
        with tf.name_scope("loss"):
            self.predictions = tf.argmax(self.fc3_out, 1, name="predictions")
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.fc3_out,labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

x_batch = x_train[0:64]
y_batch = y_train[0:64]


os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(gpu_options = gpu_options,
                                  allow_soft_placement=True,
                                  log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CharCNN()
        sess.run(tf.initialize_all_variables())
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.training:1,
            cnn.dropout_keep_prob: 1.0
        }       
        out = sess.run(cnn.accuracy,feed_dict)
def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    # data = np.array(data)
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        print("In epoch >> " + str(epoch + 1))
        print("num batches per epoch is: " + str(num_batches_per_epoch))
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            x_shuffled = x
            y_shuffled = y
        for batch_num in range(num_batches_per_epoch-1):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            x_batch = x_shuffled[start_index:end_index]
            y_batch = y_shuffled[start_index:end_index]
            batch = list(zip(x_batch, y_batch))
            yield batch
batch_size = 100
num_epochs = 40
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CharCNN()
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(
            os.path.curdir, "runs_new", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already
        # exists, so we need to create it.
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.initializers.global_variables())
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.training:1,
              cnn.dropout_keep_prob: 1.0
            }       
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op,
                    cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # write fewer training summaries, to keep events file from
            # growing so big.
            if step % (evaluate_every / 2) == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(
                    time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
        def dev_step(x_batch, y_batch, writer=None):
            dev_size = len(x_batch)
            max_batch_size = 500
            num_batches = dev_size/max_batch_size
            acc = []
            losses = []
            print("Number of batches in dev set is " + str(num_batches))
            for i in range(int(num_batches)):
                x_batch_dev = x_batch[i * max_batch_size:(i + 1) * max_batch_size]
                y_batch_dev = y_batch[i * max_batch_size: (i + 1) * max_batch_size]
                feed_dict = {
                  cnn.input_x: x_batch_dev,
                  cnn.input_y: y_batch_dev,
                  cnn.training:0,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                acc.append(accuracy)
                losses.append(loss)
                time_str = datetime.datetime.now().isoformat()
                print("batch " + str(i + 1) + " in dev >>" +" {}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            print("\nMean accuracy=" + str(sum(acc)/len(acc)))
            print("Mean loss=" + str(sum(losses)/len(losses)))
        # just for epoch counting
        num_batches_per_epoch = int(len(x_train)/batch_size) + 1
        # Generate batches
        batches = batch_iter(x_train, y_train,batch_size, num_epochs)
        evaluate_every = 300
        checkpoint_every =300
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch) 
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_test, y_test, writer=dev_summary_writer)
                print("Epoch: {}".format(
                    int(current_step / num_batches_per_epoch)))
                print("")
            if current_step % checkpoint_every == 0:
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                    


#                print("Saved model checkpoint to {}\n".format(path))

