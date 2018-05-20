import pickle
import numpy as np
import tensorflow as tf
from sklearn import preprocessing as pp

#官方提供函数打开文件
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def clean(data):
    #normalized = pp.scale(data,axis=1) #暂时只做个标准化
    imgs = data.reshape(data.shape[0], 3, 32, 32)
    grayscale_imgs = imgs.mean(1)
    cropped_imgs = grayscale_imgs[:, 4:28, 4:28]
    img_data = cropped_imgs.reshape(data.shape[0], -1)
    '''
    img_size = np.shape(img_data)[1]
    means = np.mean(img_data, axis=1)
    meansT = means.reshape(len(means), 1)
    stds = np.std(img_data, axis=1)
    stdsT = stds.reshape(len(stds), 1)
    adj_stds = np.maximum(stdsT, 1.0 / np.sqrt(img_size))
    normalized = (img_data - meansT) / adj_stds
    '''
    normalized = pp.scale(img_data,axis=1) #暂时只做个标准化
    return normalized


#读取数据并组合
def read_data(directory):
    names = unpickle('{}/batches.meta'.format(directory))['label_names']
    print('names', names)

    data, labels = [], []
    for i in range(1, 6):
        filename = '{}/data_batch_{}'.format(directory, i)
        batch_data = unpickle(filename)
        if len(data) > 0:
            data = np.vstack((data, batch_data['data']))
            labels = np.hstack((labels, batch_data['labels']))
        else:
            data = batch_data['data']
            labels = batch_data['labels']
    test_file_name = '{}/test_batch'.format(directory)
    test = unpickle(test_file_name)
    test_data = test['data']
    test_labels = test['labels']
    data = clean(data)
    test_data = clean(test_data)
    data = data[0:data.shape[0] - 16, :]
    print(np.shape(data), np.shape(labels),np.shape(test_data),np.shape(test_labels))
    data = data.astype(np.float32)
    return names, data, labels,test_data,test_labels

names, data, labels ,test_data,test_labels= read_data('./data/cifar-10-batches-py')

train_flag = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, [64, 24 * 24])
y = tf.placeholder(tf.float32, [None, len(names)])
W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
b1 = tf.Variable(tf.random_normal([64]))
W2 = tf.Variable(tf.random_normal([5, 5, 64, 64]))
b2 = tf.Variable(tf.random_normal([64]))
W3 = tf.Variable(tf.random_normal([6*6*64, 1024]))
b3 = tf.Variable(tf.random_normal([1024]))
W_out = tf.Variable(tf.random_normal([1024, len(names)]))
b_out = tf.Variable(tf.random_normal([len(names)]))


#卷积操作封一层，方便传参数
def conv_layer(X, W, b):
    conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_with_b = tf.nn.bias_add(conv, b)
    return conv_with_b
#池化操作封一层，方便传参数
def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def model(train_flag):
    x_reshaped = tf.reshape(x, shape=[-1, 24, 24, 1])
    conv_out1 = conv_layer(x_reshaped, W1, b1)
    relu_out1 = tf.nn.relu(conv_out1)
    x_norm1 = tf.layers.batch_normalization(relu_out1,axis=3, training=train_flag)
    maxpool_out1 = max_pool_2x2(x_norm1)
    conv_out2 = conv_layer(maxpool_out1, W2, b2)
    relu_out2 = tf.nn.relu(conv_out2)
    x_norm2 = tf.layers.batch_normalization(relu_out2,axis=3, training=train_flag)
    maxpool_out2 = max_pool_2x2(x_norm2)
    maxpool_reshaped = tf.reshape(maxpool_out2, [-1, W3.get_shape().as_list()[0]])
    local = tf.add(tf.matmul(maxpool_reshaped, W3), b3)
    local_out = tf.nn.relu(local)

    out = tf.add(tf.matmul(local_out, W_out), b_out)
    return out


model_op = model(train_flag)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=model_op, labels=y)
)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer().minimize(loss)

correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()
'''
var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    onehot_labels = tf.one_hot(labels, len(names), axis=-1)
    onehot_vals = sess.run(onehot_labels)
    batch_size = 64
    print('batch size', batch_size)
    for j in range(0, 100):
        avg_accuracy_val = 0.
        batch_count = 0.
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size, :]
            batch_onehot_vals = onehot_vals[i:i+batch_size, :]
            sess.run(train_op, feed_dict={x: batch_data, y: batch_onehot_vals,train_flag:True})
            accuracy_val = sess.run(accuracy, feed_dict={x: batch_data, y: batch_onehot_vals,train_flag:False})
            avg_accuracy_val += accuracy_val
            batch_count += 1.
        avg_accuracy_val /= batch_count
        print('Epoch {}. Avg accuracy {}'.format(j, avg_accuracy_val))

    g_list = tf.global_variables()
    sess.run(g_list)
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    print(bn_moving_vars)
    save_path = saver.save(sess, './model/CNN_BN_Cifar10_model')
    print("Model saved in file: %s" % save_path)
    test_onehot_labels = tf.one_hot(test_labels, len(names), axis=-1)
    test_onehot_vals = sess.run(test_onehot_labels)
    testInputs = test_data[0:64, :]
    testLabels = test_onehot_vals[0:64, :]
    testAccuracy = sess.run( accuracy, feed_dict={x: testInputs, y: testLabels,train_flag:False})
    print("test accuracy %g" % (testAccuracy))
