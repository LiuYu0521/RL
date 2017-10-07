import tensorflow as tf

with tf.name_scope('inputs'):
    x_input = tf.placeholder(tf.float32, [None, 1], name='x_input')
    y_input = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = tf.layers.dense(x_input, 10, activation=tf.nn.relu, name = 'layer1')
prediction = tf.layers.dense(l1, 1, name='output')

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - y_input), reduction_indices=1))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)