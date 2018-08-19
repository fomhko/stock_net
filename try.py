import tensorflow as tf
import zhusuan as zs
import config
import numpy as np
def q_net(x):
    with zs.BayesianNet() as encoder:
        # x = tf.placeholder(tf.float32,[10,10,],'x')
        # mean = tf.layers.dense(inputs = x,units = 10, activation = None)
        z=zs.Normal(name = 'z1',mean=x,std = tf.ones([10,10]))
    return z
@zs.reuse('decoder')
def vae(observed):
    with zs.BayesianNet(observed=observed) as decoder:
        z = zs.Normal(name='z1', mean=tf.zeros([10, 10]), logstd=tf.zeros([10, 10]))
        z = tf.layers.dense(inputs = z,units = 1,activation = None,trainable=False)
        return z

if __name__ == "__main__":
    with tf.Graph().as_default() as graph:
        # x = tf.placeholder(tf.float32, [10, 10], 'x')
        # z = q_net(x)
        # z_d = vae({'z1':z})
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     dict = {x:np.zeros([10,10],np.float32)}
        #     x = sess.run([z_d],feed_dict = dict)
        #     print(x)
        c = tf.placeholder(shape = [16,config.LATENT_SIZE],dtype=tf.float32,name = 'batch')
        a = tf.zeros([16,10])
        b = tf.layers.dense(inputs = a,units=config.LATENT_SIZE, name='g', activation=tf.nn.tanh,reuse = tf.AUTO_REUSE)
        loss = tf.reduce_mean(tf.square(tf.subtract(b,c)))
        opt = tf.train.AdamOptimizer(learning_rate=config.LR)
        opt = tf.train.AdamOptimizer(learning_rate=config.LR)
        gradients = tf.gradients(loss, tf.trainable_variables())
        clipped_grad, _ = tf.clip_by_global_norm(gradients, 5)
        optimize = opt.apply_gradients(zip(clipped_grad,
                                           tf.trainable_variables()))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed = {c:np.ones([16,config.LATENT_SIZE])}
            while(1):
                _,loss_ = sess.run([optimize,loss],feed_dict=feed)
                print(loss_)
        #

        # print(z.eval())