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
        a = []
        a .append(tf.zeros([32,10]))
        a.append( tf.ones([32,10]))
        a = tf.concat([a],axis = 1)
        a =tf.transpose(a,perm = [1,0,2])
        print(a)
            # print(z_d.eval())
        #

        # print(z.eval())
