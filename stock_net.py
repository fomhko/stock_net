import tensorflow as tf
import zhusuan as zs
from zhusuan import reuse
from dataset import preprocess
from tensorflow.python import debug as tf_debug
import config
def MIE(input_data,state): #GRU market information encoder
    with tf.variable_scope("GRU",reuse=tf.AUTO_REUSE):
        cell = tf.nn.rnn_cell.GRUCell(num_units = config.MIE_UNITS,name = 'GRUCELL')
        (cell_out,state) = cell(inputs = input_data,state = state)
        return state,cell_out

def q_net(input_data,y,market_encode,prev_z):
    with zs.BayesianNet() as encoder:
        cat = tf.concat([prev_z,
                        market_encode,
                        input_data,
                        y],1)
        h_z = tf.layers.dense(inputs = cat,units = config.LATENT_SIZE,activation=tf.nn.tanh,name = 'h_z',reuse = tf.AUTO_REUSE)
        z_mean = tf.layers.dense(inputs = h_z,units = config.LATENT_SIZE, activation=tf.nn.tanh, name = 'z_mu', reuse=tf.AUTO_REUSE)
        z_logstd = tf.layers.dense(inputs = h_z, units = config.LATENT_SIZE, activation=tf.nn.tanh, name = 'z_delta', reuse = tf.AUTO_REUSE)
        z = zs.Normal(mean = z_mean,logstd = z_logstd,group_ndims = 1,name='z',reuse = tf.AUTO_REUSE)
    return z

@zs.reuse('decoder')
def p_net(observed,input_data,market_encode,prev_z,G,Y,gen_mode = False):
    with zs.BayesianNet(observed=observed) as decoder:
        cat = tf.concat([prev_z,
                        market_encode,
                        input_data],1)
        h_z = tf.layers.dense(inputs=cat,units=config.LATENT_SIZE,activation = tf.nn.tanh,name='h_z_prior',reuse=tf.AUTO_REUSE)
        z_mean = tf.layers.dense(inputs=h_z,units=config.LATENT_SIZE,activation = None,name='z_mu_prior',reuse = tf.AUTO_REUSE)
        z_logstd = tf.layers.dense(inputs=h_z,units=config.LATENT_SIZE,activation = None,name='z_delta_prior',reuse = tf.AUTO_REUSE)
        z = zs.Normal(name='z',mean=z_mean,logstd = z_logstd,group_ndims=2,reuse=tf.AUTO_REUSE)
        p_z = zs.Normal(name = 'pz', mean=z_mean,logstd = z_logstd,group_ndims=2,reuse=tf.AUTO_REUSE)
        if gen_mode:#decode
            cat = tf.concat([input_data,
                            market_encode,
                            z_mean],1)
        else:#inference
            cat = tf.concat([input_data,
                            market_encode,
                            z],1)

        g = tf.layers.dense(inputs=cat, units=config.LATENT_SIZE, name='g', activation=tf.nn.tanh,reuse = tf.AUTO_REUSE)
        y = tf.layers.dense(inputs=g, units=2, activation=tf.nn.softmax,name='y_hat',reuse = tf.AUTO_REUSE)
        if G.__len__() < config.SEQ_LEN-1:
            Y.append(y)
            G.append(g)
        return y,g,p_z
def ATA(Y,G,g_t): #Attentive Temporal Auxiliary
    with tf.variable_scope("ATA", reuse=tf.AUTO_REUSE):
        Y = tf.concat([Y],0)
        Y = tf.transpose(Y,perm = [1,2,0])
        G = tf.concat([G],0)
        G = tf.transpose(G,perm = [1,0,2])
        v_i = tf.layers.dense(inputs = G,units = config.LATENT_SIZE,activation = tf.nn.tanh,use_bias=False,name = 'v_i_tanh',
                              reuse=tf.AUTO_REUSE)
        v_i = tf.layers.dense(inputs = v_i,units = 1,activation=None,use_bias=False,name = 'v_i',
                              reuse=tf.AUTO_REUSE)
        v_d = tf.layers.dense(inputs = G,units = config.LATENT_SIZE,activation = tf.nn.tanh,use_bias=False,name = 'v_d_tanh',
                              reuse=tf.AUTO_REUSE)
        g_t = tf.reshape(g_t,[-1,config.LATENT_SIZE,1])
        v_d = tf.matmul(v_d,g_t)
        v_star = tf.nn.softmax(tf.multiply(v_i,v_d),axis = 1)
        weighted_y = tf.matmul(Y,v_star)
        cat = tf.squeeze(tf.concat([weighted_y,g_t],axis = 1),axis = 2)
        y_T = tf.layers.dense(inputs = cat,units = 2,activation = tf.nn.softmax,name = 'classification', reuse=tf.AUTO_REUSE)
        return y_T,v_star

def train_minibatch(batch,l_batch,anneal,seq_len = config.SEQ_LEN):
    # with tf.Graph().as_default() as graph:
    state = tf.zeros(shape=[config.BATCH_SIZE, config.MIE_UNITS], name='state')
    state = tf.placeholder_with_default(state, state.shape, state.op.name)
    # z = tf.zeros(shape=[config.BATCH_SIZE, config.LATENT_SIZE], name='z')
    # z = tf.placeholder_with_default(z, z.shape, z.op.name)
    prev_z = tf.zeros(shape=[config.BATCH_SIZE, config.LATENT_SIZE], name='prev_z')
    prev_z = tf.placeholder_with_default(prev_z, prev_z.shape, prev_z.op.name)
    G=[]
    Y=[]
    f = []
    for time_step in range(seq_len):
        state,_ = MIE(batch[:,time_step,:],state)

        z = q_net(input_data=batch[:,time_step,:],
                  market_encode=state,
                  y=l_batch[:,time_step,:],
                  prev_z = prev_z)
        y,g,p_z = p_net(observed={'z':z},
                        input_data=batch[:,time_step,:],
                        market_encode=state,
                        G = G,
                        Y = Y,
                        prev_z=prev_z,
                        )
        prev_z = z
        if(time_step == seq_len - 1):
            y,v_star = ATA(Y=Y,G=G,g_t=g)
        rec_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(l_batch[:,time_step,:],1))
        kld = -0.5 * tf.reduce_sum(
                tf.log(tf.square(z.distribution.std) + 0.0001) - tf.log(tf.square(p_z.distribution.std) + 0.0001)
                - (tf.square(z.distribution.std) + tf.square(p_z.distribution.mean - p_z.distribution.mean)) / (
                            tf.square(p_z.distribution.std) + 0.0001) + 1, 1)
        rec_loss = tf.reshape(rec_loss,[config.BATCH_SIZE,1])
        kld = tf.reshape(kld,[config.BATCH_SIZE,1])
        f.append(kld+anneal*rec_loss)

    f = tf.concat([f],axis = 0)
    f = tf.transpose(f,perm=[1,0,2])
    print(f)
    v = tf.concat([config.ALPHA*v_star,tf.ones([config.BATCH_SIZE,1,1])],1)
    loss = tf.multiply(v,f)
    gradients = tf.gradients(loss, tf.trainable_variables())
    opt = tf.train.AdamOptimizer(learning_rate= config.LR)
    optimize = opt.apply_gradients(zip(gradients,
                               tf.trainable_variables()))
    return optimize

if __name__ == "__main__":
    dataset,labelset = preprocess(config.DATA_DIR+"AAPL.txt")
    import numpy as np
    with tf.Graph().as_default() as graph:
        batch = tf.placeholder(shape = [config.BATCH_SIZE,config.SEQ_LEN,3],dtype=tf.float32,name = 'batch')
        l_batch = tf.placeholder(shape = [config.BATCH_SIZE,config.SEQ_LEN,2],dtype=tf.float32, name = 'l_batch')
        anneal = tf.placeholder(dtype = tf.float32)
        optimize = train_minibatch(batch = batch,l_batch= l_batch,anneal = anneal)
        num_iters = len(dataset) // config.BATCH_SIZE
        with tf.Session() as sess:
            for e in range(config.EPOCH):
                for i in range(num_iters):
                    sess.run(tf.global_variables_initializer())
                    feed = {batch:dataset[i*config.BATCH_SIZE:(i+1)*config.BATCH_SIZE,:,:],
                            l_batch:labelset[i*config.BATCH_SIZE:(i+1)*config.BATCH_SIZE,:,:],anneal:1}
                    sess.run(optimize,feed_dict=feed)


    # with tf.Graph().as_default() as graph:
        # input_data = tf.placeholder(tf.float32,[config.BATCH_SIZE,config.SEQ_LEN,3],'input_data')
        # input_label = tf.placeholder(tf.float32,[config.BATCH_SIZE,config.SEQ_LEN,2],'input_label')
        # state = tf.zeros(shape = [config.BATCH_SIZE,config.MIE_UNITS],name='state')
        # state = tf.placeholder_with_default(state, state.shape, state.op.name)
        # z = tf.zeros(shape=[config.BATCH_SIZE, config.MIE_UNITS], name='z')
        # z = tf.placeholder_with_default(z, z.shape, z.op.name)
        # G_ = []
        # Y_ = []
        # for i in range(config.SEQ_LEN):
        #     state,_ = MIE(input_data=input_data[:, i, :],
        #                   state=state)
        #     z = q_net(input_data=input_data[:, i, :],
        #               market_encode=state,
        #               y=input_label[:, i, :],
        #               prev_z=z)
        #
        #     y, g_t = p_net(observed={'z':z},
        #                       input_data = input_data[:,i,:],
        #                       market_encode=state,
        #                       prev_z=z,
        #                       G=G_,
        #                       Y=Y_,
        #                       )
        # last_predict = ATA(Y = Y_,G= G_,g_t = g_t)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     small_data = dataset[:config.BATCH_SIZE,:,:]
        #     small_label = labelset[:config.BATCH_SIZE,:,:]
        #     dict = {input_data:small_data,input_label:small_label}
        #     y_T,v_star = sess.run(last_predict,feed_dict=dict)
            # with tf.variable_scope("GRU", reuse=tf.AUTO_REUSE):
            #     variable = sess.run([tf.get_variable("GRUCELL/gates/kernel")],feed_dict=dict)
            #     print (variable)
            # with tf.variable_scope("GRU", reuse=tf.AUTO_REUSE):
            #     variable = sess.run([tf.get_variable("GRUCELL/gates/kernel")],feed_dict=dict)
            #     print (variable)
            # for op in tf.get_default_graph().get_operations():
            #     print(str(op.name) )
            # assert (final_state== final_state).all()