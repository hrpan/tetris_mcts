import tensorflow as tf
import sys
import shutil
IMG_SIZE = (22,10,1)

IMG_H = IMG_SIZE[0]
IMG_W = IMG_SIZE[1]
IMG_C = IMG_SIZE[2]

B_SIZE = IMG_H * IMG_W

smoothing = 0.

EXP_DIR = './saved_model/model'

def pipeline(states):

    def rot_states(states,k):
        return tf.map_fn(lambda state: tf.image.rot90(state,k), states)

    rot = [rot_states(states,i) for i in range(4)]

    flip_rot = [tf.reverse(s,axis=[1]) for s in rot]

    return rot+flip_rot

def sym_policy(p):
    
    p_slice = tf.slice(p,[0,0],[-1,B_SIZE])  
    pass_slice = tf.slice(p,[0,B_SIZE],[-1,1])
    p_board = tf.reshape(p_slice,[-1,IMG_H,IMG_W,1])
    sym_p_board = pipeline(p_board)
    sym_p_board = [tf.concat([tf.reshape(s,[-1,B_SIZE]),pass_slice],axis=1) for s in sym_p_board]
    return sym_p_board

def residual_block(x,kernel_size,training):
    f = x.shape[-1]
    y = tf.layers.conv2d(x,filters=f,kernel_size=kernel_size,activation=tf.nn.relu,padding='same')
    y = tf.layers.batch_normalization(y,training=training)
    y = tf.layers.conv2d(y,filters=f,kernel_size=kernel_size,activation=tf.nn.relu,padding='same')
    y = tf.layers.batch_normalization(y,training=training)
    y = x + y
    return y

class Model:
    def __init__(self):

        self.graph = tf.Graph()

    def build_graph(self):
        self.graph.as_default()

        self.state = tf.placeholder(tf.float32,[None,IMG_H,IMG_W,IMG_C],name='state')
        self.value_true = tf.placeholder(tf.float32,[None,1],name='value_true')
        self.policy_true = tf.placeholder(tf.float32,[None,6],name='policy_true')
        self.training = tf.placeholder(tf.bool,name='training')



        n_filters = 64
        k_size = 3
        act = tf.nn.relu
        y = tf.layers.conv2d(self.state,filters=n_filters,kernel_size=k_size,activation=act,name='conv1')
#        y = tf.layers.batch_normalization(y,training=self.training)
        y = tf.layers.conv2d(y,filters=n_filters,kernel_size=k_size,activation=act,name='conv2')
#        y = tf.layers.batch_normalization(y,training=self.training)
#        y = tf.layers.conv2d(y,filters=n_filters,kernel_size=k_size,activation=act,name='conv3')
        y = tf.layers.flatten(y)
        """
        for i in range(3):
            y = residual_block(y,k_size,self.training)
        """
        
        self.value_pred = tf.layers.dense(y,1,activation=tf.exp)
        #self.value_pred = tf.layers.dense(y,1,activation=tf.nn.relu)
        tf.identity(self.value_pred,name='value_pred')
        self.policy_logit = tf.layers.dense(y,6,name='policy_logit')
        self.policy_pred = tf.nn.softmax(self.policy_logit,name='policy_pred')

        loss_v = tf.losses.mean_squared_error(self.value_true,self.value_pred)
        self.loss_value = tf.identity(loss_v,name='loss_value')
        #loss_p = tf.losses.softmax_cross_entropy(self.policy_true,self.policy_logit)
        loss_p = tf.constant(0.)
        self.loss_policy = tf.identity(loss_p,name='loss_policy')


        self.loss = tf.add(self.loss_value,self.loss_policy,name='loss')

        self.optimizer = tf.train.AdamOptimizer()
#        self.optimizer = tf.train.RMSPropOptimizer(0.01)
#        self.optimizer = tf.train.GradientDescentOptimizer(0.01)
        self.g_step = tf.placeholder(tf.int32,name='g_step')
        lr_init = 0.001
        self.lr = tf.train.exponential_decay(lr_init,self.g_step,10000,0.1,staircase=True,name='lr')
#        self.lr = tf.constant(lr_init,name='lr')
#        self.optimizer = tf.train.MomentumOptimizer(self.lr,0.9,use_nesterov=True)
#        self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = self.optimizer.minimize(self.loss,name='train_step')
        for var in tf.trainable_variables():
            print(var)

    def train(self,sess,batch,step):
        _dict = {self.state:batch[0], 
                self.value_true:batch[1], 
                self.policy_true:batch[2],
                self.training:True, 
                self.g_step:step}
        ops = [self.loss,self.loss_value,self.loss_policy,self.train_step]
        _loss, _loss_val, _loss_pol, _ = sess.run(ops,feed_dict=_dict)
        return (_loss,_loss_val,_loss_pol)

    def compute_loss(self,sess,batch):
        _dict = {self.state:batch[0], 
                self.value_true:batch[1], 
                self.policy_true:batch[2],
                self.training:False}
        ops = [self.loss,self.loss_value,self.loss_policy]
        _loss, _loss_val, _loss_pol = sess.run(ops,feed_dict=_dict)
        return (_loss,_loss_val,_loss_pol)

    def inference(self,sess,batch):
        _dict = {self.state:batch,self.training:False}
        with self.graph.as_default():
            _r = sess.run([self.value_pred,self.policy_pred],feed_dict=_dict)
        return _r

    def save(self,sess):
        while True:
            try:
                builder = tf.saved_model.builder.SavedModelBuilder(EXP_DIR)
                break
            except AssertionError:
                sys.stdout.write('WARNING: REMOVING PREVIOUS MODELS')
                sys.stdout.flush()
                shutil.rmtree(EXP_DIR)
        builder.add_meta_graph_and_variables(sess,['graph'])
        builder.save()

    def load(self,sess):
        try:
            tf.saved_model.loader.load(sess,['graph'],EXP_DIR)
            self.graph = tf.get_default_graph()
            self.state = self.graph.get_tensor_by_name('state:0')
            self.value_true = self.graph.get_tensor_by_name('value_true:0')
            self.policy_true = self.graph.get_tensor_by_name('policy_true:0')
            self.value_pred = self.graph.get_tensor_by_name('value_pred:0')
            self.policy_pred = self.graph.get_tensor_by_name('policy_pred:0')
            self.training = self.graph.get_tensor_by_name('training:0')
            self.lr = self.graph.get_tensor_by_name('lr:0')

            self.loss = self.graph.get_tensor_by_name('loss:0')
            self.loss_value = self.graph.get_tensor_by_name('loss_value:0')
            self.loss_policy = self.graph.get_tensor_by_name('loss_policy:0')
            self.g_step = self.graph.get_tensor_by_name('g_step:0')
            self.train_step = self.graph.get_operation_by_name('train_step')
        except IOError:
            sys.stdout.write('Model does not exist, building a new one...')
            sys.stdout.flush()
            self.build_graph()
            sess.run(tf.global_variables_initializer())
