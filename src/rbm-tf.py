import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np 

class RBM():
    def __init__(self, nv=30*30, nh=1800, cd_steps=3):
        self.graph = tf.Graph()
        self.nv = nv
        self.nh = nh
        with self.graph.as_default(): 
            self.W = tf.Variable(tf.truncated_normal((nv, nh)) * 0.01, name='vh_weights')
            self.bv = tf.Variable(tf.zeros((nv, 1))) 
            self.bh = tf.Variable(tf.zeros((nh, 1)))
            
            self.cd_steps = cd_steps 
            self.modelW = None 
    
    def bernoulli(self, p):
        return tf.nn.relu(tf.sign(p - tf.random_uniform(p.shape)))
    
    def energy(self, v):
        b_term = tf.matmul(v, self.bv)
        linear_tranform = tf.matmul(v, self.W) + tf.squeeze(self.bh)
        h_term = tf.reduce_sum(tf.log(tf.exp(linear_tranform) + 1), axis=1) 
        return tf.reduce_mean(-h_term -b_term)
    
    def sample_h(self, v):
        ph_given_v = tf.sigmoid(tf.matmul(v, self.W) + tf.squeeze(self.bh))
        return self.bernoulli(ph_given_v)
    
    def sample_v(self, h):
        pv_given_h = tf.sigmoid(tf.matmul(h, tf.transpose(self.W)) + tf.squeeze(self.bv))
        return self.bernoulli(pv_given_h)
    
    def gibbs_step(self, i, k, vk):
        hk = self.sample_h(vk)
        vk = self.sample_v(hk)
        return i+1, k, vk
    
    def train(self, X, lr=0.01, batch_size=64, epochs=5, modelname='vh_weights'):
        with self.graph.as_default():
            print(self.bv.shape)
            tf_v = tf.placeholder(tf.float32, [batch_size, self.bv.shape[0]])
            v = tf.round(tf_v) 
            vk = tf.identity(v)

            i = tf.constant(0)
            _, _, vk = tf.while_loop(cond = lambda i, k, *args: i <= k,
                                      body = self.gibbs_step,
                                      loop_vars = [i, tf.constant(self.cd_steps), vk],
                                      parallel_iterations=1,
                                      back_prop=False)

            vk = tf.stop_gradient(vk) 
            loss = self.energy(v) - self.energy(vk) 
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
            init = tf.global_variables_initializer()
       
        saver = tf.train.Saver([self.W])
        with tf.Session(graph=self.graph) as sess:
            init.run()
            for epoch in range(epochs): 
                losses = []
                for i in range(0, len(X)-batch_size, batch_size):
                    x_batch = X[i:i+batch_size] 
                    l, _ = sess.run([loss, optimizer], feed_dict={tf_v: x_batch})
                    losses.append(l)
                print('Epoch Cost %d: ' % (epoch), np.mean(losses))
            self.modelW = self.W.eval()
            saver.save(sess, modelname + str(self.nh))

def main():
    from utils import load_data
    import sys
    filename = sys.argv[1]
    data = load_data(filename)
    data = data.inputs.T
    rbm0 = RBM(nv=30*30, nh=1800, cd_steps=3)
    rbm0.train(X=data, lr=0.001, epochs=22)
    print(rbm0.modelW)
    rbm1 = RBM(nv=1800, nh=900, cd_steps=3)
    rbm1.train(X=rbm0.modelW, lr=0.001, epochs=70)
    print(rbm1.modelW)
    rbm2 = RBM(nv=900, nh=450, cd_steps=3)
    rbm2.train(X=rbm1.modelW, lr=0.001, epochs=90)
    print(rbm2.modelW)
    rbm3 = RBM(nv=450, nh=225, cd_steps=3)
    rbm3.train(X=rbm2.modelW, lr=0.001, epochs=100)
    print(rbm3.modelW)
    rbm4 = RBM(nv=225, nh=112, cd_steps=3)
    rbm4.train(X=rbm3.modelW, lr=0.001, epochs=250)
    print(rbm4.modelW)
    rbm5 = RBM(nv=112, nh=60, cd_steps=3)
    rbm5.train(X=rbm4.modelW, lr=0.001, epochs=500)
    print(rbm5.modelW)


if __name__ == '__main__':
    main()
