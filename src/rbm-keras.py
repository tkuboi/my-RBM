import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class InputLayer(layers.Layer):
    def __init__(self, output_dim, input_dim, trainable=False, name=None, dtype=None, dynamic=False, **kwargs):
        super(InputLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self.w = self.add_weight(shape=(input_dim, output_dim),
                                 initializer='random_normal',
                                 trainable=trainable, name='w')
        self.bv = self.add_weight(shape=(input_dim,1),
                                  initializer='zeros',
                                  trainable=trainable, name='bv')
        self.bh = self.add_weight(shape=(output_dim,1),
                                  initializer='zeros',
                                  trainable=trainable, name='bh')

    def load_weights(self, filepath):
        # Loads weights from HDF5 file
        import h5py
        f = h5py.File(filepath)
        weights = [f['param_{}'.format(p)] for p in range(f.attrs['nb_params'])]
        self.set_weights(weights)
        f.close()

    def call(self, x):
        linear_tranform = tf.matmul(x + tf.squeeze(self.bv), self.w) + tf.squeeze(self.bh)
        return tf.nn.relu(linear_tranform)

class RBM(layers.Layer):
    def __init__(self,
                 hidden_dim=1800,
                 visible_dim=900,
                 cd_steps=3,
                 name='rbm',
                 **kwargs):
        super(RBM, self).__init__(name=name, **kwargs)

        self.cd_steps = cd_steps
        self.w = self.add_weight(shape=(visible_dim, hidden_dim),
                                 initializer='random_normal',
                                 trainable=True, name='w')
        self.bv = self.add_weight(shape=(visible_dim,1),
                                  initializer='zeros',
                                  name='bv')
        self.bh = self.add_weight(shape=(hidden_dim,1),
                                  initializer='zeros',
                                  name='bh')
        self.loss = 0

    def save_weights(self, filepath, overwrite=False):
        # Save weights to HDF5
        import h5py
        import os.path
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            import sys
            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % (filepath))
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return
            print('[TIP] Next time specify overwrite=True in save_weights!')

        f = h5py.File(filepath, 'w')
        weights = self.get_weights()
        f.attrs['nb_params'] = len(weights)
        for n, param in enumerate(weights):
            param_name = 'param_{}'.format(n)
            param_dset = f.create_dataset(param_name, param.shape, dtype=param.dtype)
            param_dset[:] = param
        f.flush()
        f.close()

    def load_weights(self, filepath):
        # Loads weights from HDF5 file
        import h5py
        f = h5py.File(filepath)
        weights = [f['param_{}'.format(p)] for p in range(f.attrs['nb_params'])]
        self.set_weights(weights)
        f.close()

    def get_loss(self):
        return self.loss 

    def bernoulli(self, p, shape):
        return tf.nn.relu(tf.sign(p - tf.random.uniform(shape)))

    def energy(self, v):
        b_term = tf.matmul(v, self.bv)
        linear_tranform = tf.matmul(v, self.w) + tf.squeeze(self.bh)
        h_term = tf.reduce_sum(tf.math.log(tf.exp(linear_tranform) + 1), axis=1)
        return tf.reduce_mean(-h_term - b_term)

    def sample_h(self, v):
        h_pre = tf.matmul(v, self.w) + tf.squeeze(self.bh)
        ph_given_v = tf.sigmoid(h_pre)
        return h_pre, ph_given_v, self.bernoulli(ph_given_v, ph_given_v.shape)

    def sample_v(self, h):
        v_pre = tf.matmul(h, tf.transpose(self.w)) + tf.squeeze(self.bv)
        pv_given_h = tf.sigmoid(v_pre)
        return v_pre, pv_given_h, self.bernoulli(pv_given_h, pv_given_h.shape)

    def gibbs_step(self, visual_state):
        h_pre, h_sigm, hidden_state = self.sample_h(visual_state)
        v_pre, v_sigm, visual_state = self.sample_v(hidden_state)
        return v_pre, v_sigm, visual_state

    def mcmc_chain(self, x, nb_gibbs_steps):
        """
        Perform Markov Chain Monte Carlo, run k steps of Gibbs sampling, 
        starting from visible data, return point estimate at end of chain.

           x0 (data) -> h1 -> x1 -> ... -> xk (reconstruction, negative sample)
        """

        vs = x
        for i in range(nb_gibbs_steps):
            v_pre, v_sigm, vs = self.gibbs_step(vs)

        vs = tf.stop_gradient(vs)    # avoid back-propagating gradient through the Gibbs sampling
                                                            # this is similar to T.grad(.., consider_constant=[chain_end])
                                                            # however, as grad() is called in keras.optimizers.Optimizer, 
                                                            # we do it here instead to avoid having to change Keras' code

        return v_pre, v_sigm, vs

    def contrastive_divergence_loss(self):
        """
        Compute contrastive divergence loss with k steps of Gibbs sampling (CD-k).

        Result is a Theano expression with the form loss = f(x).
        """
        def loss(x):
            _, _, vs = self.mcmc_chain(x, self.cd_steps)
            cd = K.mean(self.energy(x)) - K.mean(self.energy(vs))
            return cd

        return loss

    def reconstruction_loss(self, nb_gibbs_steps=1):
        """
        Compute binary cross-entropy between the binary input data and the reconstruction generated by the model.

        Result is an expression with the form loss = f(x).

        Useful as a rough indication of training progress (see Hinton2010).
        Summed over feature dimensions, mean over samples.
        """

        def loss(x):
            pre, _, _ = self.mcmc_chain(x, nb_gibbs_steps)

            cross_entropy_loss = -K.mean(tf.reduce_sum(x*tf.math.log(tf.sigmoid(pre)) + (1 - x)*tf.math.log(1 - tf.sigmoid(pre)), axis=1))

            return cross_entropy_loss

        return loss

    def cross_entropy_loss(self, x, pre):
        return -K.mean(tf.reduce_sum(x*tf.math.log(tf.sigmoid(pre)) + (1 - x)*tf.math.log(1 - tf.sigmoid(pre)), axis=1))

    def call(self, inputs):
        x = inputs
        pre, _, vs = self.mcmc_chain(x, self.cd_steps)
        self.loss = K.mean(self.energy(x)) - K.mean(self.energy(vs))
        print(self.loss)
        return pre 

class RBMEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 hidden_dim=1800,
                 visible_dim=900,
                 name='rbmencoder',
                 **kwargs):
        super(RBMEncoder, self).__init__(name=name, **kwargs)
        self.input_layers = []
        self.layer = RBM(visible_dim=visible_dim,
                         hidden_dim=hidden_dim)

    def add_input_layer(self, layer):
        self.input_layers.append(layer)

    def load_input_layer_weights(self, files):
        zipped = zip(self.input_layers, files)
        for layer, filename in zipped:
            layer.load_weights(filename)

    def through_layers(self, x):
        for layer in self.input_layers:
            x = layer(x)
        return x

    def call(self, inputs):
        reconstructed = self.layer(inputs)
        #loss = self.layer.get_loss()
        #loss = self.layer.contrastive_divergence_loss()
        #self.add_loss(lambda: loss(inputs), inputs)
        return reconstructed

    def update_weights(self, w, loss, momentum_speed):
        momentum_speed = 0.9 * momentum_speed + loss
        print(momentum_speed)
        w = w + momentum_speed * self.learning_rate
        print(w)
        return w, momentum_speed

    def train(self, inputs, lr=0.001, batch_size=100, epochs=10):
        self.learning_rate = lr
        loss_metric = tf.keras.metrics.Mean()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        momentum_speed = tf.zeros(self.layer.w.shape)

        train_dataset = tf.data.Dataset.from_tensor_slices(inputs.astype('float32'))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        # Iterate over epochs.
        for epoch in range(epochs):
            print('Start of epoch %d' % (epoch,))

            # Iterate over the batches of the dataset.
            for step, x_batch_train in enumerate(train_dataset):
                x = self.through_layers(x_batch_train)
                reconstructed = self.call(x)
                loss = self.layer.get_loss()
                optimizer.apply_gradients(zip([momentum_speed + loss], [self.layer.w]))
                #self.layer.w, momentum_speed = self.update_weights(self.layer.w, loss, momentum_speed)
                loss_metric(loss)
                if step % 100 == 0:
                    print('step %s: mean loss = %s' % (step, loss_metric.result()))
        
def main():
    from utils import load_data
    import sys
    filename = sys.argv[1]
    data = load_data(filename)
    data = data.inputs.T
    #dataset = tf.data.Dataset.from_tensor_slices(data)

    rbm = RBMEncoder(50, 112)
    rbm.add_input_layer(InputLayer(1800, 900))
    rbm.add_input_layer(InputLayer(900, 1800))
    rbm.add_input_layer(InputLayer(450, 900))
    rbm.add_input_layer(InputLayer(225, 450))
    rbm.add_input_layer(InputLayer(112, 225))
    print(rbm.layer.w)
    #rbm.layer.load_weights("saved_weights.h5")
    rbm.load_input_layer_weights(["saved_weights.h5", "saved_weights2.h5", "saved_weights3.h5", "saved_weights4.h5", "saved_weights5.h5"])
    print(rbm.layer.w)
    rbm.train(data)
    print(rbm.layer.w)
    rbm.layer.save_weights("saved_weights6.h5", True)

if __name__ == '__main__':
    main()

