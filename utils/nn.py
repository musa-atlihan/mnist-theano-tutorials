try:
   import cPickle as pickle
except:
   import pickle
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool


class LogisticRegression(object):
    """Multi-class logistic regression class."""

    def __init__(self, input, n_in, n_out):
        """Initialize the parameters of the logistic regression."""

        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in a (mini)batch."""

        # Check if y has same dimension of y_pred.
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        # Check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    """Hidden layer for Multilayer perceptron."""

    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                  activation=T.tanh):

        self.input = input

        if W is None:
            W_vals = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            if activation == T.nnet.sigmoid:
                W_vals = W_vals * 4.

            W = theano.shared(value=W_vals, name='W', borrow=True)

        if b is None:
            b_vals = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_vals, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]


class MLP(object):
    """Multilayer perceptron."""

    def __init__(self, rng, input, n_in, n_hidden, n_out):

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegLayer.negative_log_likelihood
        )

        self.errors = self.logRegLayer.errors

        self.params = (
            self.hiddenLayer.params
            + self.logRegLayer.params
        )

        self.input = input


class ConvLayer(object):
    """Convolutional layer."""

    def __init__(self, rng, input, filter_shape, image_shape=None,
                            down_pooling=True,
                            pool_size=(2, 2),
                            activation=T.nnet.relu):

        if not image_shape is None:
            assert filter_shape[1] == image_shape[1]
        
        self.input = input

        self.fan_in = np.prod(filter_shape[1:])
        self.fan_out = filter_shape[0] * np.prod(filter_shape[2:])

        self.W_bound = np.sqrt(6. / (self.fan_in + self.fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-self.W_bound, high=self.W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_vals = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_vals, borrow=True)

        conv_out = conv2d(
            input=input,
            filters=self.W,
            input_shape=image_shape,
            filter_shape=filter_shape
        )

        if down_pooling:
            conv_out = pool.pool_2d(
                input=conv_out,
                ds=pool_size,
                ignore_border=True
        )

        lin_output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]

        self.filter_shape = filter_shape

        self.rng = rng

    def reset(self):
        """Reset W and b.

        """

        W_vals = np.asarray(
                self.rng.uniform(low=-self.W_bound, high=self.W_bound, size=self.filter_shape),
                dtype=theano.config.floatX
            )

        b_vals = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)

        self.W.set_value(W_vals, borrow=True)
        self.b.set_value(b_vals, borrow=True)


class LeNet(object):
    """Convolutional neural network from LeNet family."""

    def __init__(self, rng, input, filter_shapes, n_hidden, n_out, image_shape=None):

        self.layer3 = ConvLayer(
            rng=rng,
            input=input,
            filter_shape=filter_shapes[0],
            image_shape=image_shape,
            down_pooling=True,
            activation=T.tanh
        )

        image_shape2 = (
            image_shape[0],
            filter_shapes[0][0],
            (image_shape[2] - filter_shapes[0][2] + 1) // 2,
            (image_shape[3] - filter_shapes[0][3] + 1) // 2
        )

        self.layer2 = ConvLayer(
            rng=rng,
            input=self.layer3.output,
            filter_shape=filter_shapes[1],
            image_shape=image_shape2,
            down_pooling=True,
            activation=T.tanh
        )

        layer1_input = self.layer2.output.flatten(2)
        
        n_layer1_in = (
            filter_shapes[1][0] 
            * ((image_shape2[2] - filter_shapes[1][2] + 1) // 2)
            * ((image_shape2[3] - filter_shapes[1][3] + 1) // 2)
        )

        self.layer1 = HiddenLayer(
            rng=rng,
            input=layer1_input,
            n_in=n_layer1_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.layer0 = LogisticRegression(
            input=self.layer1.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.layer0.W).sum()
            + abs(self.layer1.W).sum()
            + abs(self.layer2.W).sum()
            + abs(self.layer3.W).sum()
        )

        self.L2_sqr = (
            (self.layer0.W ** 2).sum()
            + (self.layer1.W ** 2).sum()
            + (self.layer2.W ** 2).sum()
            + (self.layer3.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.layer0.negative_log_likelihood
        )

        self.errors = self.layer0.errors

        self.params = (
            self.layer0.params
            + self.layer1.params 
            + self.layer2.params 
            + self.layer3.params 
        )

        self.y_pred = self.layer0.y_pred

        self.input = input

    def reset(self):
        self.layer0.reset()
        self.layer1.reset()
        self.layer2.reset()
        self.layer3.reset()