try:
   import cPickle as pickle
except:
   import pickle
import os
import numpy as np
import sys
import timeit
import theano
import theano.tensor as T
import utils.data as utld
import utils.nn as nn


dataset_path = 'path_to_file/mnist.pkl'


def halt():
    sys.exit(0)


def optimize_logreg(learning_rate=0.13, n_epochs=1000,
                            dataset=dataset_path,
                            batch_size=600):
    """Optimize Logistic regression model using mini-batch
    gradient descent.
    """

    train_set, valid_set, test_set = utld.load_data(dataset_path)

    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    # Check images and classes
    #i=50
    #print train_set_y[i].eval()
    #image =  disp_img(train_set_x.get_value()[i])
    #plt.imshow(image)
    #plt.show()

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('... building model')

    index = T.lscalar() # index to a [mini]batch

    x = T.matrix('x')
    y = T.ivector('y')

    classifier = nn.LogisticRegression(input=x, n_in=28*28, n_out=10)

    cost = classifier.negative_log_likelihood(y)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # gradients
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # # # # # # # #
    # Train Model: 
    # # # # # # # #
    print('... training model')
    
    # early-stopping parameters
    patience = 5000 # look as this many examples regardless
    patience_increase = 2
    improvement_threshold = 0.995 # a relative improvement of 
                                  # this much is considered significant
    valid_freq = min(n_train_batches, patience // 2)

    best_valid_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % valid_freq == 0:
                # compute validation
                valid_losses = [validate_model(i) 
                                for i in range(n_valid_batches)]
                this_valid_loss = np.mean(valid_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_valid_loss * 100.
                    )
                )

                if this_valid_loss < best_valid_loss:
                    if this_valid_loss < best_valid_loss * \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_valid_loss = this_valid_loss

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        (
                        'epoch %i, minibatch %i/%i, test error of, '
                        'best model %f %%' 
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save best model
                    with open('best_model_logreg.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = False
                break

    end_time = timeit.default_timer()

    print(
        (
        'Optimization complete with best validation score of %f %%,'
        'with test performance %f %%'
        ) %
        (
            best_valid_loss * 100.,
            test_score * 100.
        )
    )

    print(
        'Code run for %d epochs, with %.1f seconds.' % (
            epoch, (end_time - start_time))
    )


if __name__ == '__main__':
    optimize_logreg()