try:
   import cPickle as pickle
except:
   import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

def load_data(dataset_path):
    """Load the dataset."""

    with open(dataset_path, 'rb') as data_file:
        try:
            train_set, valid_set, test_set = pickle.load(data_file, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(data_file)
    
    def shared_dataset(data_xy, borrow=True):
        """Load dataset into shared variables."""

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, 
                                                dtype=theano.config.floatX),
                                borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                                dtype=theano.config.floatX),
                                borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')
    
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval


def disp_img(image_array, img_size=28):
    """Display image with image data array."""

    image = np.zeros((img_size, img_size), dtype=np.float32)
    for i in range(img_size):
        image[i,:] = image_array[i * img_size : (i + 1) * img_size]
    return image
   
   
def predict():
    """Predict labels."""

    classifier = pickle.load(open('best_model.pkl'))

    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred
    )

    datasets = load_data(dataset_path)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])

    print('predicted values are:')
    print(predicted_values)