# Logo Detection
This is a Brand logo classification done by neural networks based on TFLeanrn and python. The dataset is taken from flickr 27 logo dataset.


### Prerequisites

* Python
* TFlearn
* numpy+mkl 
* scipy
* scikitlearn 
* <a href=" https://www.tensorflow.org/install/ ">Tensor flow</a>

*    You can find most of the Python libraries <a href=" http://www.lfd.uci.edu/~gohlke/pythonlibs ">Here</a>.


### How to use

Inorder to train the neural nets and classify the data set.

1. ``` python preprocess.py ```, it does the image processing and splits the data into train and test sets.

2. ``` python make_pickle.py ```, to index the data into a pickle file which contains features of the logo such as width, height and no. of channel.

3. As now we have the pickle file, we can train the neural nets to do this ```python train_cnn.py``` has to be executed.

4. ``` python testing.py ```, to test images run 


## Additional information.

* To achieve high accuracy the number of epochs has to be adjusted according to the training dataset size.
* The batch size should not be kept too small or too large.
* The size of a layer of neural network can be adjusted according to the processing power available.
* Traning accuracy between 0.95 to 0.99 is achieved and validation i.e test accuracy between 0.97 to 0.99 is achieved.







