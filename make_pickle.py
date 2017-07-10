import numpy as np
import os
from scipy import ndimage
from six.moves import cPickle as pickle
import re
import clas

width = 32     # Width of the image
height = 32    # Height of the image
channel = 3         # Number of channels
pix_val = 255.0
dir = 'flickr_logos_27_dataset'
pp_dir = os.path.join(
    dir, 'processed')               # Directory where processed images are stored
pickle_file = 'logo_dataset.pickle'   # Name of the pickle file 

train_size = 70000  # Number of images you want to store as training dataset in pickle file from the processed images
val_size = 5000
test_size = 7000    # Number of images you want to store as test dataset in pickle file from the processed images


def load_logo(data_dir):   # Opening the image 
    image_files = os.listdir(data_dir)      
    dataset = np.ndarray(
        shape=(len(image_files), height, width, channel),
        dtype=np.float32)
    print(data_dir)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(data_dir, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -        
                          pix_val / 2) / pix_val
            if image_data.shape != (height, width, channel):
                raise Exception('Unexpected image shape: %s' %
                                str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e,
                  '-it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]                           
    print('Full dataset tensor:', dataset.shape)       # Tell processed number of images for a particular class 
    print('Mean:', np.mean(dataset))                   # Calculate mean over that entire class
    print('Standard deviation:', np.std(dataset))      # Calculate standard deviation over that entire class
    return dataset


def pickling(data_dirs, force=False):     # Creating Pickle file
    dataset_names = []
    for dir in data_dirs:
        set_filename = dir + '.pickle'
        dataset_names.append(set_filename)

         # You may use and not force if you dont want to recreate already created pickle file
        if os.path.exists(set_filename) and force:    
    
            print('%s already present - Skipping pickling. ' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_logo(dir)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names

# Creates array of dataset

def array(nb_rows, image_width, image_height, image_ch=1):     
    if nb_rows:
        dataset = np.ndarray(                               #  stores its height, width and channel into an array
            (nb_rows, image_height, image_width, image_ch), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)        #  stores its labels
    else:
        dataset, labels = None, None
    return dataset, labels


def combine(pickle_files, train_size, val_size=0):    # Merging pickle files of all the classes into one pickle file. 
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = array(val_size, width,
                                              height, channel)
    train_dataset, train_labels = array(train_size, width,
                                              height, channel)
    vsize_per_class = val_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                logo_set = pickle.load(f)
                np.random.shuffle(logo_set)
                if valid_dataset is not None:
                    valid_logo = logo_set[:vsize_per_class, :, :, :]
                    valid_dataset[start_v:end_v, :, :, :] = valid_logo
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                train_logo = logo_set[vsize_per_class:end_l, :, :, :]
                train_dataset[start_t:end_t, :, :, :] = train_logo
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    return valid_dataset, valid_labels, train_dataset, train_labels


def makepickle(train_dataset, train_labels, valid_dataset, valid_labels,   
                test_dataset, test_labels):
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)      # Saving data of the images into a pickle file
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def main():  
    dirs = [
        os.path.join(pp_dir, class_name, 'train')      # Look into all the train folder of the class
        for class_name in clas.CLASS_NAME
    ]
    test_dirs = [
        os.path.join(pp_dir, class_name, 'test')        # Look into all the test folder of the class
        for class_name in clas.CLASS_NAME
    ]

    train_datasets = pickling(dirs)
    test_datasets = pickling(test_dirs)

    valid_dataset, valid_labels, train_dataset, train_labels = combine(   # function called for merging
        train_datasets, train_size, val_size)
    _, _, test_dataset, test_labels = combine(test_datasets, test_size)

    train_dataset, train_labels = randomize(train_dataset, train_labels)   # function called for randomizing
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)

    makepickle(train_dataset, train_labels, valid_dataset, valid_labels,   # function called for making a pickle file.
                test_dataset, test_labels)
    statinfo = os.stat(pickle_file)                         # Shows size of the file 
    print('Compressed pickle size:', statinfo.st_size)


if __name__ == '__main__':
    main()
