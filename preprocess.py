
import numpy as np
import os
from PIL import Image
from collections import defaultdict
from itertools import product
from sklearn.model_selection import train_test_split
import shutil
import re
import glob

width = 32
height = 32

posshiftshift_min = -5
posshiftshift_max = 5
scales = [0.9, 1.1]
rot_min = -15
rot_max = 15

dir = 'flickr_logos_27_dataset'
imgdir = os.path.join(dir, 'flickr_logos_27_dataset_images')
pp_dir = os.path.join(
    dir, 'processed')
annot = 'flickr_logos_27_dataset_training_set_annotation.txt'

def parse_annot(annot):
    fn = annot[0].decode('utf-8')
    class_name = annot[1].decode('utf-8')
    train_subset_class = annot[2].decode('utf-8')
    return fn, class_name, train_subset_class


def get_rect(annot):            # extarcting the x and y coordinates of the logo given in the txt file 
    rect = defaultdict(int)
    x1, y1, x2, y2 = rect_coord(annot[3:])
    cx, cy, wid, hgt = center_wid_hgt(x1, y1, x2, y2)
    rect['x1'] = x1
    rect['y1'] = y1
    rect['x2'] = x2
    rect['y2'] = y2
    rect['cx'] = cx
    rect['cy'] = cy
    rect['wid'] = wid
    rect['hgt'] = hgt
    return rect


def posshift(annot, im):   # Shifting position of the extracted logo.
    posshift_ims = []
    posshift_suffixes = []

    rect = get_rect(annot)
    for sx, sy in product(                        
            range(posshiftshift_min, posshiftshift_max),
            range(posshiftshift_min, posshiftshift_max)):
        cx = rect['cx'] + sx
        cy = rect['cy'] + sy
        cropped_im = im.crop((cx - rect['wid'] // 2, cy - rect['hgt'] // 2,
                              cx + rect['wid'] // 2, cy + rect['hgt'] // 2))
        resized_im = cropped_im.resize((width, height))
        posshift_ims.append(resized_im)
        posshift_suffixes.append('p' + str(sx) + str(sy))
        cropped_im.close()

    return posshift_ims, posshift_suffixes


def scale(annot, im):   # For resizing the extracted logo.
    scale_ims = []
    scale_suffixes = []

    rect = get_rect(annot)
    for s in scales:
        w = int(rect['wid'] * s)
        h = int(rect['hgt'] * s)
        cropped_im = im.crop((rect['cx'] - w // 2, rect['cy'] - h // 2,
                              rect['cx'] + w // 2, rect['cy'] + h // 2))
        resized_im = cropped_im.resize((width, height))
        scale_ims.append(resized_im)
        scale_suffixes.append('s' + str(s))
        cropped_im.close()

    return scale_ims, scale_suffixes


def rotate(annot, im):   # Rotating the extracted logo.
    rotate_ims = []
    rotate_suffixes = []

    rect = get_rect(annot)
    for r in range(rot_min, rot_max):
        rotated_im = im.rotate(r)
        cropped_im = rotated_im.crop(
            (rect['cx'] - rect['wid'] // 2, rect['cy'] - rect['hgt'] // 2,
             rect['cx'] + rect['wid'] // 2, rect['cy'] + rect['hgt'] // 2))
        resized_im = cropped_im.resize((width, height))
        rotate_ims.append(resized_im)
        rotate_suffixes.append('r' + str(r))
        rotated_im.close()
        cropped_im.close()

    return rotate_ims, rotate_suffixes


def crop(annot, im):                        #Cropping the logo
    x1, y1, x2, y2 = rect_coord(annot[3:])
    cropped_im = im.crop((x1, y1, x2, y2))
    cropped_im = cropped_im.resize((width, height))
    cropped_suffix = 'p00'
    return [cropped_im], [cropped_suffix]


def rect_coord(annot_part):
    return list(map(int, annot_part)) # apply on each list item and collect all the return values.  


def center_wid_hgt(x1, y1, x2, y2):
    cx = x1 + (x2 - x1) // 2
    cy = y1 + (y2 - y1) // 2
    wid = (x2 - x1)
    hgt = (y2 - y1)
    return cx, cy, wid, hgt


def is_skip(annot_part):
    x1, y1, x2, y2 = rect_coord(annot_part)
    _, _, wid, hgt = center_wid_hgt(x1, y1, x2, y2)
    if wid <= 0 or hgt <= 0:
        return True
    else:
        return False


def save_im(annot, cnt, *args):         # Saving the processed image.
    fn, class_name, train_subset_class = parse_annot(annot)
    dst_dir = os.path.join(pp_dir, class_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for i, arg in enumerate(args):
        for im, suffix in zip(arg[0], arg[1]):
            save_fn = '_'.join([
                fn.split('.')[0], class_name, train_subset_class, str(cnt),
                suffix
            ]) + os.path.splitext(fn)[1]
            im.save(os.path.join(dst_dir, save_fn))


def close_im(*args):
    for ims in args:
        for im in ims:
            im.close()


def crop_and_aug(annot_train):     # parent function which call all the sub functions.
    cnt_per_file = defaultdict(int)
    for annot in annot_train:
        # for generating a file name
        fn, _, _ = parse_annot(annot)
        cnt_per_file[fn] += 1

        # skip if width or height equal zero
        if is_skip(annot[3:]):
            print('Skip: ', fn)
            continue

        # open an image
        im = Image.open(os.path.join(imgdir, fn))

        # normal cropping
        cropped_ims, cropped_suffixes = crop(annot, im)

        # augment by shifting a center
        shifted_ims, shifted_suffixes = posshift(annot, im)

        # augment by scaling
        scaled_ims, scaled_suffixes = scale(annot, im)

        # augment by rotation
        rotated_ims, rotated_suffixes = rotate(annot, im)

        # save images
        save_im(annot, cnt_per_file[fn], [cropped_ims, cropped_suffixes],
                [shifted_ims, shifted_suffixes], [scaled_ims, scaled_suffixes],
                [rotated_ims, rotated_suffixes])

        # close image file
        close_im([im], cropped_ims, shifted_ims, scaled_ims, rotated_ims)

def crop_and_aug_with_none(annot_train, with_none=False):
    # root directory to save processed images
    if not os.path.exists(pp_dir):
        os.makedirs(pp_dir)

    # crop images and apply augmentation
    crop_and_aug(annot_train)

    # print results
    org_imgs = [img for img in os.listdir(imgdir)]
    crop_and_aug_imgs = [
        fname
        for root, dirs, files in os.walk(pp_dir)
        for fname in glob.glob(os.path.join(root, '*.jpg'))  # look for the file with .jpg extension.
    ]
    print('original: %d' % (len(org_imgs)))
    print('cropped: %d' % (len(crop_and_aug_imgs)))          # prints number of images cropped.


def do_train_test_split():                                   # For spliting dataset into 2 directories Train and Test.
    class_names = [cls for cls in os.listdir(pp_dir)]
    for class_name in class_names:                           # create directories under a particular class name.
        if os.path.exists(                                  
                os.path.join(pp_dir, class_name, 'train')):
            continue
        if os.path.exists(
                os.path.join(pp_dir, class_name, 'test')):
            continue

        imgs = [
            img
            for img in os.listdir(
                os.path.join(pp_dir, class_name))
        ]
        # train=0.75, test=0.25
        train_imgs, test_imgs = train_test_split(imgs)
        # move images to train or test directory
        os.makedirs(os.path.join(pp_dir, class_name, 'train'))         # create directories
        os.makedirs(os.path.join(pp_dir, class_name, 'test'))
        for img in train_imgs:
            dst = os.path.join(pp_dir, class_name, 'train')
            src = os.path.join(pp_dir, class_name, img)
            shutil.move(src, dst)                                      # moving image into that directory
        for img in test_imgs:
            dst = os.path.join(pp_dir, class_name, 'test')
            src = os.path.join(pp_dir, class_name, img)
            shutil.move(src, dst)                                      # moving image into that directory


def main():
    annot_train = np.loadtxt(os.path.join(dir, annot), dtype='a')
    print('train_annotation: %d, %d ' % (annot_train.shape))    # prints annotation text file info.       

    # cropping and data augmentation
    crop_and_aug_with_none(annot_train)

    # train_test_split
    do_train_test_split()


if __name__ == '__main__':
    main()
