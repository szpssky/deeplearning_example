# import matplotlib.pyplot as plt
import random
import numpy as np
import os
import sys
from scipy import ndimage
import pickle

image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load_letter(folder, min_num_images):
    tmps = os.listdir(folder)
    file_folders = []
    for tmp in tmps:
        file_folders.append(os.path.join("../../notMNIST_small/", tmp))

    print(file_folders)
    file_names = ['A','B','C','D','E','F','G','H','I','J']
    count_file=0
    for file_folder in file_folders:
        num_images = 0
        image_files = os.listdir(file_folder)
        set_filename = file_names[count_file]+'.pickle'
        count_file+=1
        print(file_folder)
        dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)

        for image in image_files:
            try:
                image_file = os.path.join(file_folder, image)
            # print(image_file)
                image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth

                if image_data.shape != (image_size, image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[num_images,:,:] = image_data
                num_images+=1
            except IOError as e:
                print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')


        dataset = dataset[0:num_images, :, :]
        print(dataset.shape)

        try:
            with open(set_filename, 'wb') as f:
                pickle.dump(dataset, f,pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', set_filename, ':', e)


def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

def merge_datasets(train_size, valid_size=0):
    pickle_files = ['A.pickle','B.pickle','C.pickle','D.pickle','E.pickle','F.pickle','G.pickle','H.pickle','I.pickle','J.pickle']

    train_dataset,train_labels =  make_arrays(train_size, image_size)

    tsize_per_class = train_size // 10
    start_t = 0
    end_t = tsize_per_class

    for label, pickle_file in enumerate(pickle_files):
        print('latbel %d' % label)
        try:
            with open(pickle_file,'rb') as f:
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                train_letter = letter_set[0:tsize_per_class, :, :]

                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label

                start_t += tsize_per_class
                end_t += tsize_per_class
                print(end_t)
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise


    train_dataset ,train_labels= randomize(train_dataset,train_labels)

    pickle_file_merge = 'notMNIST_Test.pickle'
    f = open(pickle_file_merge, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()

# merge_datasets(10000)


# load_letter("../../notMNIST_small", 1800)

# def plot_sample_dataset(dataset, labels, title):
#     plt.suptitle(title, fontsize=16, fontweight='bold')
#     items = random.sample(range(len(labels)), 12)
#     for i, item in enumerate(items):
#         plt.subplot(3, 4, i + 1)
#         plt.axis('off')
#         plt.title(chr(ord('A') + labels[item]))
#         plt.imshow(dataset[item])
#     plt.show()



