'''
Author: Tao Tao, Xiaoxiao Yan
Data: Dec 17, 2018
'''

import numpy as np
import os, shutil, tarfile, pickle, imageio
from PIL import Image
from six.moves.urllib import request
# pip install googledrivedownloader
from google_drive_downloader import GoogleDriveDownloader as gdd

###############################################################################
# hyper-parameters
###############################################################################
LR_HR_RETIO = 4
IMSIZE = 256
LOAD_SIZE = int(IMSIZE * 76 / 64)
###############################################################################




    
###############################################################################
# download data
###############################################################################
# download Oxford-102 dataset and corresponding captions
class flower_downloader(object):
    def __init__(self, root='Data'):
        self.im_src = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
        self.im_dst = os.path.join(root, self.im_src.split('/')[-1])
        self.de_fileid = '0B3y_msrWZaXLaUc0UXpmcnhaVmM'
        self.de_dst = os.path.join(root, 'flowers.zip')
        self.data_dir = os.path.join(root, 'flowers')
        # download image & embedding
        if (self.im_src != None) and (self.im_dst != None):
            self.download_image()
        if (self.de_fileid != None) and (self.de_dst != None):
            self.download_description()
        # reorganize data directions
        self.reorganize_data()
    
    # download images from Oxford-102 dataset
    def download_image(self):
        if not os.path.isfile(self.im_dst):
            request.urlretrieve(url=self.im_src, filename=self.im_dst)
    
    # download captions for Oxford-102 dataset
    def download_description(self):
        if not os.path.isfile(self.de_dst):
            gdd.download_file_from_google_drive(file_id=self.de_fileid,
                                                dest_path=self.de_dst, unzip=True)
    
    # reorganize data directions
    def reorganize_data(self):
        datafile = self.im_dst.split('/')[-1]
        # copy & paste
        shutil.copyfile(self.im_dst, os.path.join(self.data_dir, datafile))
        # unpack .tgz file
        t = tarfile.open(os.path.join(self.data_dir, datafile))
        t.extractall(path=self.data_dir)  
        t.close() 
        # remove intermediate file
        os.remove(os.path.join(self.data_dir, datafile))
###############################################################################





###############################################################################
# pre-processing data
###############################################################################
class flower_convertor(object):
    def __init__(self, root='Data'):
        self.data_dir = root + '/flowers'
        self.low_size = int(LOAD_SIZE / LR_HR_RETIO)
        # deal with training set
        train_dir = os.path.join(self.data_dir, 'train')
        train_filenames = self.load_filenames(train_dir)
        self.save_data_list(train_dir, train_filenames)
        # deal with test set
        test_dir = os.path.join(self.data_dir, 'test')
        test_filenames = self.load_filenames(test_dir)
        self.save_data_list(test_dir, test_filenames)
    
    # load names of images in Oxford-102 dataset
    def load_filenames(self, direction):
        filepath = os.path.join(direction, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from:', filepath)
        print('Number of images:', len(filenames))
        return filenames
    
    # process one image
    def get_image(self, image_path, image_size):
        # read one image
        img = imageio.imread(image_path)
        # regulate image dimensionality
        if img.ndim == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = np.concatenate([img, img, img], axis=2)
        if img.shape[2] == 4:
            img = img[:, :, 0:3]
        # reshape one image
        transformed_image = Image.fromarray(img).resize((image_size, image_size), Image.BICUBIC)
        return np.array(transformed_image)
    
    # write images in Oxford-102 dataset
    def save_data_list(self, outpath, filenames):
        hr_images = []
        lr_images = []
        # generate data of images
        count = 0
        for key in filenames:
            f_name = os.path.join(self.data_dir, key+'.jpg')
            img = self.get_image(f_name, LOAD_SIZE)
            img = img.astype('uint8')
            hr_images.append(img)
            lr_img = np.array(Image.fromarray(img).resize((self.low_size, self.low_size), Image.BICUBIC))
            lr_images.append(lr_img)            
            count += 1
            if count % 1000 == 0:
                print('Loading %d...' % count)
        # write data of high-resolution images
        outfile = os.path.join(outpath, str(LOAD_SIZE)+'images.pickle')
        with open(outfile, 'wb') as f_out:
            pickle.dump(hr_images, f_out)
            print('Saved to: ', outfile)
        # write data of low-resolution images
        outfile = os.path.join(outpath, str(self.low_size)+'images.pickle')
        with open(outfile, 'wb') as f_out:
            pickle.dump(lr_images, f_out)
            print('Saved to: ', outfile)
###############################################################################





###############################################################################
# load pre-processed training data
###############################################################################
class flower_loader(object):
    def __init__(self, root='Data'):
        self.low_size = int(LOAD_SIZE / LR_HR_RETIO)
        self.data_dir = root + '/flowers'
        self.train_dir = self.data_dir + '/train'
        self.test_dir = self.data_dir + '/test'
        self.load_high_images()
        self.load_lwo_images()
        self.load_captions()
    
    def load_captions(self):
        train_caption_path = os.path.join(self.train_dir, 'char-CNN-RNN-embeddings.pickle')
        with open(train_caption_path, 'rb') as f:
            train_caption = pickle.load(f, encoding='latin1')
            train_caption = np.array(train_caption)
        test_caption_path = os.path.join(self.test_dir, 'char-CNN-RNN-embeddings.pickle')
        with open(test_caption_path, 'rb') as f:
            test_caption = pickle.load(f, encoding='latin1')
            test_caption = np.array(test_caption)
        self.captions = np.concatenate([train_caption, test_caption])
    
    def load_lwo_images(self):
        train_limage_path = os.path.join(self.train_dir,
                                         str(self.low_size)+'images.pickle')
        with open(train_limage_path, 'rb') as f:
            train_limage = pickle.load(f, encoding='latin1')
            train_limage = np.array(train_limage)
        test_limage_path = os.path.join(self.test_dir,
                                        str(self.low_size)+'images.pickle')
        with open(test_limage_path, 'rb') as f:
            test_limage = pickle.load(f, encoding='latin1')
            test_limage = np.array(test_limage)
        self.low_images = np.concatenate([train_limage, test_limage])
    
    def load_high_images(self):
        train_himage_path = os.path.join(self.train_dir,
                                         str(LOAD_SIZE)+'images.pickle')
        with open(train_himage_path, 'rb') as f:
            train_himage = pickle.load(f, encoding='latin1')
            train_himage = np.array(train_himage)
        test_himage_path = os.path.join(self.test_dir,
                                        str(LOAD_SIZE)+'images.pickle')
        with open(test_himage_path, 'rb') as f:
            test_himage = pickle.load(f, encoding='latin1')
            test_himage = np.array(test_himage)
        self.high_images = np.concatenate([train_himage, test_himage])
    
    def get_captions(self):
        return self.captions
    
    def get_low_images(self):
        return self.low_images
    
    def get_high_images(self):
        return self.high_images
###############################################################################





if __name__ == '__main__':
    # make a folder to store data
    root = 'Data'
    if not os.path.exists(root): 
        os.makedirs(root)
    
    # download data
    flower_downloader(root)
    
    # pre-process data
    flower_convertor(root)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    