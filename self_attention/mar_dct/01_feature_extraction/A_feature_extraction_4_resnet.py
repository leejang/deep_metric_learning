#!/usr/bin/env python
import os, glob, sys
import caffe
import numpy as np
#import leveldb
from utils import *
from sklearn.preprocessing import normalize
import xml.etree.ElementTree as ET
#from check_computed import *

log= set_logger("Feature_Extraction", level_logger="info", format_logger="%(asctime)s- %(name)s - %(levelname)s - %(message)s")

class Local_Extractor(caffe.Net):
    """
    Class to perform feature extraction from an specific layer of a CNN network
    """
    def __init__(self, model_file, pretrained_file, mean_value=None,
        layer=['pool5'], input_size = None ):

        caffe.set_mode_gpu()
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # get name input layer
        self.list_layers = layer
        self.mean_value = mean_value

        # set transformer object
        self.transformer = caffe.io.Transformer({'data': self.blobs['data'].data.shape})
        self.transformer.set_transpose( 'data', (2,0,1) )

        if mean_value is not None:
            self.transformer.set_mean('data', mean_value)

        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2,1,0))

        """
        if input_size is not None:
            #reshape the input
            print "New input! {}".format(input_size)
            self.reshape_input( input_size[0], input_size[1],  input_size[2], input_size[3]  )
        """

    def reshape_input(self, batch, channels, dx, dy):
        self.blobs['data'].reshape(batch, channels, dx, dy)
        self.reshape()
        # reset the pre-processing
        self.transformer = caffe.io.Transformer({'data': self.blobs['data'].data.shape})
        self.transformer.set_transpose( 'data', (2,0,1) )

        if self.mean_value is not None:
            self.transformer.set_mean('data', self.mean_value)
            self.transformer.set_raw_scale('data', 255)
            self.transformer.set_channel_swap('data', (2,1,0))

    def extract(self, batch_images ):
        """
        Extract features from selected layers
        Return a dictionary with local activations
        Key values are the selected layers
        """
        n_images = len(batch_images)

        res = {}
        # preprocess images
        for k, ima in enumerate(batch_images):
            self.blobs['data'].data[k,...] = self.transformer.preprocess('data', ima)

        # fordward image through the net
        self.forward()

        # extract features from list of layers
        for layer in self.list_layers:
            feats = self.blobs[ layer ].data.copy().astype(np.float32)
            # features for input images (just in case the batch contains less than the
            # number of images expected by the network
            res[layer]=feats[:n_images,...]

        return res

    # Same thing we do in the reader!
    def get_feature_maps_single_image(self, image, new_dim=None, interpolation=1):
        # maps for a single image
        maps=self.extract([image])

        # check dimensions
        if image.shape[0]<image.shape[1] and new_dim is not None:
            new_dim=(new_dim[1], new_dim[0])

        # default dim-first layer
        dim=maps[self.list_layers[0]]
        for layer in self.list_layers:
            # RESHAPING MAPS -----------------------------------------------------------------------
            # if we've set new dimensions...
            if new_dim is not None and new_dim!=dim[1:]:
                log.debug("New DIM: Interpolating {}, {}, {}".format(interpolation, maps[layer].shape, new_dim))
                #RESHAPE -- check reshape_maps / reshape_maps_zoom
                t0 = time.time()
                if interpolation == 0:
                    maps[layer] = reshape_maps(maps[layer].squeeze(), new_dim)
                elif interpolation==1 or interpolation==2:
                    maps[layer] = reshape_maps_zoom(maps[layer].squeeze(), new_dim, interpolation)
                t1 = time.time()

                # else, set all layers to the same dim! (first one)
            else:
                # reshape rest of layers to the first layer dim in the list
                new_dim = maps[self.list_layers[0]].shape[1:]
                #RESHAPE -- check reshape_maps / reshape_maps_zoom
                t0 = time.time()
                if maps[layer].shape[1:] !=new_dim:
                    log.debug("Default DIM: Interpolating {} ".format(interpolation))
                    if interpolation == 0:
                        maps[layer] = reshape_maps(maps[layer].squeeze(), new_dim)
                    elif interpolation==1 or interpolation==2:
                        print new_dim
                        maps[layer] = reshape_maps_zoom(maps[layer].squeeze(), new_dim, interpolation)
                t1 = time.time()
            log.debug("Layer {} ; {}; elapsed {}".format( layer, maps[layer].shape, t1-t0 ))

        # we return all features resized to an unique dimension (so we can contatenate them :-) )
        return maps

    def get_features_single_image(self, image, new_dim=None, interpolation=1, pca_model=None):
        '''
        Reshape the features into (n_samples, n_dim) and apply l2-normalization
        and concatenate layers (increase n_dim)
        '''

        #dictionary with maps per layer - UNIQUE SIZE
        t0 = time.time()
        maps = self.get_feature_maps_single_image( image, new_dim=new_dim, interpolation=interpolation )
        t1 = time.time()
        log.debug("maps computed! : elapsed {}".format( t1-t0))

        # If we only have one layer to process...
        #extract info of the first layer in the list
        dim = maps[self.list_layers[0]].shape
        t0=time.time()
        maps[self.list_layers[0]] = np.swapaxes(maps[self.list_layers[0]], 0,1).swapaxes(1,2)
        features = np.reshape(maps[self.list_layers[0]], (dim[1]*dim[2], dim[0]))
        t1=time.time()
        log.debug("reshaping n_samples, n_dim : elapsed {}".format( t1-t0 ))


        #if we read more than one layer then concatenate activations
        if len( self.list_layers )>1:
            for layer in self.list_layers[1:]:
                maps[layer] = np.swapaxes(maps[layer], 0,1).swapaxes(1,2)
                features_ = np.reshape(maps[layer], (dim[1]*dim[2], dim[0]))
                features = np.concatenate( (features, features_), axis=1 )
                log.debug("Concatenation done! {}".format(features.shape[1]))

        # normalize features if requiered
        t0=time.time()
        features = normalize(features)
        t1=time.time()
        log.debug("feats normalized! : elapsed {}".format( t1-t0) )

        # apply PCA for dim reduction
        if pca_model!=None:
            t0=time.time()
            features = pca_model.transform(features)
            t1=time.time()
            #re-normalize features!
            features = normalize(features)
            log.debug("feats with PCA! : elapsed {}".format( t1-t0) )

        log.debug("Dimensions features {}".format(features.shape) )
        return features


    @property
    def batch_size(self):
        """
        Max number of images that the network process
        """
        return self.blobs['data'].data.shape[0]

    @property
    def channels(self):
        """
        Number of channels
        """
        return self.blobs['data'].data.shape[1]

    @property
    def input_shape(self):
        """
        Dimensions of the networ input
        """
        return self.blobs['data'].data.shape[1:]

    @property
    def layer_dimensions(self):
        """
        Return dictionary with layer dimensions
        """
        dic_dim = {}
        for layer in self.blobs.keys():
            dic_dim[layer] = self.blobs[layer].data.shape[1:]
        return dic_dim


    @classmethod
    def from_settings(cls, settings):
        #init class from a settings json objec

        if "input_size" in settings.keys():
            in_size = settings["input_size"]
        else:
            in_size=None

        network_name = str(settings["Feature_extractor"])
        list_layers = np.array(settings["Layer_output"]).astype(str)
        prototxt = str(settings["Feature_extractors"][network_name]["prototxt"])
        caffemodel = str(settings["Feature_extractors"][network_name]["caffemodel"])

        meanValue = settings["Feature_extractors"][network_name]["meanValue"]

        if type(meanValue) is not list:
            # load image mean for the second network
            print ("Loading image mean for the network")

            image_mean_bytes = open(meanValue, 'rb').read()

            # noinspection PyUnresolvedReferences
            blob = caffe.proto.caffe_pb2.BlobProto()
            blob.ParseFromString(image_mean_bytes)
            a = np.array(caffe.io.blobproto_to_array(blob))
            assert a.shape[0] == 1, \
                "Input image mean blob protobuf consisted of more than one " \
                "image. Not sure how to handle this yet."
            a = a.reshape(a.shape[1:])
            print ("Loaded image mean from protobuf bytes")

            assert a.shape[0] in [1, 3], \
                "Currently asserting that we either get 1 or 3 channel images. " \
                "Got a %d channel image." % a[0]

            # TODO: Instead of always using pixel mean, try to use image-mean if
            #       given. Might have to rescale if image/data layer shape is
            #       different.
            a_mean = a.mean(1).mean(1)
            print ("Initializing data transformer for the network-- mean")
            meanValue = a_mean

            """
            #compute average from the average image
            meanValue = np.load( str(meanValue) )
            val1 = np.average( meanValue[0,:,:], )
            val2 = np.average( meanValue[1,:,:], )
            val3 = np.average( meanValue[2,:,:], )
            meanValue = np.array([val1, val2, val3])
            """
        else:
            meanValue = np.array(meanValue)

        return cls(prototxt, caffemodel, mean_value=meanValue, layer=list_layers, input_size = in_size)

    @classmethod
    def from_settings_for_ranking(cls, settings, network_name, layers, in_size):
        # init class from a settings json objecr

        prototxt = str(settings["Feature_extractors"][network_name]["prototxt"])
        caffemodel = str(settings["Feature_extractors"][network_name]["caffemodel"])

        meanValue = settings["Feature_extractors"][network_name]["meanValue"]
        if type(meanValue) is not list:
            #compute average from the average image
            meanValue = np.load( str(meanValue) )
            val1 = np.average( meanValue[0,:,:], )
            val2 = np.average( meanValue[1,:,:], )
            val3 = np.average( meanValue[2,:,:], )
            meanValue = np.array([val1, val2, val3])
        else:
            meanValue = np.array(meanValue)

        return cls(prototxt, caffemodel, mean_value=meanValue, layer=layers, input_size = (1,3, in_size[2], in_size[3]))


class LeveldbStorer():
    """
    Class to store Features of each layer in a leveldb format.

    Need to read [main feature path] from json.
    Keys are images filenames and values are parsed to strings from the the
    numpy array with the original network size (np.float32 values!)
    """

    def __init__(self, path_local_features, list_layers):
        self.db_dict = {}

        #check if the path exists
        if not os.path.exists(path_local_features):
            os.makedirs(path_local_features)

        self.path = path_local_features

        for layer in list_layers:
            self.db_dict[layer] = leveldb.LevelDB( os.path.join(self.path,layer+"_db") )

    def write_batch(self, keyframes, dic_feat):

        for layer in self.db_dict.keys():
            batch = leveldb.WriteBatch()
            for keyframe, v in zip(keyframes, dic_feat[layer]):
                log.debug("extracting layer {} shape {} key db {}".format(layer, v.shape, os.path.basename(keyframe) ))

                #store_dimensions for reading
                dim_str = "@"
                for d in v.shape:
                    dim_str+=str(d)+"@"

                log.debug( "check key {}".format(os.path.basename(keyframe)+dim_str) )
                batch.Put( os.path.basename(keyframe)+dim_str, v.tostring() )
            self.db_dict[layer].Write(batch, sync=True)
            print "layer {} - {}".format(layer, dim_str)

    @classmethod
    def from_settings(cls, settings):
        dim_input=settings["input_size"]
        network=str(settings["Feature_extractor"])
        pathDB=os.path.join( str(settings["featuresDB"]),network, str(dim_input[2])+"_"+str(dim_input[3])  )
        return cls( pathDB, settings["Layer_output"] )


def extract_features( fx, key_frames, storer ):
    """
    Main loop -  Made into a function to allow processing vertical and
    horitzontal images indepently.
    """
    # loop over images
    status = ProcessingStatus( fx.batch_size, len(key_frames) )
    log.info(status.start())

    for i in range( 0, len(key_frames), fx.batch_size ):
        if i+fx.batch_size > len(key_frames):
            log.debug("last batch!")
            images = load_batch_images( key_frames[i:] )
            dic_feats = fx.extract(images)
            storer.write_batch( key_frames[i:], dic_feats  )

            log.info("{} last images.".format(len(key_frames[i:])))

        else:
            images = load_batch_images( key_frames[i:i+fx.batch_size] )
            dic_feats = fx.extract(images)
            storer.write_batch( key_frames[i:], dic_feats  )

            log.info(status.update())


def get_img_fname_n_bbxes(txt_fp):

    with open(txt_fp, 'r') as txt_f:
        lines = txt_f.readlines()
        # remove whitespace
        lines = [x.strip() for x in lines]

    all_info = lines[0].split()

    img_fname = all_info[0]

    bbxes = [x for x in all_info[1:]]

    return img_fname, bbxes

"""
# Gallery Images
if __name__ == "__main__":

    img2vid_txt_fp = 'img2vid.txt'
    with open(img2vid_txt_fp) as f:
        content = f.readlines()
    content = [x.strip().split() for x in content]
    img2vid_array = np.array(content)

    img_dic = {}
    for img in img2vid_array:
        img_name = img[0]
        v_id = img[1]

        if img_name in img_dic:
            img_dic[img_name].append(v_id)
        else:
            img_dic[img_name] = [v_id]

    # to save file and cls names
    save_fp = '/home/smqtk/img_names_and_cls_gallery.txt'
    sf = open(save_fp, 'w')

    # veid dataset
    settings = load_settings("/home/smqtk/4_veid/settings_resnet.json")

    # image path
    img_path = '/home/smqtk/gallery/*.jpg'

    # Feature extractor - [Image Dimensions fixed to 1/3 in the prototxt]
    fx = Local_Extractor.from_settings(settings)

    Y_train = []
    label_train = []

    for img in sorted(glob.glob(img_path)):

        img_name = os.path.basename(img)
        cls = img_dic[img_name[:-4]]

        print (img_name, cls)
        sf.write("%s, %s\n" %(img_name, cls))
        image = load_batch_images([img])
        dic_feats = fx.extract(image)
        img_feat = dic_feats['pool5'].flatten()

        # to save feature vectors with label
        Y_train.append(img_feat)
        label_train.append(cls)

    # save all
    np.save('/home/smqtk/Y_train', Y_train)
    np.save('/home/smqtk/label_train', label_train)

    sf.close()

    print ("Done!")
"""

# Query images
if __name__ == "__main__":

    # to save file and cls names
    #save_fp = '/home/smqtk/img_names_and_cls_train.txt'
    save_fp = '/home/smqtk/img_names_and_cls_test.txt'
    sf = open(save_fp, 'w')

    # veid dataset
    settings = load_settings("/home/smqtk/01_feature_extraction/settings_resnet.json")

    # Feature extractor - [Image Dimensions fixed to 1/3 in the prototxt]
    fx = Local_Extractor.from_settings(settings)

    #Y_train = []
    #label_train = []
    Y_test = []
    label_test = []

    # root_dir
    #root_dir = '/home/smqtk/sc5_w_id_crop_ag/Train'
    root_dir = '/home/smqtk/sc5_w_id_crop_ag/Test'

    for root, subdirs, files in os.walk(root_dir):

        #print ('root, subdirs, files: ' + root, subdirs, files)
        for f in files:
            if f.endswith('.jpg'):
                print ('root = ' + root)
                annot = root.strip().split('/')
                cls = annot[-1]

                img = root + '/' + f
                print ('img = ' + img)

                # check image name and class
                print (f, cls)
                sf.write("%s, %s\n" %(f, cls))
 
                image = load_batch_images([img])
                dic_feats = fx.extract(image)
                img_feat = dic_feats['res5c']
        
                # to save feature vectors with label
                #Y_train.append(img_feat)
                #label_train.append(cls)
                Y_test.append(img_feat)
                label_test.append(cls)

    # save all
    #np.save('/home/smqtk/Y_train', Y_train)
    #np.save('/home/smqtk/label_train', label_train)
    np.save('/home/smqtk/Y_test', Y_test)
    np.save('/home/smqtk/label_test', label_test)

    print ("Done!")
