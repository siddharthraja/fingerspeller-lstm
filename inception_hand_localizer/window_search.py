import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import base64
from os import listdir
import numpy as np

INPUT_LAYER_NAME = 'DecodeJpeg/contents:0'
SOFTMAX_LAYER_NAME = 'final_result:0'
GRAPH_PROTO_LOCATION = 'tmp/output_graph.pb'
OUTPUT_LAYER = 'final_matmul:0'

similarity_session = tf.Session()
graph_def = tf.GraphDef()
graph_def.ParseFromString(gfile.FastGFile(GRAPH_PROTO_LOCATION, 'rb').read())
similarity_session.graph.as_default()
tf.import_graph_def(graph_def, name='')
similarity_tensor = similarity_session.graph.get_tensor_by_name(SOFTMAX_LAYER_NAME)


def print_ops():
    operations = similarity_session.graph.get_operations()

    for operation in operations:
        print 'operation: %s' % str(operation.name)
        for k in operation.outputs:
            print operation.name, 'output ', k.name
        print '\n'


def image_features(image_file):
    image_data = gfile.FastGFile(image_file, 'rb').read()
    output = similarity_session.run(similarity_tensor, {INPUT_LAYER_NAME:image_data})
    return output


def image_window_search(image):

    best_prob = 0
    best_crop = []
    best_r, best_c = 0, 0

    for i in range(len(image)/64):
        for j in range(len(image[0])/64):
          
            r = i*64
            c = j*64

            crop = image[r:r+128, c:c+128]
            imsave(temp_image, crop)

            output = image_features(temp_image)
            probs = output[0]
            pos_prob = probs[0]

            if pos_prob > best_prob:
                best_crop = crop
                best_prob = pos_prob
                best_r, best_c = r, c

    crop = best_crop
    crop2 = image[best_r-128:best_r + 128, best_c-128:best_c + 128]
    
    return crop, crop2, best_r, best_c

if __name__ == '__main__':

    print_ops()

    from skimage.io import imread,imsave

    temp_image = 'temp.jpg'

    best_crop = []
    best_prob = 0

    best_r, best_c = 0,0

    root_data_dir = 'input_data/'
    
    out_dir = 'normalized_hands/'

    words = listdir(root_data_dir)


    for word in words:
        sessions = listdir(root_data_dir + word)

        for session in sessions:
            images = listdir(root_data_dir + word+ '/' + session )

            for name in images:
                print name

                image_file = root_data_dir + word + '/' + session + '/' + name

                image = imread(image_file)
                image = image[len(image)/2:, len(image)/2:]

                crop, crop2, best_r, best_c = image_window_search(image)

                imsave(out_dir + word+'_'+session+'_'+str(best_r) +'_'+ str(best_c)+'_' +name, crop )
                imsave(out_dir + word+'_'+session+'_'+str(best_r) +'_'+ str(best_c) +'_secondbest_' +name, crop2 )


