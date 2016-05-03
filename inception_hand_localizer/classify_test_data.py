import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

graph_session = tf.Session()
graph_def = tf.GraphDef()
graph_def.ParseFromString(gfile.FastGFile('tmp/output_graph.pb', 'rb').read())
graph_session.graph.as_default()
tf.import_graph_def(graph_def, name='')
softmax_tensor = graph_session.graph.get_tensor_by_name('final_result:0')

INPUT_LAYER_NAME = 'DecodeJpeg/contents:0'

def process_image(gray_image):
    predictions = graph_session.run(softmax_tensor, {INPUT_LAYER_NAME: gray_image})
    normalized_predictions = np.squeeze(predictions)
    #normalized_predictions = [normalized_predictions[0], normalized_predictions[1] + normalized_predictions[2]]
    best = np.argmax(normalized_predictions)
    return best, normalized_predictions[best]

if __name__ == '__main__':
    from os import listdir
    root_data_dir = 'data/cropped_test_images/'
    out_file = open('new_results.txt', 'a')
    image_files = listdir(root_data_dir)
    

    classifications = []

    for image_file in image_files:
        print image_file
        image_data = gfile.FastGFile(root_data_dir + image_file, 'rb').read()
        results = process_image(image_data)
        print results
        out, prob = results
        print prob
        classifications.append((image_file.replace('.jpg',''), out, prob) )
    

    classifications = sorted(classifications, key=lambda x: x[0])
    for c in classifications:
        img, out, prob = c
        if out == 0:
            prob = 1-prob
        out_file.write('%s\t%f\n'%(img, prob))
        
    out_file.flush()
    out_file.close()

