#!/usr/bin/env python3
import os
import tensorflow as tf
import numpy as np
import logger

args = tf.app.flags
args.DEFINE_string('ckpath',        None, 'Checkpoint path saving embedding variable value. (ex: model.ckpt-100) [%(default)s]')
args.DEFINE_string('config_pbtxt',  'projector_config.pbtxt', 'protobuf file which recorded linking information. [%(default)s]')
args.DEFINE_string('tsv_file',      'embedding_var.tsv', 'TSV file to write embedding variable value. [%(default)s]')
FLAGS = args.FLAGS

def main(_):
    try:
        tf.train.import_meta_graph(FLAGS.ckpath + '.meta')
        folder = os.path.split(FLAGS.ckpath)[0]
        config_pbtxt = os.path.join(folder, FLAGS.config_pbtxt)
        tsv_file = os.path.join(folder, FLAGS.tsv_file)
        tensor_name = "image_embedding:0"
        embedding_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, tensor_name)[0]
        with tf.Session() as sess:
            sess.run(embedding_var.initializer)
            embedding_np = sess.run(embedding_var)
        np.savetxt(tsv_file, embedding_np, delimiter='\t')
    except TypeError:
        logger.logger.error('Cannot find "{}"'.format(FLAGS.ckpath))

if __name__ == '__main__':
    tf.app.run()
