#!/usr/bin/env python

#import theano
#import theano.tensor as T
import numpy
import numpy as np
from collections import OrderedDict
import cPickle as pickle

#from theano import config
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import tensorflow as tf
import tflearn




def build_model( options  ):
   # trng=RandomStreams(1234)
    drop_ratio=options['drop_ratio']
    batch_size=options['batch_size']
    n_dim=options['n_dim']
    n_words=options[ 'n_words']
    n_emb=options['n_emb']
    n_image_feat=options[ 'n_image_feat'  ]
    n_common_feat=options[ 'n_common_feat' ]
    n_output=options[ 'n_output' ]
    n_attention=options[ 'n_attention' ]

    with tf.Graph().as_default():
    # get the transformed image features
        #h_0=tf.placeholder( shape=( None,n_dim ) , dtype=tf.float32  )
        #c_0=tf.placeholder( shape=( None,n_dim ) , dtype=tf.float32  )

        image_feat=tf.placeholder( shape=( None , n_image_feat  ) ,dtype=tf.float32  )

        max_length=tf.placeholder(tf.int32)
        # index of the input
        input_idx=tf.placeholder( shape=( None, None ) ,dtype=tf.int32)

        #label as input
        label=tf.placeholder( shape=batch_size,dtype=tf.int32)

        # input mask
        #input_mask=tf.placeholder( shape=(None,n_words) , dtype= tf.float32  )

        input_mask=tf.zeros_like( input_idx )

        # input embedding
        w_emb= tf.Variable(tf.random_uniform([n_words, n_emb], -1.0, 1.0),name="Embedding_Matrix")

        input_emb=w_emb[ input_idx ]

        if options['sent_drop']:
            input_emb=tflearn.dropout(input_emb,drop_ratio)

        h_encode=tflearn.lstm( input_emb , n_emb  )

        h_encode=h_encode[-1]  #-- check the dimension

        image_feat_down=tflearn.fully_connected( image_feat ,n_dim) # dim -- has to be figured )

        image_feat_attention_1=tflearn.fully_connected( image_feat_down , n_attention)   # dim -- has to be figured   )

        h_encode_attention_1=tflearn.fully_connected( h_encode , n_attention) # dim -- has to be figured  )

        combined_feat_attention_1 = image_feat_attention_1 + \
                                    h_encode_attention_1[:, None, :]

        if options['use_attention_drop']:
            combined_feat_attention_1 = tflearn.dropout(combined_feat_attention_1,drop_ratio)

        combined_feat_attention_1 = tflearn.fully_connected(combined_feat_attention_1 , 1) # dim -- to be determined  )
        prob_attention_1 = tf.nn.softmax(combined_feat_attention_1[:, :, 0])
        image_feat_ave_1 = (prob_attention_1[:, :, None] * image_feat_down).sum(axis=1)

        combined_hidden_1 = image_feat_ave_1 + h_encode


        # Second Layer Attention Model

        image_feat_attention_2 = tflearn.fully_connected(image_feat_down,  n_attention)  # dim -- to be determined  )
        h_encode_attention_2 = tflearn.fully_connected(combined_hidden_1 , n_attention)  # dim --to be dertermined  )


        combined_feat_attention_2 = image_feat_attention_2 + h_encode_attention_2[:, None, :]
        if options['use_attention_drop']:
            combined_feat_attention_2 = tflearn.dropout(combined_feat_attention_2,drop_ratio)

        combined_feat_attention_2 = fflayer(combined_feat_attention_2,1)              #dim -- to be determined)
        prob_attention_2 = tf.nn.softmax(combined_feat_attention_2[:, :, 0])

        image_feat_ave_2 = (prob_attention_2[:, :, None] * image_feat_down).sum(axis=1)

        if options.get('use_final_image_feat_only', False):
            combined_hidden = image_feat_ave_2 + h_encode
        else:
            combined_hidden = image_feat_ave_2 + combined_hidden_1

        for i in range(options['combined_num_mlp']):
            if options.get('combined_mlp_drop_%d'%(i), False):
                combined_hidden = tflearn.dropout(combined_hidden,drop_ratio)
            if i==0 and options[ 'combined_num_mlp'  ]==1:
                combined_hidden=tflearn.fully_connected( combined_hidden , n_output  )

            elif i==0 and options['combined_num_mlp'] != 1:
                combined_hidden = tflearn.fully_connected( combined_hidden , n_common_feat )

            elif i == options['combined_num_mlp'] - 1:
                combined_hidden = tflearn.fully_connected(combined_hidden, n_output) # dim -- to be determined )
            else:
                combined_hidden = tflearn.fully_connected(combined_hidden, n_common_feat)   #dim --to be determined)

        # drop the image output
        prob = tf.nn.softmax(combined_hidden)
        prob_y = prob[arange(prob.shape[0]), label]
        pred_label = tf.argmax(prob, axis=1)
        # sum or mean?
        cost = -tf.reduce_mean(tf.log(prob_y))
        accu = tf.reduce_mean(tf.eq(pred_label, label))

        return image_feat, input_idx, input_mask, \
            label, dropout, cost, accu



