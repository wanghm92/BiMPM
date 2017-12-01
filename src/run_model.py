# -*- coding: utf-8 -*-
from __future__ import print_function
from vocab_utils import Vocab
from SentenceMatchModelGraph import SentenceMatchModelGraph
import namespace_utils, SentenceMatchTrainer
import argparse, time
import tensorflow as tf

FLAGS = None

vocabs = {}
mode = None
out_path = None

sess = None
valid_graph = None
testDataStream = None

tf.logging.set_verbosity(tf.logging.ERROR)  # DEBUG, INFO, WARN, ERROR, and FATAL

def init():
    '''
    Use this function to do all the initialization.
    After this, the model should be already loaded to memory and ready to predict
    Use the predifined arguments by calling FLAGS.model_path as an example

    This function shouldn't return anything. Use global variables if necessary
    '''

    global vocabs
    global testDataStream
    global valid_graph
    global sess
    global mode
    global FLAGS
    global out_path

    # Define all the command line arguments (e.g. model path, num of threads, etc) here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='the path to the test file.')
    parser.add_argument('--out_path', type=str, required=True, help='The path to the output file.')
    parser.add_argument('--word_vec_path', type=str, required=True, help='word embedding file for the input file.')
    parser.add_argument('--mode', type=str, default="prediction", help='prediction or probs')

    args, unparsed = parser.parse_known_args()

    model_prefix = args.model_prefix
    in_path = args.in_path
    out_path = args.out_path
    word_vec_path = args.word_vec_path

    mode = args.mode

    # print('Loading configurations.')
    # start_time = time.time()

    # load the configuration file
    FLAGS = namespace_utils.load_namespace(model_prefix + ".config.json")

    # time_used = time.time() - start_time
    # print('Time used for loading configs = %ds'%time_used)

    with_POS=False
    if hasattr(FLAGS, 'with_POS'): with_POS = FLAGS.with_POS
    with_NER=False
    if hasattr(FLAGS, 'with_NER'): with_NER = FLAGS.with_NER
    wo_char = False
    if hasattr(FLAGS, 'wo_char'): wo_char = FLAGS.wo_char

    wo_left_match = False
    if hasattr(FLAGS, 'wo_left_match'): wo_left_match = FLAGS.wo_left_match

    wo_right_match = False
    if hasattr(FLAGS, 'wo_right_match'): wo_right_match = FLAGS.wo_right_match

    wo_full_match = False
    if hasattr(FLAGS, 'wo_full_match'): wo_full_match = FLAGS.wo_full_match

    wo_maxpool_match = False
    if hasattr(FLAGS, 'wo_maxpool_match'): wo_maxpool_match = FLAGS.wo_maxpool_match

    wo_attentive_match = False
    if hasattr(FLAGS, 'wo_attentive_match'): wo_attentive_match = FLAGS.wo_attentive_match

    wo_max_attentive_match = False
    if hasattr(FLAGS, 'wo_max_attentive_match'): wo_max_attentive_match = FLAGS.wo_max_attentive_match

    # print('Loading vocabs.')
    # start_time = time.time()

    # load vocabs
    vocabs['word'] = Vocab(word_vec_path, fileformat='txt3')
    vocabs['label'] = Vocab(model_prefix + ".label_vocab", fileformat='txt2')
    # print('word_vocab: {}'.format(vocabs['word'].word_vecs.shape))
    # print('label_vocab: {}'.format(vocabs['label'].word_vecs.shape))
    num_classes = vocabs['label'].size()

    vocabs['pos'] = None
    vocabs['ner'] = None
    if with_POS: vocabs['pos'] = Vocab(model_prefix + ".POS_vocab", fileformat='txt2')
    if with_NER: vocabs['ner'] = Vocab(model_prefix + ".NER_vocab", fileformat='txt2')
    vocabs['char'] = Vocab(model_prefix + ".char_vocab", fileformat='txt2')
    # print('char_vocab: {}'.format(vocabs['char'].word_vecs.shape))

    # time_used = time.time() - start_time
    # print('Time used for loading vocabs = %ds'%time_used)
    # start_time = time.time()

    # print('Build SentenceMatchDataStream ... ')
    testDataStream = SentenceMatchTrainer.SentenceMatchDataStream(in_path, word_vocab=vocabs['word'], char_vocab=vocabs['char'],
                                              POS_vocab=vocabs['pos'], NER_vocab=vocabs['ner'], label_vocab=vocabs['label'],
                                              batch_size=FLAGS.batch_size, isShuffle=False, isLoop=True, isSort=False,
                                              max_char_per_word=FLAGS.max_char_per_word, max_sent_length=FLAGS.max_sent_length)
    # print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    # print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))
    # time_used = time.time() - start_time
    # print('Time used for Build SentenceMatchDataStream = %ds'%time_used)

    if wo_char: vocabs['char'] = None

    init_scale = 0.01
    best_path = model_prefix + ".best.model"

    # start_time = time.time()

    # print('Building Graph:')
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.variable_scope("Model", reuse=False, initializer=initializer):
            valid_graph = SentenceMatchModelGraph(num_classes, word_vocab=vocabs['word'], char_vocab=vocabs['char'],
                                                  POS_vocab=vocabs['pos'], NER_vocab=vocabs['ner'],
                                                  dropout_rate=FLAGS.dropout_rate, learning_rate=FLAGS.learning_rate,
                                                  optimize_type=FLAGS.optimize_type,
                                                  lambda_l2=FLAGS.lambda_l2, char_lstm_dim=FLAGS.char_lstm_dim,
                                                  context_lstm_dim=FLAGS.context_lstm_dim,
                                                  aggregation_lstm_dim=FLAGS.aggregation_lstm_dim, is_training=False,
                                                  MP_dim=FLAGS.MP_dim,
                                                  context_layer_num=FLAGS.context_layer_num,
                                                  aggregation_layer_num=FLAGS.aggregation_layer_num,
                                                  fix_word_vec=FLAGS.fix_word_vec,
                                                  with_filter_layer=FLAGS.with_filter_layer,
                                                  with_highway=FLAGS.with_highway,
                                                  word_level_MP_dim=FLAGS.word_level_MP_dim,
                                                  with_match_highway=FLAGS.with_match_highway,
                                                  with_aggregation_highway=FLAGS.with_aggregation_highway,
                                                  highway_layer_num=FLAGS.highway_layer_num,
                                                  with_lex_decomposition=FLAGS.with_lex_decomposition,
                                                  lex_decompsition_dim=FLAGS.lex_decompsition_dim,
                                                  with_left_match=(not wo_left_match),
                                                  with_right_match=(not wo_right_match),
                                                  with_full_match=(not wo_full_match),
                                                  with_maxpool_match=(not wo_maxpool_match),
                                                  with_attentive_match=(not wo_attentive_match),
                                                  with_max_attentive_match=(not wo_max_attentive_match))

        # time_used = time.time() - start_time
        # print('Time used for SentenceMatchModelGraph = %ds' % time_used)
        # start_time = time.time()

        # construct variable list to restore from saved model
        # we do not save word_embedding after training
        vars_ = {}
        for var in tf.all_variables():
            if "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        # load tf model
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, best_path)

        # time_used = time.time() - start_time
        # print('Time used for loading model parameters = %ds' % time_used)

def calc_relevance_scores(testDataStream):
    '''
    Use this function to calculate the relevance scores for a list of candidates given a query
    query: A dictionary containing all the information for the query
           e.g. {'content': 'Text of the query'}
    candidates: A list of dictionaries containing all the candidates.
           e.g. [{'content': 'Text of candidate 1'}, {'content': 'Text of candidate 2'}]

    return: The same list of the candidates for the given query, but with "relevance_score" added as an element in each dictionary.
            e.g. {'Text of the query': [{'content': 'Text of candidate 1', 'relevance_score': 0.9}, {'content': 'Text of candidate 2', 'relevance_score': 1.0}]}

    Right now, we only have the 'content' information, later we will add more meta information.

    '''
    # start_time = time.time()

    # Decoding on the test set:
    probs_list = SentenceMatchTrainer.evaluate(testDataStream, valid_graph, sess, outpath=out_path, label_vocab=vocabs['label'], mode=mode, char_vocab=vocabs['char'], POS_vocab=vocabs['pos'], NER_vocab=vocabs['ner'])

    # time_used = time.time() - start_time
    # print('Time used for Decoding = %ds (%.4f secs/sample)' % (time_used, time_used*1.0/testDataStream.get_num_instance()))
    # print(probs_list.iteritems().next())

    return probs_list

if __name__ == '__main__':
    init()
    calc_relevance_scores(testDataStream)