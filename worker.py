#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='yelp', choices=['yelp'])
parser.add_argument('--mode', default='train', choices=['train', 'eval'])
parser.add_argument('--checkpoint-frequency', type=int, default=100)
parser.add_argument('--eval-frequency', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=30)
parser.add_argument("--device", default="/cpu:0")
parser.add_argument("--max-grad-norm", type=float, default=5.0)
parser.add_argument("--lr", type=float, default=0.01)
args = parser.parse_args()

import importlib
import os
import pickle
import random
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm

import ujson
import json
from data_util import batch
import gc
import onehot

from attention_plot import attention_plot

import csv
import pandas as pd
csv_file = open('loss_log.csv', 'a')
csv_writer = csv.writer(csv_file, lineterminator='\n')

with open('../study_data/data/src/dataset.json') as jsonfile:
    vector_dict = json.load(jsonfile)
#with open('../study_data/data/src/data.json') as json:
    #token2vec = json.load(f)

md = onehot.MakeData()
    
task_name = 'deepfix-HAN'
#task = importlib.import_module(task_name)
#print(type(task))
train_dir = '../study_data/yelp'
checkpoint_dir = os.path.join(train_dir, 'checkpoint')
tflog_dir = os.path.join(train_dir, 'tflog')
checkpoint_name = task_name + '-model'
checkpoint_dir = os.path.join(train_dir, 'checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

# @TODO: move calculation into `task file`
#trainset = task.read_trainset(epochs=1)
#print(type(trainset))

#class_weights = pd.Series(Counter([l for _, l in trainset]))
#class_weights = pd.Series(Counter(['{}'.format(l) for l in range(11)]))
class_weights = pd.Series(Counter([l for l in vector_dict]))
for num in range(11):
    class_weights['{}'.format(num)] = 0
    for i in vector_dict['{}'.format(num)]:
        class_weights['{}'.format(num)] += 1
print(class_weights)
print(class_weights)
class_weights = 1/(class_weights/class_weights.mean())
print(class_weights)
class_weights = class_weights.to_dict()
print(class_weights)

#vocab = task.read_vocab()
#vocab = len(vector_dict)
#labels = task.read_labels()

#classes = max(labels.values())+1
classes = 11
print("------")
print(classes)
print("------")
#vocab_size = task.vocab_size
vocab_size = 180
print("------")
print(vocab_size)
print("------")

#labels_rev = {int(v): k for k, v in labels.items()}
labels_rev = {}
for num in range(11):
    labels_rev[num] = num
print("------")
print(labels_rev)
print("------")

#vocab_rev = {int(v): k for k, v in vocab.items()}


def HAN_model_1(session, restore_only=False):
  """Hierarhical Attention Network"""
  import tensorflow as tf
  try:
    from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, DropoutWrapper
  except ImportError:
    MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
    GRUCell = tf.nn.rnn_cell.GRUCell
  from bn_lstm import BNLSTMCell
  from HAN_model import HANClassifierModel

  is_training = tf.placeholder(dtype=tf.bool, name='is_training')

  #cell = BNLSTMCell(50,is_training) # h-h norm LSTMCell
  cell = GRUCell(50)
  cell = MultiRNNCell([cell]*3) #cell = MultiRNNCell([cell]*5)

  model = HANClassifierModel(
      vocab_size=vocab_size,
      embedding_size=200,
      classes=classes,
      word_cell=cell,
      sentence_cell=cell,
      word_output_size=181, #100
      sentence_output_size=25,
      device=args.device,
      learning_rate=args.lr,
      max_grad_norm=args.max_grad_norm,
      dropout_keep_proba=0.8,
      is_training=is_training,
  )

  saver = tf.train.Saver(tf.global_variables())
  checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
  if checkpoint:
    print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
    saver.restore(session, checkpoint.model_checkpoint_path)
  elif restore_only:
    raise FileNotFoundError("Cannot restore model")
  else:
    print("Created model with fresh parameters")
    session.run(tf.global_variables_initializer())
  # tf.get_default_graph().finalize()
  return model, saver

model_fn = HAN_model_1

def decode(ex):
  #print('text: ' + '\n'.join([' '.join([vocab_rev.get(wid, '<?>') for wid in sent]) for sent in ex[0]]))
  print('label: ', labels_rev[ex[1]])

print('data loaded')

def batch_iterator(dataset, batch_size, max_epochs):
  print("dataset is nani, I Kininar")
  print(dataset)

  #from load_data import load_data
  #program_dataset = load_data(0)
  #train_x, train_y,_,_,_,_ = program_dataset.get_data

  #for i in range(max_epochs):
    #xb = []
    #yb = []
    #for ex in train_x:
        
  for i in range(max_epochs):
    xb = []
    yb = []
    for ex in dataset:
      x, y = ex
      x = [[2,4,6,7],[7,4,6,1,1],[9,9,9,9],[1,4,5,8,2],[5,3,1,3,8]]
      y = 0
      #print(x)
      #print(y)
      xb.append(x)
      yb.append(y)
      if len(xb) == batch_size:
        yield xb, yb
        xb, yb = [], []


def ev(session, model, dataset):
  predictions = []
  labels = []
  examples = []
  #for x, y in programs_and_labels_test
  #for x, y in tqdm(batch_iterator(dataset, args.batch_size, 1)):
  #print(dataset)
  #for x, y in tqdm(dataset):
  x = []
  y = []
  for eval_x, eval_y in dataset:
    eval_x_list = []
    eval_y_list = []
    for i,separate_line in enumerate(eval_x):
      eval_x_list.append(md.onehot_vec(separate_line, vocab_size))
      #y.append(labels)
      #y = labels
      eval_y_list.append('{}'.format(eval_y))  
    #print(x)
    #print(y)
    x.append(eval_x_list)
    y.append(eval_y_list)
  for j in tqdm(range(len(x))):
    examples.extend(x[j])
    labels.extend(y[j])
    prediction_output, word_attention_output, sentence_attention_output = session.run([
            model.prediction,
            model.word_level_output,
            model.sentence_level_output,
        ], model.get_feed_data(x[j], is_training=False))
    #attention_plot(x[j],sentence_attention_output,j)
    predictions.extend(prediction_output)
    
    #predictions.extend(session.run(model.prediction, model.get_feed_data(x[j], is_training=False)))
  
  for i,prediction_int in enumerate(predictions):
    predictions[i] = str(prediction_int)

  df = pd.DataFrame({'predictions': predictions, 'labels': labels, 'examples': examples})

  return df


def evaluate(dataset):
  tf.reset_default_graph()
  #config = tf.ConfigProto(allow_soft_placement=True)
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=False)
  #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  #config = tf.ConfigProto(gpu_options=gpu_options)
  config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
      visible_device_list="0", # specify GPU number
      allow_growth=True
    )
  )
  with tf.Session(config=config) as s:
    model, _ = model_fn(s, restore_only=True)
    df = ev(s, model, dataset)
  print((df['predictions'] == df['labels']).mean())
  #print(df)
  #import IPython
  #IPython.embed()


def train(programs_and_labels):
  epochs = 100
  tf.reset_default_graph()
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
  #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  #config = tf.ConfigProto(allow_soft_placement=True)
  ##config = tf.ConfigProto(gpu_options=gpu_options)
  #with tf.Session(config=config) as s:
  config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
      visible_device_list="0", # specify GPU number
      allow_growth=True
    )
  )
  eval_case = programs_and_labels[:100]
  del programs_and_labels[:100]    
  x = []
  y = []
  for programs, labels in programs_and_labels:
    train_x = []
    train_y = []
    for i, separate_line in enumerate(programs):
        train_x.append(md.onehot_vec(separate_line, vocab_size))
#        print(len(x))
        train_y.append('{}'.format(labels))
#        print(len(y))
#        print(y)
    x.append(train_x)
    y.append(train_y)
  #print(np.shape(x))
  #print(np.shape(y))
  eval_x = []
  eval_y = []
  for eval_prog, eval_label in eval_case:
    eval_x_2 = []
    eval_y_2 = []
    for j, sep_line in enumerate(eval_prog):
        eval_x_2.append(md.onehot_vec(sep_line, vocab_size))
        eval_y_2.append('{}'.format(eval_label))
    eval_x.append(eval_x_2)
    eval_y.append(eval_y_2)
    
  with tf.Session(config=config) as s:
   #with tf.Session() as s:
   summary_writer = tf.summary.FileWriter(tflog_dir, graph=tf.get_default_graph())
   model, saver = model_fn(s)
   for epoch in range(epochs):
    print("epochs ", epoch, " start")

    # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
    pconf = projector.ProjectorConfig()

    # # You can add multiple embeddings. Here we add only one.
    # embedding = pconf.embeddings.add()
    # embedding.tensor_name = m.embedding_matrix.name

    # # Link this tensor to its metadata file (e.g. labels).
    # embedding.metadata_path = vocab_tsv

    # print(embedding.tensor_name)

    # Saves a configuration file that TensorBoard will read during startup.

    #print(len(list(programs_and_labels)))
#    eval_case = programs_and_labels[:100]
#    del programs_and_labels[:100]
    
#    for programs, labels in programs_and_labels:
#        x = []
#        y = []
#        for i,separate_line in enumerate(programs):
#            x.append(md.onehot_vec(separate_line, vocab_size))
#            #y.append(labels)
#            #y = labels
#            y.append('{}'.format(labels))
        #for i, (x, y) in enumerate(batch_iterator(task.read_trainset(epochs=3), args.batch_size, 300)):
        #print("---------")
        #print(x)
        #print(np.shape(x))
        #print('+++++++++')
        #print(y)
        #print('*********')
    for i in range(len(x)):
        #print(len(x))
        #print(len(y))
        #print(y)
        #print(x[i])
        #print(y[i])
        #print(np.shape(x[i]))
        #print(y[i])
        fd = model.get_feed_data(x[i], y[i], class_weights=class_weights)
        #print("x 's value is :")
        #print(np.array(x).shape)
        #print(x)
        #print("y 's value is :")
        #print(np.array(y).shape)
        #print(y)
        #print("fd 's value is :")
        #print(fd.items())

        # import IPython
        # IPython.embed()
        #del x
        #del y
        #del programs_and_labels
        #del programs
        #del labels
        #del i
        #del separate_line
        #gc.collect()

        t0 = time.time()
        step, summaries, loss, accuracy, _ = s.run([
            model.global_step,
            model.summary_op,
            model.loss,
            model.accuracy,
            model.train_op,
        ], fd)
        '''
        step, summaries, loss, accuracy, _, word_attention_output, word_dropout_output, sentence_attention_output, sentence_dropout_output = s.run([
            model.global_step,
            model.summary_op,
            model.loss,
            model.accuracy,
            model.train_op,
            model.word_level_output,
            model.word_level_output_dropout,
            model.sentence_level_output,
            model.sentence_level_output_dropout
        ], fd)
        '''
        '''
        step, sample_weights, _ = s.run([
            model.global_step,
            model.sample_weights,
            model.train_op,
        ], fd)
        '''
        td = time.time() - t0

        summary_writer.add_summary(summaries, global_step=step)
        projector.visualize_embeddings(summary_writer, pconf)
        if step % 1 == 0:
          print('step %s, loss=%s, accuracy=%s, t=%s, inputs=%s' % (step, loss, accuracy, round(td, 2), fd[model.inputs].shape))
        if step != 0 and step % args.checkpoint_frequency == 0:
          print('checkpoint & graph meta')
          saver.save(s, checkpoint_path, global_step=step)
          print('checkpoint done')
        if step != 0 and step % args.eval_frequency == 0:
          print('evaluation at step %s' % i)
          dev_df = ev(s, model, eval_case)
          print('dev accuracy: %.2f' % (dev_df['predictions'] == dev_df['labels']).mean())
        del fd
        gc.collect()
    # 100 - test
    csv_writer.writerow([epoch,s.run(model.loss, model.get_feed_data(eval_x[0], eval_y[0], class_weights=class_weights))])
    #evaluate(eval_case)
    print("epochs ", epoch, " end")

def main():
  #dict_size = len(token2vec)
  programs_and_labels_data = md.program_sep(vector_dict)
  #print(programs_and_labels)
  data_list = list(programs_and_labels_data)
  programs_and_labels = random.sample(data_list,len(data_list))
  del programs_and_labels[:1870]
  #print(len(programs_and_labels))
  test_case = programs_and_labels[:100]
  del programs_and_labels[:100]
  #print(len(test_case))
  #print(test_case)
  #print(len(programs_and_labels))
  print(len(programs_and_labels))
  if args.mode == 'train':
    train(programs_and_labels)
    evaluate(test_case)
  elif args.mode == 'eval':
    evaluate(test_case)
    #evaluate(task.read_devset(epochs=1))

if __name__ == '__main__':
  main()
