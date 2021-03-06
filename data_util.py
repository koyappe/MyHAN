import numpy as np
#import cupy


def batch(inputs):
  batch_size = len(inputs)
  #print(batch_size)
  document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
  #print(document_sizes)
  document_size = document_sizes.max()
  #print(document_size)
  #for doc in inputs:  
    #for sent in doc:
      #print(sent)
  sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
  sentence_size = max(map(max, sentence_sizes_))

  b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32) # == PAD

  sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
  for i, document in enumerate(inputs):
    for j, sentence in enumerate(document):
      sentence_sizes[i, j] = sentence_sizes_[i][j]
      for k, word in enumerate(sentence):
        b[i, j, k] = word

  return b, document_sizes, sentence_sizes