from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy
from six.moves import xrange
#from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import tensorflow as tf

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32)
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]
  
def check_dims(f, gridSize, nbDim):
  print('Check dimensions ', f.name,  flush = True)
  with f as bytestream:
    headerSize = _read32(bytestream)
    magic = _read32(bytestream)
    if magic != 7919:
      raise ValueError('Invalid magic number %d in maps file: %s' %
                       (magic, f.name))
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    lays = _read32(bytestream)
    assert(rows == gridSize)
    assert(cols == gridSize)
    assert(lays == gridSize)
    chan = _read32(bytestream)
    assert(chan == nbDim)
  
def extract_maps(f):

  #print('Extracting', f.name,  flush = True)
  with f as bytestream:
    headerSize = _read32(bytestream)
    magic = _read32(bytestream)
    if magic != 7919:
      raise ValueError('Invalid magic number %d in maps file: %s' %
                       (magic, f.name))
    rows = _read32(bytestream)
    #print("rows "+str(rows))
    cols = _read32(bytestream)
    #print("cols "+str(cols))
    lays = _read32(bytestream)
    #print("lays "+str(lays))
    chan = _read32(bytestream)
    #print("chan "+str(chan))
    metaSize = _read32(bytestream)
    #print("metaSize "+str(metaSize))
    num_maps = _read32(bytestream)
    #print("num_maps "+str(num_maps))
    header_end = bytestream.read(headerSize - 4*8)
    if num_maps<=0 :
      return None,None
    size = int(rows) * int(cols) * int(lays) * int(chan) * int(num_maps)
    size += int(metaSize) * int(num_maps)
    try :
      buf = bytestream.read(size)
    except OverflowError :
      return None, None
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_maps, -1)
    meta = numpy.ascontiguousarray(data[:, -int(metaSize):]).view(dtype=numpy.int32)
    
    ss_dict = {0: -1,
                66:0,#B
                98:0,#b
                67:1,#C
                69:2,#E
                71:3,#G
                72:4,#H
                73:5,#I
                84:6,#T
                }
    #meta[:,3] = [ss_dict[x] for x in meta[:,3]] #Y commented!
    res_dict = {0:-1,
                65:0, #A
                67:1, #C
                68:2, #D
                69:3, #E
                70:4, #F
                71:5, #G
                72:6, #H
                73:7, #I
                75:8, #K
                76:9, #L
                77:10,#M
                78:11,#N
                80:12,#P
                81:13,#Q
                82:14,#R
                83:15,#S
                84:16,#T
                86:17,#V
                87:18,#W
                89:19 #Y
                }
    #meta[:,1] = [res_dict[x] for x in meta[:,1]]
    #print(meta[:,3])
    #print(meta[:,2])

                
    data = data[:,:-int(metaSize)]
    return data , meta

class DataSet(object):

  def __init__(self,
               maps,
               meta,
               dtype=dtypes.float32,
               seed=None,
               prop = 1,
               shuffle = False):

    # prop means the percentage of maps from the data that are put in the dataset, useful to make the dataset lighter
    # when doing that shuffle is useful to take different residue each time

    seed1, seed2 = random_seed.get_seed(seed)
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32, dtypes.float16):
      raise TypeError('Invalid map dtype %r, expected uint8 or float32 or float16' %
                      dtype)
   
    
    
    
    if dtype == dtypes.float32:
      maps = maps.astype(numpy.float32)
      numpy.multiply(maps, 1.0 / 255.0, out = maps)
    if dtype == dtypes.float16:
      maps = maps.astype(numpy.float16)
      numpy.multiply(maps, 1.0 / 255.0, out = maps)

    if shuffle:
      perm0 = numpy.arange(maps.shape[0])[:int(maps.shape[0]*prop)]
      self._maps = maps[perm0]
      self._meta = meta[perm0]
    else:
      self._maps = maps
      self._meta = meta


    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_res = self._maps.shape[0]

  @property
  def maps(self):
    return self._maps


  @property
  def meta(self):
    return self._meta

  @property
  def num_res(self):
    return self._num_res

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True, select_residue = -1):
    """Return the next `batch_size` examples from this data set."""

    # Select residue is not used anymore, just kept for compatibility purposes

    start = self._index_in_epoch
    # Shuffle for the first epoch

    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_res)
      numpy.random.shuffle(perm0)
      self._maps = self.maps[perm0]
      self._meta = self._meta[perm0]      # Go to the next epoch

    if start + batch_size > self._num_res:
      # Finished epoch
      self._epochs_completed += 1

      # Get the rest examples in this epoch
      rest_num_examples = self._num_res - start
      maps_rest_part = self._maps[start:self._num_res]
      meta_rest_part = self._meta[start:self._num_res]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_res)
        numpy.random.shuffle(perm)
        self._maps = self.maps[perm]
        self._meta = self.meta[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      maps_new_part = self._maps[start:end]
      meta_new_part = self._meta[start:end]
      return numpy.concatenate((maps_rest_part, maps_new_part), axis=0) , numpy.concatenate((meta_rest_part, meta_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._maps[start:end], self._meta[start:end]

  def append(self, dataSet_):
    self._maps = numpy.concatenate((self._maps, dataSet_._maps))
    self._meta = numpy.concatenate((self._meta, dataSet_._meta))
    self._num_res += dataSet_._num_res

  def is_res(self, index, res_code):
    if index < self._num_res :
      if self._meta[index, 1] == res_code:
        return True
    else:
      print('index = num_res')
    return False

  def find_next_res(self, index, res_code):
    i = index + 1

    while (not self.is_res(i, res_code)) and i < self._num_res - 1:
      i += 1
    if self.is_res(i, res_code):
      return i
    return -1

def read_data_set(filename,
                   dtype=dtypes.float32,
                   seed=None,
                   shuffle = False,
                   prop = 1):
  local_file = filename
  try :
    with open(local_file, 'rb') as f:
      train_maps,train_meta = extract_maps(f)
    if train_maps is None :
      return None
    train = DataSet(
        train_maps, train_meta, dtype=dtype, seed=seed, shuffle = shuffle, prop = prop)
    return train
  except ValueError :
    return None
