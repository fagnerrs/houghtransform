import numpy as np

def getMaxValue(accumulator):

  arg = np.argmax(accumulator)

  row = accumulator.shape[1]*accumulator.shape[2]
  col = int(arg / row)
  mod = arg % row

  row = int(mod / accumulator.shape[2])
  cc = mod % accumulator.shape[2]

  return accumulator[col, row, cc]