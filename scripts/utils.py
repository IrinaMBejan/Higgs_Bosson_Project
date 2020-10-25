import numpy as np

def save_weights_model(model, filename):
  with open(filename, 'wb') as f:
      np.save(filename, model.best_weights)

def load_weights_model(filename):
  with open(filename, 'rb') as f:
      return np.load(f, allow_pickle=True)