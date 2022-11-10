import numpy as np

class NoNormalization:

    def __init__(self):
        pass
    
    def __mul__(self, x):
        return x
        
    def __rmul__(self, x):
        return x

class Periodic:

    def __init__(self, x_max):
    
        self.x_max = x_max
    
    def __mul__(self, x):  # TODO: check if runs with multiple batches
    
        x = 2 * np.pi * x / self.x_max
        x_sin = np.sin(x)
        x_cos = np.cos(x)
    
        return np.array([(x_sin+1)/2.0, (x_cos+1)/2.0])
    
    def __rmul__(self, x):
    
        x = 2 * np.pi * x / self.x_max
        x_sin = np.sin(x)
        x_cos = np.cos(x)
    
        return np.array([(x_sin+1)/2.0, (x_cos+1)/2.0])

class OneHotEncoding:

    def __init__(self, classes):
    
        self.classes = classes
    
    def __mul__(self, x):
    
        x = x.astype(int)
        output = np.zeros((x.shape[0], len(self.classes)))
        output[np.arange(x.size), x] = 1

        return output
    
    def __rmul__(self, x):
    
        output = np.zeros((x.shape[0], len(self.classes)))
        output[np.arange(x.size), x] = 1

        return output
    
class Normalize:
    
    def __init__(self, x_min, x_max):
    
        self.x_min = x_min
        self.x_max = x_max
    
    def __mul__(self, x): # TODO: check if runs with multiple batches
    
        if self.x_min == self.x_max:
            return 0
        else:
            return (x - self.x_min)/(self.x_max - self.x_min)
    
    def __rmul__(self, x):
    
        if self.x_min == self.x_max:
            return 0
        else:
            return (x - self.x_min)/(self.x_max - self.x_min)