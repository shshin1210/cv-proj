import numpy as np

class C100Dataset:
    train_x = None
    train_y = None
    test_x = None
    test_y = None
    def __init__(self, filename):
        #   image name,class
        datax, datay = np.loadtxt(filename, unpack = True, max_rows=49999, delimiter=',', dtype='str')
        self.train_x = np.array([i for i in datax if '/train/' in i])
        self.train_y = np.array([datay[i] for i,x in enumerate(datax) if '/train/' in x])

        datax = np.loadtxt(filename, unpack = True, skiprows=49999, delimiter=',', dtype='str')
        self.test_x = np.array([i for i in datax if '/test/' in i])

    def getDataset(self):
        return [self.train_x, self.train_y, self.test_x]

class C100Testset:
    test_x = None
    test_y = None
    def __init__(self, filename):
        #   image name,class
        datax, datay = np.loadtxt(filename, unpack = True, max_rows=9999, delimiter=',', dtype='str')
        self.test_x = np.array([i for i in datax if '/test/' in i])
        self.test_y = np.array([datay[i] for i,x in enumerate(datax) if '/test/' in x])


    def getDataset(self):
        return [self.test_x, self.test_y]
