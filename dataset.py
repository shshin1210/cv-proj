from cv2 import transform
from matplotlib import transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
import torchvision.transforms as transforms

class C100Dataset(Dataset):

    # Todo
    classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
               'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
            'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
            'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
            'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    def __init__(self, train = True, transfrom = None ):
        super().__init__()
        self.csv_dir = './dataset/data/cifar100_nl.csv' if train else './dataset/data/cifar100_nl_test.csv'
        self.transfrom = transfrom
        dataset = pd.read_csv(self.csv_dir, names = ['filename', 'classname']) # 59998개

        if train:
            # trainset '/train/'
            dataset = dataset[:49999] 

        # img paths
        self.img_paths = dataset['filename']
        # img_labels
        labels = dataset['classname']

        self.img_labels = []
        for label in labels:
            self.img_labels.append(C100Dataset.classes.index(label))


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = './dataset/' + self.img_paths[idx]
        image = read_image(img_path)
        label = self.img_labels[idx]

        if self.transfrom:
            image = self.transfrom(image)

        return [image, label]
    
if __name__ == "__main__":
    # Transform
    stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(*stats)])

    trainset = C100Dataset(train=True, transfrom=transform)
    testset = C100Dataset(train=False, transfrom=transform)
    
    # Todo
    data_size = len(trainset)
    subtrain_size = int(0.8*data_size)
    val_size = data_size - subtrain_size
    subtrain_set, val_set =  random_split(trainset, [subtrain_size, val_size])

    # data loader
    subtrain_loader = DataLoader(subtrain_set, batch_size=4, shuffle = True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2, drop_last=False)

    # Check subtrain_lodaer 
    for images, labels in subtrain_loader:
        print(len(labels))
        print("Batch of images has shape: ",images.size())
        print("Batch of labels has shape: ", labels.size())

    for images, labels in val_loader:
        print("Batch of images has shape: ",images.size())
        print("Batch of labels has shape: ", labels.size())
    
    for images, labels in test_loader:
        print("Batch of images has shape: ",images.size())
        print("Batch of labels has shape: ", labels.size())
    # Todo : val_loader, test_loader
