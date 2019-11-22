import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError

class ImageDataset(Dataset):
    def __init__(self, dir, transform):
        super(ImageDataset, self).__init__()
        self.image_dir = os.path.join(dir, 'image')
        self.image_names = os.listdir(self.image_dir)
        self.transform = transform

        print('=' * 40)
        print('Dataset: {} images'.format(len(self.image_names)))
        print('=' * 40)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        img = Image.open(os.path.join(self.image_dir, image_name)).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.image_names)
    
    def name(self):
        return 'ImageDataset'


def train_transform(size=64):
    transform_list = [
        transforms.Resize(size=(size, size)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)