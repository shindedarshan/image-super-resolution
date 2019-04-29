import os
from glob import glob
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import PIL

class faces_super(data.Dataset):
    def __init__(self, data_path,transform):
        assert data_path, print('no data_path specified')
        self.transform = transform
        self.img_list = []
        list_name = (glob(os.path.join(data_path, "*.jpg")))
        list_name.sort()
        for filename in list_name:#jpg
            self.img_list.append(filename)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        inp16 = Image.open(self.img_list[index])
        inp64 = inp16.resize((64, 64), resample=PIL.Image.BICUBIC)
        data['img64'] = self.transform(inp64)
        data['img16'] = self.transform(inp16)
        data['imgpath'] = self.img_list[index]
        return data

class dataset_maker(data.Dataset):
    def __init__(self, root_dir1, root_dir2, transform= None):
        self.root_dir1=root_dir1
        self.root_dir2=root_dir2
        self.hrlist = glob(root_dir1+'*.jpg')
        self.lrlist = glob(root_dir2+'*.jpg')
        self.transform=transform
        
    def __len__(self):
        return min(len(self.hrlist),len(self.lrlist))
    
    def __getitem__(self, idx):
        data = {}
        hr = Image.open(self.hrlist[idx])
        lr = Image.open(self.lrlist[idx])
        if self.transform:
            data['hr'] = self.transform(hr)
            data['lr'] = self.transform(lr)
        return data
    
def get_loader(path1 = None, path2 = None, bs = 1):
    transform = transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if path2 == None:
        dataset = faces_super(path1, transform)
    if path1 != None and path2 != None:
        dataset = dataset_maker(path1, path2, transform)    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=bs,
                             shuffle=False, num_workers=2, pin_memory=True)
    return data_loader
