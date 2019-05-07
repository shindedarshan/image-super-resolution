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
        assert root_dir1, print('no HR_image_path specified')
        assert root_dir2, print('no LR_image_path specified')
        self.root_dir1=root_dir1
        self.root_dir2=root_dir2
        self.hrlist = glob(self.root_dir1+'*.jpg')
        self.lrlist = glob(self.root_dir2+'*.jpg')
        self.transform=transform
        
    def __len__(self):
        return min(len(self.hrlist),len(self.lrlist))
    
    def __getitem__(self, idx):
        data = {}
        hr = Image.open(self.hrlist[idx])
        hlr = hr.resize((16, 16), resample=PIL.Image.BICUBIC)
        lr = Image.open(self.lrlist[idx])
        if self.transform:
            data['hr'] = self.transform(hr)
            data['hlr'] = self.transform(hlr)
            data['lr'] = self.transform(lr)
        return data
    
class dataset_l2h(data.Dataset):
    def __init__(self, root_dir, transform= None):
        assert root_dir, print('no LR_image_path specified')
        self.root_dir=root_dir
        self.lrlist = glob(self.root_dir+'*.jpg')
        self.transform=transform
        
    def __len__(self):
        return len(self.lrlist)
    
    def __getitem__(self, idx):
        data = {}
        lr = Image.open(self.lrlist[idx])
        if self.transform:
            data['lr'] = self.transform(lr)
        data['imgpath'] = self.lrlist[idx]
        return data    

class dataset_h2l(data.Dataset):
    def __init__(self, root_dir, transform= None):
        assert root_dir, print('no HR_image_path specified')
        self.root_dir=root_dir
        self.hrlist = glob(self.root_dir+'*.jpg')
        self.transform=transform
        
    def __len__(self):
        return len(self.hrlist)
    
    def __getitem__(self, idx):
        data = {}
        hr = Image.open(self.hrlist[idx])
        if self.transform:
            data['hr'] = self.transform(hr)
            data['imgpath'] = self.hrlist[idx]
        return data    

class dataset_combined(data.Dataset):
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
        inp64 = Image.open(self.img_list[index])
        inp16 = inp64.resize((16, 16), resample=PIL.Image.BICUBIC)
        data['hr'] = self.transform(inp64)
        data['lr'] = self.transform(inp16)
        data['imgpath'] = self.img_list[index]
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

def get_loader_l2h(path = None, bs = 1):
    transform = transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dataset_l2h(path, transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=bs,
                             shuffle=False, num_workers=2, pin_memory=True)
    return data_loader
    
def get_loader_h2l(path = None, bs = 1):
    transform = transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dataset_h2l(path, transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=bs,
                             shuffle=False, num_workers=2, pin_memory=True)
    return data_loader

def get_loader_combined(path = None, bs = 1):
    transform = transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dataset_combined(path, transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=bs,
                             shuffle=False, num_workers=2, pin_memory=True)
    return data_loader
    
