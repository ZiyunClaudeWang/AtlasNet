import torch

import time
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from .event_utils import gen_discretized_event_volume, normalize_event_volume
from easydict import EasyDict
from tqdm import tqdm
import os
import cv2
import pdb
from scipy import ndimage

class TrainerDataset(object):
    def __init__(self):
        super(TrainerDataset, self).__init__()

    def build_dataset(self):
        classes = ["02691156",  
                        "02828884",  
                        "02933112",  
                        "02958343",  
                        "03001627",  
                        "03211117", 
                        "03636649",  
                        "03691459", 
                        "04090263",
                        "04256520", 
                        "04379243", 
                        "04401088", 
                        "04530566"]

        if self.opt.random_data:
            dset_all = RandomShapeNet(class_name=classes[0])
        else:
            dset_all = EvShapeNet(class_name=classes[0], use_mask_input=self.opt.use_mask_input)

        train_len = int(0.9 * len(dset_all))
        val_len = len(dset_all) - train_len
        train_dataset, val_dataset = random_split(dset_all, [train_len, val_len])

        self.datasets = EasyDict()
        # Create Datasets
        self.datasets.dataset_train = train_dataset
        self.datasets.dataset_test = val_dataset

        if not self.opt.demo:
            # Create dataloaders
            self.datasets.dataloader_train = torch.utils.data.DataLoader(self.datasets.dataset_train,
                                                                         batch_size=self.opt.batch_size,
                                                                         shuffle=True,
                                                                         num_workers=int(self.opt.workers))
            self.datasets.dataloader_test = torch.utils.data.DataLoader(self.datasets.dataset_test,
                                                                        batch_size=self.opt.batch_size_test,
                                                                        shuffle=True, num_workers=int(self.opt.workers))
            self.datasets.len_dataset = len(self.datasets.dataset_train)
            self.datasets.len_dataset_test = len(self.datasets.dataset_test)

class EvShapeNet(Dataset):
    def __init__(self, width=256, 
                        height=256, 
                        volume_time_slices=10, 
                        delta_t=0.01,
                        mode='train',
                        class_name=None,
                        use_mask_input=False,
                        num_views=45,
                        meta_path='/Datasets/cwang/event_shapenet/shapenet_r2n2.txt',
                        event_folder = '/Datasets/cwang/event_shapenet_corrected_events',
                        gt_folder='/Datasets/cwang/event_shapenet_corrected'):

        self.width = width
        self.height = height
        self.volume_time_slices = volume_time_slices
        self.mode = mode
        self.class_name = class_name
        self.event_folder = event_folder
        self.gt_folder = gt_folder
        self.delta_t = delta_t

        self.use_mask_input = use_mask_input
        self.num_views = num_views
        self.paths = self.read_meta(gt_folder, meta_path, class_name=class_name)
        print("There are {} objects in the current dataset".format(len(self.paths)))
        
    def read_meta(self, data_folder, meta_file, class_name=None):
        classes = [c for c in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, c))]
        meta_file = open(meta_file, 'r')
        all_paths = []

        # generate list of models
        for l in meta_file.readlines():
            l = l.strip("\n")
            if class_name is None or class_name in l:
                split_name = l.split("/") 
                cname = split_name[0]
                oname = split_name[1]
                model_path = os.path.join(cname, oname)

                # TODO: hack check if the events are generated 
                event_path = os.path.join(self.event_folder, model_path, "events.npz")
                if os.path.exists(event_path):
                    all_paths.append(model_path)
                    

        return all_paths

    def __len__(self):
        return len(self.paths)

    def rotate(self, inputs, x, axis=[1, 2]):
        return ndimage.rotate(inputs, x, reshape=False, axes=axis)

    def __getitem__(self, index):
        path = self.paths[index]
        output = {}
        
        # find events based on image time
        if self.use_mask_input:
            # read sil masks
            masks = []
            for i in range(45):
                data = np.load(os.path.join(self.gt_folder, path, "{:05}_gt.npz".format(i)))
                masks.append(data['sil_mask'])
            network_input = np.stack(masks, axis=0).astype(np.float32)
        else:
            try:
                event_data = dict(np.load(os.path.join(self.event_folder, path, "events.npz")))
                event_volume = gen_discretized_event_volume(
                                                    torch.from_numpy(event_data['x']).long(),
                                                    torch.from_numpy(event_data['y']).long(), 
                                                    torch.from_numpy(event_data['t'].astype(np.float32)), 
                                                    torch.from_numpy(event_data['p']),
                                                    [self.volume_time_slices*2,
                                                     self.height,
                                                     self.width])
                network_input = normalize_event_volume(event_volume).float()
            except:
                print("Invalid Path:", path)

        model = o3d.io.read_triangle_mesh(os.path.join(self.gt_folder, path, "model.obj"))

        # sample 1000 points from model
        points = np.array(model.sample_points_uniformly(number_of_points=1000).points)
        # normalize events and convert to event volume
        # get sample points
        output = {
                    "input_data": network_input,
                    "points": points.astype(np.float32)
                }
        return output

class RandomShapeNet(Dataset):
    def __init__(self, width=256, 
                        height=256, 
                        volume_time_slices=10, 
                        delta_t=0.01,
                        mode='train',
                        class_name=None,
                        meta_path='/Datasets/cwang/event_shapenet/shapenet_r2n2.txt',
                        event_folder = '/Datasets/cwang/event_shapenet_corrected_events',
                        gt_folder='/Datasets/cwang/event_shapenet_corrected'):

        self.width = width
        self.height = height
        self.volume_time_slices = volume_time_slices
        self.mode = mode
        self.class_name = class_name
        self.event_folder = event_folder
        self.gt_folder = gt_folder
        self.delta_t = delta_t
        self.paths = self.read_meta(gt_folder, meta_path, class_name=class_name)
        print("There are {} objects in the current dataset".format(len(self.paths)))
        
    def read_meta(self, data_folder, meta_file, class_name=None):
        classes = [c for c in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, c))]
        meta_file = open(meta_file, 'r')
        all_paths = []

        # generate list of models
        for l in meta_file.readlines():
            l = l.strip("\n")
            if class_name is None or class_name in l:
                split_name = l.split("/") 
                cname = split_name[0]
                oname = split_name[1]
                model_path = os.path.join(cname, oname)
                event_path = os.path.join(self.event_folder, model_path, "events.npz")
                if os.path.exists(event_path):
                    all_paths.append(model_path)
        return all_paths
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        path = self.paths[index]
        
        # find events based on image time
        model = o3d.io.read_triangle_mesh(os.path.join(self.gt_folder, path, "model.obj"))

        # sample 1000 points from model

        points = np.array(model.sample_points_uniformly(number_of_points=1000).points)

        # normalize events and convert to event volume
        event_volume = np.ones([self.volume_time_slices*2, self.height, self.width]) * index / self.__len__()
        output = {
                    "event_volume": event_volume.astype(np.float32),
                    "points": points.astype(np.float32)
                }
        return output

if __name__ == "__main__":
    classes = ["02691156",  
                    "02828884",  
                    "02933112",  
                    "02958343",  
                    "03001627",  
                    "03211117", 
                    "03636649",  
                    "03691459", 
                    "04090263",
                    "04256520", 
                    "04379243", 
                    "04401088", 
                    "04530566"]

    dset = EvShapeNet(class_name=classes[0])
    loader = DataLoader(dset, batch_size=1, shuffle=False)
    for index, b in enumerate(loader):
        print(b['event_volume'].shape)
        pdb.set_trace()

