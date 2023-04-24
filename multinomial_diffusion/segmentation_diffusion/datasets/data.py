import math
import torch
import numpy as np
import torchio
from glob import glob
import os
from torch.utils.data import DataLoader, Dataset
# from torchflow.data.loaders.nde.image import MNIST
from torchvision.transforms import RandomHorizontalFlip, Pad, RandomAffine, \
    CenterCrop, RandomCrop, Compose, ToPILImage, ToTensor

from cityscapes.cityscapes_fast import CityscapesFast, \
    cityscapes_indices_segmentation_to_img, \
    cityscapes_only_categories_indices_segmentation_to_img

dataset_choices = {
    'cityscapes_coarse', 'cityscapes_fine',
    'cityscapes_coarse_large', 'cityscapes_fine_large', 'shape_net_chairs','shape_net_dishwasher'}

def add_cubes(tensor):
    shape = tensor.shape[1]
    if np.random.randint(2):
        tensor[:shape//2,:shape//2,:shape//2] = np.ones_like(tensor[:shape//2,:shape//2,:shape//2])
    else:
        tensor[:shape//2,:shape//2,-shape//2:] = np.ones_like(tensor[:shape//2,:shape//2,-shape//2:])
    # if np.random.randint(2):
    #     tensor[-shape//2:,-shape//2:,-shape//2:] = np.ones_like(tensor[-shape//2:,-shape//2:,-shape//2:])
    # else:
    #     tensor[-shape//2:,-shape//2:,:shape//2] = np.ones_like(tensor[-shape//2:,-shape//2:,:shape//2])
    # if np.random.randint(2):
    #     tensor[:shape//2,-shape//2:,:shape//2] = np.ones_like(tensor[:shape//2,-shape//2:,:shape//2])
    # else:
    #     tensor[:shape//2,-shape//2:,-shape//2:] = np.ones_like(tensor[:shape//2,-shape//2:,-shape//2:])
    # if np.random.randint(2):
    #     tensor[-shape//2:,:shape//2,:shape//2] = np.ones_like(tensor[-shape//2:,:shape//2,:shape//2])
    # else:
    #     tensor[-shape//2:,:shape//2,-shape//2:] = np.ones_like(tensor[-shape//2:,:shape//2,-shape//2:])
    return tensor

def cut_slow(tensor):
    energy = np.count_nonzero(tensor>0)
    shape = tensor.shape[-1]
    a = b = c = 0
    for x in range(shape,0, -1):
        if np.count_nonzero(tensor[x:-x, ...]>0) == energy:
            a = x
            break
    for y in range(shape,0, -1):
        if np.count_nonzero(tensor[::, y:-y, ::]>0) == energy:
            b = y
            break
    for z in range(shape,0, -1):
        if np.count_nonzero(tensor[..., z:-z]>0) == energy:
            c = z
            break
    if a == 0:
        if b == 0:
            if c == 0:
                return tensor
            else:
                return tensor[:, :, c:-c]
        else:
            if c == 0:
                return tensor[:, b:-b, :]
            else:
                return tensor[:, b:-b, c:-c]
    else:
        if b == 0:
            if c == 0:
                return tensor
            else:
                return tensor[a:-a, :, c:-c]
        else:
            if c == 0:
                return tensor[a:-a, b:-b, :]
            else:
                return tensor[a:-a, b:-b, c:-c]

class Dataset3D(Dataset):
    def __init__(self, path, transform=None):
        self.files = glob(os.path.join(path, '*.npy'))
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = np.load(self.files[idx])
        if self.transform is not None:
            file = self.transform(file)
            
        return file, {}



def add_data_args(parser):

    # Data params
    parser.add_argument('--dataset', type=str, default='shape_net_dishwasher',
                        choices=dataset_choices)
    parser.add_argument('--data_dir', type=str, default='/home/sharfikeg/my_files/VoxelDiffusion/data/ShapeNet15k_voxels/dishwasher')
    

    # Train params
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=eval, default=False)
    parser.add_argument('--augmentation', type=str, default=None)


def get_plot_transform(args):
    if args.dataset in ('cityscapes_coarse', 'cityscapes_coarse_large'):
        return cityscapes_only_categories_indices_segmentation_to_img

    elif args.dataset in ('cityscapes_fine', 'cityscapes_fine_large'):
        return cityscapes_indices_segmentation_to_img

    else:
        def identity(x):
            return x
        return identity


def get_data_id(args):
    return '{}'.format(args.dataset)


def get_data(args):
    assert args.dataset in dataset_choices

    # Dataset
    # data_shape = get_data_shape(args.dataset)

    if args.dataset == 'cityscapes_coarse':
        data_shape = (1, 32, 64)
        num_classes = 8
        pil_transforms = get_augmentation(args.augmentation, args.dataset,
                                          (32, 64))
        pil_transforms = Compose(pil_transforms)
        train = CityscapesFast(split='train', resolution=(32, 64), transform=pil_transforms, only_categories=True)
        test = CityscapesFast(split='test', resolution=(32, 64), transform=pil_transforms, only_categories=True)
    elif args.dataset == 'cityscapes_fine':
        data_shape = (1, 32, 64)
        num_classes = 34
        pil_transforms = get_augmentation(args.augmentation, args.dataset,
                                          (32, 64))
        pil_transforms = Compose(pil_transforms)
        train = CityscapesFast(split='train', resolution=(32, 64), transform=pil_transforms, only_categories=False)
        test = CityscapesFast(split='test', resolution=(32, 64), transform=pil_transforms, only_categories=False)

    elif args.dataset == 'cityscapes_coarse_large':
        data_shape = (1, 128, 256)
        num_classes = 8
        pil_transforms = get_augmentation(args.augmentation, args.dataset,
                                          (128, 256))
        pil_transforms = Compose(pil_transforms)

        train = CityscapesFast(
            split='train', resolution=(128, 256), transform=pil_transforms,
            only_categories=True)
        test = CityscapesFast(
            split='test', resolution=(128, 256), transform=pil_transforms,
            only_categories=True)

    elif args.dataset == 'cityscapes_fine_large':
        data_shape = (1, 128, 256)
        num_classes = 34
        pil_transforms = get_augmentation(args.augmentation, args.dataset,
                                          (128, 256))
        pil_transforms = Compose(pil_transforms)
        train = CityscapesFast(
            split='train', resolution=(128, 256), transform=pil_transforms,
            only_categories=False)
        test = CityscapesFast(
            split='test', resolution=(128, 256), transform=pil_transforms,
            only_categories=False)
        
    elif args.dataset == 'shape_net_dishwasher' or args.dataset == 'shape_net_chairs':
        data_shape = (1, 32, 32, 32)
        num_classes = 2
        augmentations = Compose([
            # add_cubes,
            # cut_slow,

            # lambda array: ((torchio.transforms.Resize(32)(torch.FloatTensor(array).unsqueeze(0)))>0).long(),
            lambda array: (torch.FloatTensor(array[0]).unsqueeze(0)>0).long(),
            # *2.0-1.0,
            torchio.transforms.RandomFlip(
                axes=(0,1,2),
                flip_probability=0.5
            ),
            # torchio.transforms.RandomAffine(
            #     scales = (1,1),
            #     degrees=(0, 360),
            #     translation=(-10,10),
            # ),
            lambda array: (array>0).long(),
        ])
        dataset = Dataset3D(
            args.data_dir, augmentations
        )

    else:
        raise ValueError

    # Data Loader
    # if args.dataset == 'shape_net_dishwasher':
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)

    return train_loader, eval_loader, data_shape, num_classes


def get_augmentation(augmentation, dataset, data_shape):
    h, w = data_shape
    if augmentation is None:
        pil_transforms = []
    elif augmentation == 'horizontal_flip':
        pil_transforms = [RandomHorizontalFlip(p=0.5)]
    elif augmentation == 'shift':
        pad_h, pad_w = int(0.07 * h), int(0.07 * w)
        if 'cityscapes' in dataset and 'large' in dataset:
            # Annoying, cityscapes images have a 3-border around every image.
            # This messes up shift augmentation and needs to be dealt with.
            assert h == 128 and w == 256
            print('Special cityscapes transform')
            pad_h, pad_w = int(0.075 * h), int(0.075 * w)
            pil_transforms = [CenterCrop((h - 2, w - 2)),
                              RandomHorizontalFlip(p=0.5),
                              Pad((pad_h, pad_w), padding_mode='edge'),
                              RandomCrop((h - 2, w - 2)),
                              Pad((1, 1), padding_mode='constant', fill=3)]

        else:
            pil_transforms = [RandomHorizontalFlip(p=0.5),
                              Pad((pad_h, pad_w), padding_mode='edge'),
                              RandomCrop((h, w))]
    elif augmentation == 'neta':
        assert h == w
        pil_transforms = [Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]
    elif augmentation == 'eta':
        assert h == w
        pil_transforms = [RandomHorizontalFlip(),
                          Pad(int(math.ceil(h * 0.04)), padding_mode='edge'),
                          RandomAffine(degrees=0, translate=(0.04, 0.04)),
                          CenterCrop(h)]

    # torchvision.transforms.s
    return pil_transforms


def get_data_shape(dataset):
    if dataset == 'bmnist':
        return (28, 28)

    elif dataset == 'mnist_1bit':
        return (28, 28)
