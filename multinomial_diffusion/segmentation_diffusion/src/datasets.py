from torch.utils.data import Dataset
import blobfile as bf
import numpy as np
import torch

def _list_voxel_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_voxel_files_recursively(full_path))
    return results

class FeatureDataset(Dataset):
    ''' 
    Dataset of the pixel representations and their labels.
    :param X_data: pixel representations [num_pixels, feature_dim]
    :param y_data: pixel labels [num_pixels]
    '''
    def __init__(
        self, 
        X_data: torch.Tensor, 
        y_data: torch.Tensor
    ):    
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class ImageLabelDataset(Dataset):
    ''' 
    :param data_dir: path to a folder with images and their annotations. 
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''
    def __init__(
        self,
        data_dir: str,
        resolution: int,
        num_images= -1,
        transform=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.transform = transform
        self.image_paths = _list_voxel_files_recursively(data_dir)
        self.image_paths = sorted(self.image_paths)

        if num_images > 0:
            print(f"Take first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]

        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        image_path = self.image_paths[idx]
        object_and_mask = np.load(image_path)
        

        # assert pil_image.size[0] == pil_image.size[1], \
        #        f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        tensor_object_and_mask = self.transform(object_and_mask)
        tensor_object = tensor_object_and_mask[0].unsqueeze(0)
        # Load a corresponding mask and resize it to (self.resolution, self.resolution)
        # label_path = self.label_paths[idx]
        # label = np.load(label_path).astype('uint8')
        # label = cv2.resize(
        #     label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        # )
        tensor_label = tensor_object_and_mask[1:,...]
        return tensor_object, tensor_label