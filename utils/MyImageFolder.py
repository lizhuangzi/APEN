import os
import random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def rebuild_images(root, images):
    rebud_images = []
    for image in images:
        path = os.path.join(root, image[0])
        item = (path, image[1])
        rebud_images.append(item)

    return rebud_images

def getClasses_idxs_Imgs(root):
    classes, class_to_idx = find_classes(root)
    imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)

    return classes, class_to_idx, imgs

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)

    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for _, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(target, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


class process(data.Dataset):
    def __init__(self, root, transform=None,transform1=None,transform2=None, target_transform=None,
                 loader=pil_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.transform1 = transform1
        self.transform2 = transform2
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(os.path.join(self.root,path))

        if self.transform is not None:
            img = self.transform(img)
            imhr = self.transform1(img)
            imlr =self.transform2(img)
            imlr=self.transform1(imlr)


        return imhr,imlr, target

    def __len__(self):
        return len(self.imgs)


