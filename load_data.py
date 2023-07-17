import os
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader,Dataset
from my_utils.utils import *

def find_classes(directory):
    img_name_list  = os.listdir(directory)
    classes = sorted(img_name for img_name in img_name_list )
    classes = sorted(list(set(classes)))
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    class_idx = [i for i, cls_name in enumerate(classes)]
    return classes, class_to_idx, class_idx

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)

    plt.show()

CLASSES = ['background','butterfly']

class Dataset_interface(Dataset):
    def __init__(self, images_dir, masks_dir, classes = None, model = None, augmentation = None,augmentation_color = None, preprocessing = None, image_size = 640):
        super(Dataset_interface, self).__init__()
        _butterfly_root  = "../segment/data/largebutterfly"
        _butterfly_root_class = "../segment/data/largebutterfly/JPEGImages"
        self.classes, self.class_to_idx, self.classes_idx = find_classes(os.path.join(_butterfly_root_class))
        if model == "train":
            _split_f_dir = [ os.path.join(_butterfly_root,'train.txt')]
        elif model == "val":
            _split_f_dir = [ os.path.join(_butterfly_root,'val.txt')]
        else:
            _split_f_dir = [ os.path.join(_butterfly_root,'test.txt')]
        img_path_list = []
        for _split_f in _split_f_dir:
            with open (os.path.join(_split_f), 'r') as lines:
                for line in lines:
                     img_path_list.append(os.path.join(line.rstrip('\n')))

        self.images_fps = [os.path.join( _butterfly_root, images_dir, img_name) for img_name in img_path_list]
        self.masks_fps = [os.path.join( _butterfly_root, masks_dir, img_name.split(".")[0] + "_mask.png") for img_name in img_path_list]


        # convert str names to class values on masks
        self.class_values = [CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.augmentation_color = augmentation_color
        self.image_size = (image_size,image_size)

    def __getitem__(self, index):

        image= read_img(self.images_fps[index])
        image = resize_img(image,self.image_size)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask = read_img(self.masks_fps[index],cv2.IMREAD_GRAYSCALE)
        mask = resize_no_new_pixel(mask,self.image_size[0],self.image_size[1])

        masks = [(mask == v * 255) for v in self.class_values]
        mask = np.stack(masks,axis= -1).astype('float')
        class_label = self.images_fps[index].split("/")[-2]
        label = self.class_to_idx[class_label]

        #  apply augmentations shape
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image_1 = image
        mask_1 = mask

        #  apply augmentations color
        if self.augmentation_color:
            sample_1 = self.augmentation_color(image=image,mask = mask)
            image_1, mask_1= sample_1['image'],sample_1['mask']


        # apply preprocessing
        if self.preprocessing:
            preces = self.preprocessing(image = image, mask=mask)
            image, mask = preces['image'], preces['mask']

            preces_1 = self.preprocessing(image = image_1, mask=mask_1)
            image_1, mask_1= preces_1['image'], preces_1['mask']


        return image_1,image_1,mask,label

    def __len__(self):

        return len(self.images_fps)


