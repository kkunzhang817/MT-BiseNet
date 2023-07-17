import os
import random
import glob
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image
from my_utils.utils import *

class ImageAugmentation(object):
    def __init__(self,  image_aug_dir, segmentationClass_aug_dir, image_start_num):
        self.image_aug_dir = image_aug_dir
        self.segmentationClass_aug_dir = segmentationClass_aug_dir
        self.image_start_num = image_start_num  # 增强后图片的起始编号

        if not os.path.exists(self.image_aug_dir):
            os.mkdir(self.image_aug_dir)
        if not os.path.exists(self.segmentationClass_aug_dir):
            os.mkdir(self.segmentationClass_aug_dir)

    # def seed_set(self, seed=1):
    #     np.random.seed(seed)
    #     random.seed(seed)
    #     ia.seed(seed)

    def array2p_mode(self, alpha_channel):
        """alpha_channel is a binary image."""
        # assert set(alpha_channel.flatten().tolist()) == {0, 1}, "alpha_channel is a binary image."


        alpha_channel[alpha_channel > 0] = 255
        h, w = alpha_channel.shape
        image_arr = np.zeros((h, w, 3))
        image_arr[:, :, 0] = alpha_channel
        img = Image.fromarray(np.uint8(image_arr))
        img_p = img.convert("L")
        return img_p


    def augmentor(self, image):
        height, width, _ = image.shape

        seq = iaa.Sequential([
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),  # mirror
            # iaa.Multiply((0.7, 1.2)),  # change brightness, doesn't affect BBs 改变亮度
            # iaa.GaussianBlur(sigma=(0, 0.2)),  # iaa.GaussianBlur(0.5),
            # iaa.Sharpen(alpha=(0.0,0.7), lightness=(0.9, 1.2)),       # 锐化
            # iaa.ContrastNormalization((0.8, 1.2), per_channel=False),     # 对比度
            # iaa.WithChannels(),
            # iaa.Invert(0.02, per_channel=True),
            # iaa.Add((-3, 3), per_channel=0.1),

            iaa.Affine(
                translate_px={"x": 10, "y": 10},
                scale=(0.9, 1.2),
                rotate=(-70, 70)
            )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
        ])

        return seq



    def augment_img(self, image_name, segmap_name):
        # 1.Load an image.
        image = Image.open(image_name)  # RGB
        segmap = Image.open(segmap_name)  # P

        name = self.image_start_num
        image.save(self.image_aug_dir + name + ".png")
        segmap.save(self.segmentationClass_aug_dir + name + ".png")


        image = np.array(image)
        segmap = SegmentationMapsOnImage(np.array(segmap), shape=image.shape)


        seq = self.augmentor(image)


        for _, value in enumerate(range(2)):

            name = self.image_start_num + "_{}".format(str(_))

            images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
            images_aug_i = Image.fromarray(images_aug_i)
            images_aug_i.save(self.image_aug_dir + name + ".png")
            segmaps_aug_i_ = segmaps_aug_i.get_arr()
            segmaps_aug_i_[segmaps_aug_i_>0] = 255
            write_img(self.segmentationClass_aug_dir + name + ".png",segmaps_aug_i_)






if __name__ == "__main__":
    import glob
    import tqdm
    image_aug_dir = r"D:\\zkk_project\\segmentation\\segment\\data\\Leaf\\train\\aug\\JPEG_AUG\\"
    segmentationClass_aug_dir = r"D:\\zkk_project\\segmentation\\segment\\data\\Leaf\\train\\aug\\SegmentationClass_AUG\\"
    image_dir = glob.glob(r"D:\zkk_project\segmentation\segment\data\Leaf\train\image\*.png")

    for img_path in tqdm.tqdm(image_dir):
        img_name = img_path.split("\\")[-1]
        mask_path = os.path.join(r"D:\zkk_project\segmentation\segment\data\Leaf\train\label_0_255",img_name)
        image_augmentation = ImageAugmentation(image_aug_dir, segmentationClass_aug_dir, img_name[:-4])
        image_augmentation.augment_img(img_path, mask_path)
