import os
import glob
import random
from PIL import Image
from my_utils.utils import *
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

number_list  = [1,3,5]
ori_img_path = glob.glob(r'D:\zkk_project\segmentation\segment\data\Leaf\train\image\*.png')
ori_mask_dir = r'D:\zkk_project\segmentation\segment\data\Leaf\train\label_0_255'
choose_img_path = r'D:\software\baidudownload\segment\data\leaf_cut\train\image\*.png'
choose_img_list = glob.glob(choose_img_path)
choose_mask_path = r'D:\software\baidudownload\segment\data\leaf_cut\train\label'
for number in number_list:
    for img_path in ori_img_path:
        image = Image.open(img_path)  # RGB
        segmap = Image.open(os.path.join(ori_mask_dir,img_path.split("\\")[-1]))

        image = np.array(image)
        image = rgb2bgr(image)
        # show_img(rgb2bgr(image))
        segmap = SegmentationMapsOnImage(np.array(segmap), shape=image.shape)
        # print(segmap.shape)
        segmaps_aug_i_ = segmap.get_arr()
        # show_img(segmaps_aug_i_)
        choose_list = random.sample(choose_img_list, number)
        H, W, _ = image.shape
        for choose_img in choose_list:
            img_need_name = choose_img.split("\\")[-1]
            segmap_need = os.path.join(choose_mask_path,img_need_name)
            img_need = read_img(choose_img)
            segmap_open = Image.open(segmap_need)
            segmap_open = segmap_open.convert('L')
            segmap_xianshi = SegmentationMapsOnImage(np.array(segmap_open), shape=img_need.shape)
            segmaps_need_aug_i_ = segmap_xianshi.get_arr()
            segmaps_need_aug_i_[segmaps_need_aug_i_ > 0] = 255
            h, w, _ = img_need.shape
            y, x = random.randint(0,H - h),random.randint(0, W - w)
            image[y: y + h, x: x + w] = img_need
            segmaps_aug_i_[y: y + h, x: x + w] = segmaps_need_aug_i_
        write_img(os.path.join(r'~/Downloads/segment/data/Leaf/train/aug/JPEG_AUG',"_{}_".format(str(number)) + img_path.split("\\")[-1]),image)
        write_img(os.path.join(r'~/Downloads/segment/data/Leaf/train/aug/SegmentationClass_AUG',"_{}_".format(str(number)) + img_path.split("\\")[-1]),segmaps_aug_i_)
        # show_img(image)
        # show_img(segmaps_aug_i_)
        # show_img(segmaps_need_aug_i_)