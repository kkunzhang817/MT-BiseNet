
import base64
import cv2
import imutils
import json
import math
import os
import threading
import time

import numpy as np


class MyThread(threading.Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def get_result(self):
        try:
            return self.func(*self.args)
        except:
            return None


def start_thread(target, args):
    thread = threading.Thread(target=target, args=args)
    thread.setDaemon(True)
    thread.start()


def empty_img(size):
    return np.ones(size, dtype=np.uint8)


def read_img(file_path, flags=1):
    return cv2.imdecode(np.fromfile(file_path,dtype=np.uint8), flags)


def write_img(write_path, img):
    # if ":" in write_path:
    #     write_path = write_path.replace(":", "_")
    dir_path = os.path.dirname(write_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    cv2.imencode(".%s" % write_path.split(".")[-1], img)[1].tofile(write_path)


def show_video(img, name="tmp"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    key = cv2.waitKey(20)
    return key


def show_img(img, show_time = -1, name = "tmp"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    if show_time ==-1:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(show_time)
    if key == ord('q'): cv2.destroyWindow(name)
    elif key == ord('f'): cv2.destroyAllWindows()
    return key


def debug_img(img, show_time=0, name="debug"):
    """
    如果show_time为0,则不显示
    :param img:
    :param show_time:
    :param name:
    :return:
    """
    if show_time == 0:
        return
    show_img(img, show_time, name)


def debug_print(info, is_show=True):
    if not is_show:
        return
    print(info)


def resize_img(img, size):
    return cv2.resize(img, size, cv2.INTER_NEAREST)

def resize_no_new_pixel(src_img, out_h,out_w):
    dst_img = np.zeros((out_h,out_w))

    height = src_img.shape[0]
    width = src_img.shape[1]

    w_scale = float(width/ out_w)
    h_scale = float(height/ out_h)

    for j in range(out_h):
        for i in range(out_w):
            raw_w = int(i*w_scale)
            raw_h = int(j*h_scale)
            dst_img[j][i] = src_img[raw_h][raw_w]
    return  dst_img

def gamma_correction(img, gamma=1):
    look_up_table = np.empty((1,256), np.uint8)
    for i in range(256):
        look_up_table[0, i] = np.clip(pow(i/255.0, gamma)*255.0, 0, 255)
    return cv2.LUT(img, look_up_table)


def hist_contrast(img):
    out_min, out_max = 0, 255
    in_min, in_max = img.min(), img.max()
    a = (out_max-out_min) / max((in_max-in_min), 1e-8)
    b = out_min - a * in_min
    return np.uint8(img*a+b)


def contrast_brightness(img, alpha, beta):
    blank = np.zeros(img.shape, img.dtype)
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst


def bilateral_filter(img, d, c, s):
    return cv2.bilateralFilter(img, d, c, s)


def adaptive_threshold(img, block_size, c, method=cv2.ADAPTIVE_THRESH_MEAN_C, threshold_type=cv2.THRESH_BINARY):
    return cv2.adaptiveThreshold(bgr2gray(img), 255, method, threshold_type, block_size, c)


def otsu_threshold(img):
    return cv2.threshold(bgr2gray(img), 0, 255, cv2.THRESH_OTSU)[1]


def bin_threshold(img, value=128):
    return cv2.threshold(bgr2gray(img), value, 255, cv2.THRESH_BINARY)[1]


def reverse_img(img):
    return 255 - img


def erode(img, kernel_size, degree=1):
    return cv2.erode(img, np.ones(kernel_size, dtype=np.uint8), iterations=degree)


def dilate(img, kernel_size, degree=1):
    return cv2.dilate(img, np.ones(kernel_size, dtype=np.uint8), iterations=degree)


def open_op(img, kernel_size):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones(kernel_size, dtype=np.uint8))


def close_op(img, kernel_size):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones(kernel_size, dtype=np.uint8))


def rotate_img(img, angle, center=None):
    '''
    :param center: None => w//2, h//2
    :param angle: 小于0，顺时针
    '''
    return imutils.rotate(img, angle, center)


def rotate_point_list(point_list, base, angle):
    """
    :param point_list:
    :param base:
    :param angle: 大于0，顺时针
    :return:
    """
    angle = -angle
    point_mat, base_mat = np.array(point_list).astype(np.float16), np.array(base).astype(np.float16)
    ret_point_list = point_mat.copy()
    ret_point_list[:, 0] = (point_mat[:, 0]-base_mat[0]) * np.cos(np.pi / 180.0 * angle) - (point_mat[:, 1]-base_mat[1]) * np.sin(np.pi / 180.0 * angle) + base_mat[0]
    ret_point_list[:, 1] = (point_mat[:, 0]-base_mat[0]) * np.sin(np.pi / 180.0 * angle) + (point_mat[:, 1]-base_mat[1]) * np.cos(np.pi / 180.0 * angle) + base_mat[1]
    ret_point_list = ret_point_list.astype(np.int16)
    return ret_point_list.tolist()


def rotate_point_list_exactly(point_list, base, angle):
    """
    :param point_list:
    :param base:
    :param angle: 大于0，顺时针
    :return:
    """
    angle = -angle
    point_mat, base_mat = np.array(point_list).astype(np.float32), np.array(base).astype(np.float32)
    ret_point_list = point_mat.copy()
    ret_point_list[:, 0] = (point_mat[:, 0]-base_mat[0]) * np.cos(np.pi / 180.0 * angle) - (point_mat[:, 1]-base_mat[1]) * np.sin(np.pi / 180.0 * angle) + base_mat[0]
    ret_point_list[:, 1] = (point_mat[:, 0]-base_mat[0]) * np.sin(np.pi / 180.0 * angle) + (point_mat[:, 1]-base_mat[1]) * np.cos(np.pi / 180.0 * angle) + base_mat[1]
    return ret_point_list.tolist()


def calc_rotate_centre(pa0, pb0, pa1, pb1):
    x1, y1 = pa0
    x2, y2 = pb0
    x3, y3 = pa1
    x4, y4 = pb1
    '''
    Note: x1 != x2 , x3 != x4
    '''
    xc1, yc1 = (x1 + x2) / 2, (y1 + y2) / 2
    xc2, yc2 = (x3 + x4) / 2, (y3 + y4) / 2
    k1 = -1 / ((y2 - y1) / (x2 - x1))
    k2 = -1 / ((y4 - y3) / (x4 - x3))
    b1 = yc1 - k1 * xc1
    b2 = yc2 - k2 * xc2
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    print(x, y) # 临时测试
    return x, y


def draw_rectangle(img, l, t, r, b, color, thickness):
    return cv2.rectangle(gray2bgr(img), (l,t), (r,b), color, thickness)


def draw_circle(img, x, y, r, color, thickness):
    return cv2.circle(gray2bgr(img), (x,y), r, color, thickness)


def draw_line(img, x1, y1, x2, y2, color, thickness):
    return cv2.line(gray2bgr(img), (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def draw_cross(img, x, y, s, color, thickness):
    img = gray2bgr(img)
    img = draw_line(img, x-s, y, x+s, y, color, thickness)
    img = draw_line(img, x, y-s, x, y+s, color, thickness)
    return img


def draw_text(img, x, y, char, scale, color, thickness):
    return cv2.putText(gray2bgr(img), char, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def contour_length(contour):
    return cv2.arcLength(contour, True)


def contour_area(contour):
    return cv2.contourArea(contour)


def contour_angle(contour):
    angle = cv2.minAreaRect(contour)[-1]
    if angle < -45:
        angle += 90
    return angle


def contour_min_rect(contour):
    '''
    :return: [lt,lb,rt,rb], angle
    '''
    c, s, a = cv2.minAreaRect(contour)
    if a < -45:
        a += 90
    cx, cy = c
    w, h = s
    # w = max(s)
    # h = min(s)
    l = cx - w / 2
    t = cy - h / 2
    r = cx + w / 2
    b = cy + h / 2
    min_rect_corner = rotate_point_list([[l,t],[l,b],[r,t],[r,b]], [cx, cy], -a)
    return min_rect_corner, a


def contour_max_rect(contour):
    '''
    :return: x, y, w, h
    '''
    return cv2.boundingRect(contour)


def find_contour(img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE):
    contour_list, _ = cv2.findContours(img, mode, method)
    return contour_list


def draw_contour(img, contour_list, contour_index=-1, color=(0, 0, 255), thickness = 1):
    return cv2.drawContours(gray2bgr(img), contour_list, contour_index, color, thickness)


def bgr2gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def rgb2bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def bgr2rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gray2bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def gray2rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def bit_op(src1, src2, op):
    if type(op) is not str:
        return None
    if op.lower() == "and":
        return cv2.bitwise_and(src1, src2)
    elif op.lower() == "or":
        return cv2.bitwise_or(src1, src2)
    elif op.lower() == "not":
        return cv2.bitwise_not(src1)
    elif op.lower() == "xor":
        return cv2.bitwise_xor(src1, src2)
    else:
        return None


def parse_threshold_condition(dst, c_v):
    """
    :param dst:
    :param c_v:
    :return:
    """
    condition_type, min_threshold, max_threshold, is_necessary = c_v
    if condition_type == "max":
        condition_value = dst.max()
    elif condition_type == "min":
        condition_value = dst.min()
    else:
        if condition_type > 0:
            condition_value = np.sum(dst >= abs(condition_type))
        else:
            condition_value = np.sum(dst <= abs(condition_type))

    if min_threshold <= 1:
        min_threshold *= dst.size
    if max_threshold <= 1:
        max_threshold *= dst.size

    # print(condition_type, condition_value, min_threshold, max_threshold)

    if min_threshold <= condition_value <= max_threshold:
        return True, is_necessary

    return False, is_necessary


def my_threshold(img, para_dict, show_time=0, name="my_threshold"):
    dst = bgr2gray(img)

    if "DEBUG" in para_dict.keys():
        show_time = para_dict["DEBUG"]

    debug_img(dst, show_time, "_".join([name, "gray"]))
    for k, v in para_dict.items():
        #
        if k.isupper():
            continue
        #
        p, p_flag = v["p"], True
        for cp_k, cp_v in v.items():
            if "cp" in cp_k:
                ans, necessary = parse_threshold_condition(dst, cp_v)
                if not ans and necessary:
                    return None
                if not ans and not necessary:
                    p_flag = False
                    break
        if not p_flag:
            continue
        if "a" in k:
            dst = adaptive_threshold(dst, p[0], p[1])
        elif "b" in k:
            dst = bilateral_filter(dst, p[0], p[1], p[2])
        elif "c" in k:
            dst = close_op(dst, p[:2])
        elif "d" in k:
            dst = dilate(dst, p[:2], p[-1])
        elif "e" in k:
            dst = erode(dst, p[:2], p[-1])
        elif "g" in k:
            dst = gamma_correction(dst, p)
        elif "h" in k:
            dst = hist_contrast(dst)
        elif "o" in k:
            dst = open_op(dst, p[:2])
        elif "r" in k:
            dst = reverse_img(dst)
        elif "s" in k:
            dst = resize_img(dst, [p[0], p[1]])
        elif "t" in k:
            dst = otsu_threshold(dst)

        debug_img(dst, show_time, "_".join([name, k]))

        for pc_k, pc_v in v.items():
            if "pc" in pc_k:
                ans, necessary = parse_threshold_condition(dst, pc_v)
                if not ans and necessary:
                    return None
    return dst


def crop_img(img, centre_list, rotate_list, size):
    img_list = []
    img_h, img_w = img.shape[:2]
    for rotate_angle, (cx, cy) in zip(rotate_list, centre_list):
        top, bot, left, right = cy-size//2, cy+size//2, cx-size//2, cx+size//2
        if top < 0:
            bot -= top
            top = 0
        elif bot > img_h:
            top -= (bot - img_h)
            bot = img_h
        if left < 0:
            right -= left
            left = 0
        elif right > img_w:
            left -= (right - img_w)
            right = img_w
        if rotate_angle == 0:
            img_list.append(img[top:bot, left:right])
        elif rotate_angle == -90:
            img_list.append(cv2.rotate(img[top:bot, left:right], cv2.ROTATE_90_COUNTERCLOCKWISE))
        elif rotate_angle == 90:
            img_list.append(cv2.rotate(img[top:bot, left:right], cv2.ROTATE_90_CLOCKWISE))
        elif rotate_angle == 180:
            img_list.append(cv2.rotate(img[top:bot, left:right], cv2.ROTATE_180))
    return img_list


def crop_coord2img_coord(crop_coord_list, crop_centre_list, crop_size):
    if len(crop_coord_list)==0: return crop_coord_list
    crop_coord_mat, crop_centre_mat = np.array(crop_coord_list), np.array(crop_centre_list)
    crop_start = crop_centre_mat - crop_size // 2
    return (crop_start + crop_coord_mat).tolist()


def combine_close_coord(coord_list, min_distance):
    coord_list_copy = coord_list.copy()
    ret_coord_list = []
    while len(coord_list_copy) > 1:
        coord = coord_list_copy.pop()
        distance_list = calc_distance(coord, coord_list_copy)
        if min(distance_list) > min_distance:
            ret_coord_list.append(coord)
    ret_coord_list.extend(coord_list_copy)
    return ret_coord_list


def split_list(scr_list, max_num):
    ret_list, sub_list = [], []
    for ele_index, element in enumerate(scr_list):
        if (ele_index+1) % max_num == 0:
            sub_list.append(element)
            ret_list.append(sub_list)
            sub_list = []
        else:
            sub_list.append(element)
    if len(sub_list) > 0:
        ret_list.append(sub_list)
    return ret_list


def equal_divide(length, num):
    cell = (length-1)/(num+1)
    return [int(round(each*cell)) for each in range(1, num+1)]


def cut_edge(img, main_value):
    h, w = img.shape[:2]
    for top in range(h):
        if (img[top]).tolist().count(main_value) > 0: break
    for bot in range(h-1, 0, -1):
        if (img[bot]).tolist().count(main_value) > 0: break
    for left in range(w):
        if (img[:, left]).tolist().count(main_value) > 0: break
    for right in range(w-1, 0, -1):
        if (img[:, right]).tolist().count(main_value) > 0: break
    return left, top, right, bot


def calc_angle_by_vector(v1, v2):
    '''
    计算由v1到v2的夹角，v1到v2顺时针，角度为正; 逆时针，角度为负
    :param v1: [x1, y1, x2, y2]
    :param v2: [x1, y1, x2, y2]
    :return: angle between v1 and v2
    '''
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = angle1 * 180/math.pi
    angle2 = math.atan2(dy2, dx2)
    angle2 = angle2 * 180/math.pi
    return angle1 - angle2


def calc_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1)-np.array(p2))**2, -1)).tolist()


def sort_coord(coord_list):
    coord_list.sort(key=lambda k:k[1])
    y_t = (coord_list[0][1] + coord_list[-1][1])/2
    top_coord_list, bot_coord_list = [], []
    for x,y in coord_list:
        if y<y_t: top_coord_list.append([x,y])
        else: bot_coord_list.append([x,y])
    top_coord_list.sort(key=lambda k:k[0])
    bot_coord_list.sort(key=lambda k:k[0])
    coord_list = top_coord_list.copy()
    coord_list.extend(bot_coord_list)
    return [top_coord_list[0], bot_coord_list[0], top_coord_list[-1], bot_coord_list[-1]], coord_list


def de_warp_img(img, mtx_path, dist_path):
    camera_matrix = np.load(mtx_path)
    dist_coefs = np.load(dist_path)
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    img = cv2.undistort(img, camera_matrix, dist_coefs, None, new_camera_mtx)
    return img


def img2b64str(img):
    img = np.uint8(img)
    img_bytes = cv2.imencode('.jpg', img)[1].tostring()
    b64en_bytes = base64.b64encode(img_bytes)
    b64en_str = str(b64en_bytes, encoding='utf8')
    return b64en_str


def save_img(path, name, img):
    t = time.localtime()
    name = "%d_%d_%d_%d_%d_%d_%s.jpg" % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, name)
    if not os.path.exists(path):
        os.makedirs(path)
    write_img(os.path.join(path, name), img)


def read_json(json_path):
    """
    返回json字典
    :param json_path: json file path
    :return: json 2 dict
    """
    with open(json_path, 'r') as f:
        para_dict = json.load(f)
    return para_dict


def write_dict2json(json_path, para_dict):
    """
    将dict写入json
    :param json_path: json file path
    :param para_dict: dict
    :return:
    """
    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path))
    para_str = str(para_dict).replace(": False", ": false").replace(": True", ": true").replace("\'", "\"")
    tab_cnt, entry_cnt, tab = 0, 0, '  '
    with open(json_path, 'w') as f:
        for char in para_str:
            if char == '[':
                entry_cnt += 1
            elif char == ']':
                entry_cnt -= 1
            if char == ',' and entry_cnt == 0:
                f.write("%s\n%s" % (char, tab * tab_cnt))
            elif char == '{':
                tab_cnt += 1
                f.write("%s\n%s" % (char, tab * tab_cnt))
            elif char == '}':
                tab_cnt -= 1
                f.write("\n%s%s" % (tab * tab_cnt, char))
            else: f.write('%s' % char)


def split_img(img, centre, size):
    """
    裁剪图像，获取裁剪图像列表
    :param img:
    :param centre:
    :param size:
    :return:
    """
    h, w = img.shape[:2]
    ret_list, bbox_list = [], []
    for (x,y) in centre:
        l, r, t, b = x - size//2, x + size//2, y - size//2, y + size//2
        if l < 0:
            r -= l
            l = 0
        elif r > w:
            l += (w - r)
            r = w

        if t < 0:
            b -= t
            t = 0
        elif b > h:
            t += (h - b)
            b = h
        ret_list.append(img[t:b, l:r])
        bbox_list.append([t,b,l,r])
    return ret_list, bbox_list


def filter_nearest_point(data_list, point_list, num_list):
    data = np.array(data_list)
    ret_list = []
    for num, point in zip(num_list, point_list):
        ret_list.append(np.sum((data - point) ** 2, 1).argsort()[:num])
    return ret_list


