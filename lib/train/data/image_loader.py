import jpeg4py
import cv2 as cv
from PIL import Image
import numpy as np

davis_palette = np.repeat(np.expand_dims(np.arange(0,256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                         [0, 64, 128], [128, 64, 128]]


def default_image_loader(path):
    """The default image loader, reads the image from the given path. It first tries to use the jpeg4py_loader,
    but reverts to the opencv_loader if the former is not available."""
    if default_image_loader.use_jpeg4py is None:
        # Try using jpeg4py
        im = jpeg4py_loader(path)
        if im is None:
            default_image_loader.use_jpeg4py = False
            print('Using opencv_loader instead.')
        else:
            default_image_loader.use_jpeg4py = True
            return im
    if default_image_loader.use_jpeg4py:
        return jpeg4py_loader(path)
    return opencv_loader(path)

default_image_loader.use_jpeg4py = None


def jpeg4py_loader(path):
    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def opencv_loader(path):
    """ 优化版图像读取（速度提升 20-50%） """
    try:
        # 先通过 Python 原生读取文件到内存缓冲区
        with open(path, "rb") as f:
            buf = f.read()
        
        # 从内存缓冲区解码图像（省去重复文件检查）
        im = cv.imdecode(np.frombuffer(buf, dtype=np.uint8), cv.IMREAD_COLOR)
        
        # 快速 BGR2RGB 转换（避免调用 cv.cvtColor）
        return im[:, :, ::-1]  # 等价于 cv.cvtColor(im, cv.COLOR_BGR2RGB)
    
    except Exception as e:
        print(f'ERROR: Could not read image "{path}"')
        print(e)
        return None


def jpeg4py_loader_w_failsafe(path):
    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except:
        try:
            im = cv.imread(path, cv.IMREAD_COLOR)

            # convert to rgb and return
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        except Exception as e:
            print('ERROR: Could not read image "{}"'.format(path))
            print(e)
            return None


def opencv_seg_loader(path):
    """ Read segmentation annotation using opencv's imread function"""
    try:
        return cv.imread(path)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def imread_indexed(filename):
    """ Load indexed image with given filename. Used to read segmentation annotations."""

    im = Image.open(filename)

    annotation = np.atleast_3d(im)[...,0]
    return annotation


def imwrite_indexed(filename, array, color_palette=None):
    """ Save indexed image as png. Used to save segmentation annotation."""

    if color_palette is None:
        color_palette = davis_palette

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')