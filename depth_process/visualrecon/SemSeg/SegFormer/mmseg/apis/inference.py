import matplotlib.pyplot as plt
import numpy as np
import mmcv
import torch
import cv2
import sys
from pathlib import Path
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
sys.path.append(str(Path(__file__).parent.parent.parent))
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""
    def __init__(self, scale) -> None:
        self.scale = scale

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        if self.scale != 1.0:
            img = mmcv.imresize(img, (int(img.shape[1]*self.scale), int(img.shape[0]*self.scale)))
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_segmentor(model, img, scale=1.0):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    # test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = [LoadImage(scale)] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


def show_result_pyplot(model, img, result, palette=None, fig_size=(15, 10)):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, palette=palette, show=False)
    cv2.imwrite('demo_res.png', mmcv.bgr2rgb(img))
    # plt.figure(figsize=fig_size)
    # plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()

def save_results(model, img, result, palette=None, save_path='./', result_name='demo.jpg'):
    """save the results (visualized and raw results) in save_path
    """
    if hasattr(model, 'module'):
        model = model.module
    # img = model.show_result(img, result, palette=palette, show=False)
    img = model.show_result(img, result, palette=palette, show=False)
    mask = result[0] >= 10
    mask = np.array(mask, dtype=np.uint8)
    # kernel = np.ones((4,4))
    # mask_d = cv2.dilate(mask, kernel, iterations=1)
    # mask_d = mask
    cv2.imwrite(f'{save_path}/visualize/{result_name}', mmcv.bgr2rgb(img))
    cv2.imwrite(f'{save_path}/mask/{result_name}', mask*255)
    
def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()    

def save_results_all(model, img, result, palette=None, save_path='./', result_name='demo.jpg'):
    """save the results (visualized and raw results) in save_path
    """

    '''
    if hasattr(model, 'module'):
        model = model.module
    img, color_seg = model.show_result(img, result, palette=palette, show=False)
    '''

    # kernel = np.ones((4,4))
    # mask_d = cv2.dilate(mask, kernel, iterations=1)
    # mask_d = mask

    # mask = result[0] >= 8  # equal to seg
    # mask = np.array(mask, dtype=np.uint8)

    '''
    # cv2.imwrite(f'{save_path}/visualize/{result_name}', mmcv.bgr2rgb(img))
    # cv2.imwrite(f'{save_path}/color_seg/{result_name}', mmcv.bgr2rgb(color_seg))
    '''
    # cv2.imwrite(f'{save_path}/mask/{result_name}', mask*255)
    save_pfm(f"{save_path}/seg/{result_name.split('.')[0]}.pfm", np.array(result[0], dtype=np.float32))