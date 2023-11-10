from utils import *
from preprocess import *
from postprocess import *
from PIL import Image, ImageDraw
from configs import *


def prediction(session, image, cfg):
    """
    Perform object detection on an input image using a loaded model session.

    Args:
        session (object): The loaded inference session with a pre-trained model.
        image (ndarray): An input image in numpy array format.
        cfg (object): A configuration object containing detection parameters.

    Returns:
        ndarray: An array of detected objects with their bounding boxes and labels.
    """
    image, ratio, (padd_left, padd_top) = resize_and_pad(
        image, new_shape=cfg.image_size
    )
    img_norm = normalization_input(image)
    pred = infer(session, img_norm)
    pred = postprocess(pred, cfg.conf_thres, cfg.iou_thres)[0]
    paddings = np.array([padd_left, padd_top, padd_left, padd_top])
    pred[:, :4] = (pred[:, :4] - paddings) / ratio
    return pred


def visualize(image, pred):
    """
    Draw bounding boxes around detected objects on an image.

    Args:
        image (Image): A PIL Image object.
        pred (ndarray): An array of detected objects with their bounding boxes and labels.

    Returns:
        Image: A PIL Image with bounding boxes drawn around detected objects.
    """
    img_ = image.copy()
    drawer = ImageDraw.Draw(img_)
    for p in pred:
        x1, y1, x2, y2, _, id = p
        id = int(id)
        drawer.rectangle((x1, y1, x2, y2), outline=IDX2COLORs[id], width=3)
    return img_
