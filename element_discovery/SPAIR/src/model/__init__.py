from .space.space import Space
import os
import datetime
from .tensorboardX import SummaryWriter

__all__ = ['get_model']

def get_model(cfg):
    """
    Also handles loading checkpoints, data parallel and so on
    :param cfg:
    :return:
    """
    
    model = None
    if cfg.model == 'SPACE':
        log_dir = os.path.join(cfg.checkpointdir, datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        log_dir = os.path.abspath(log_dir)
        writer = SummaryWriter(log_dir)

        image_shape = [3, 128, 128]
        device = 'cpu'
        model = Space(image_shape, writer, device)     
    return model
