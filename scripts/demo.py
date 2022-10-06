import os
import sys
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from dataclasses import dataclass
sys.path.append('.')

from training.config import get_config
from training.inference import Inference
from training.utils import create_logger, print_args

@dataclass
class Args:
    save_path: str = "result"
    load_path: str = "ckpts/sow_pyramid_a5_e3d2_remapped.pth"
    device: torch.device = torch.device('cuda:0')


def transfer(config,
             source_name,
             reference_name=None,
             reference_name_lip=None,
             reference_name_eye=None,
             reference_name_skin=None,
             source_dir="assets/images/non-makeup",
             reference_dir="assets/images/makeup",
             ):
    args = Args()

    inference = Inference(config, args, 'ckpts/sow_pyramid_a5_e3d2_remapped.pth')


    source_img = Image.open(os.path.join(source_dir, source_name)).convert('RGB')
    if reference_name is not None:
        reference_img = Image.open(os.path.join(reference_dir, reference_name)).convert('RGB')
        result = inference.transfer(source_img, reference_img, postprocess=True)

    else:
        reference_parts = [
            reference_name_lip,
            reference_name_eye,
            reference_name_skin,
        ]

        refs = [
            Image.open(os.path.join(reference_dir, reference_name_part)).convert('RGB')
            if reference_name_part is not None
            else source_img
            for reference_name_part in reference_parts
        ]

        result = inference.joint_transfer(source_img, *refs, postprocess=True) 
        
    if result is None:
        print("smth wrong with images")
        return

    source_img = np.array(source_img)
    h, w, _ = source_img.shape
    result = result.resize((h, w)); result = np.array(result)
    save_path = os.path.join("result", "transfer.png")
    Image.fromarray(result.astype(np.uint8)).save(save_path)


if __name__ == "__main__":
    if not os.path.exists("result"):
        os.makedirs("result")
    
    config = get_config()
    #transfer(config,
    #         source_name="source_1.png",
    #         reference_name="reference_1.png")

    transfer(config,
             source_name="source_1.png",
             reference_name_lip="reference_1.png")
