import os

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

folder_path = '/home/guest/Desktop/Cutie/four_hole2_rgb/'
save_path = '/home/guest/Desktop/Cutie/results/four_hole2_rgb'
def read_images_masks(ti, folder_path, folders, img_name):
    if ti == 0:
        images = []
        masks = []
        final_images = []
        for folder_i in folders:
            img =  to_tensor(Image.open(f'{folder_path}/{folder_i}/images/{img_name}'))
            mask = np.array(Image.open(f'{folder_path}/{folder_i}/masks/000.png').convert('L'))
            final_img = Image.open(f'{folder_path}/{folder_i}/images/{img_name}')
            images.append(img)
            masks.append(mask)
            final_images.append(final_img)
        masks = torch.from_numpy(np.stack(masks)).cuda()
        images = torch.stack(images).cuda().float()
        return images, masks, final_images
    else:
        images = []
        final_images = []
        for folder_i in folders:
            img =  to_tensor(Image.open(f'{folder_path}/{folder_i}/images/{img_name}'))
            final_img = Image.open(f'{folder_path}/{folder_i}/images/{img_name}')
            images.append(img)
            final_images.append(final_img)
        images = torch.stack(images).cuda().float()
        return images, None, final_images

def combine_images_horizontally(image_list):
    heights, widths = zip(*(img.size for img in image_list))
    total_width = sum(widths)
    max_height = sum(heights)
    combined_image = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in image_list:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    return combined_image

def create_final_image(rgb_list, mask_list):
    # Ensure mask images are converted to "L" mode (grayscale)
    # mask_list = [img.convert('L') for img in mask_list]
    
    combined_rgb = combine_images_horizontally(rgb_list)
    combined_mask = combine_images_horizontally(mask_list)
    
    final_width =combined_rgb.width * len(rgb_list)
    final_height = combined_rgb.height + combined_mask.height
    
    final_image = Image.new('RGB', (final_width, final_height))
    final_image.paste(combined_rgb, (0, 0))
    final_image.paste(combined_mask.convert('RGB'), (0, combined_rgb.height))
    
    return final_image

@torch.inference_mode()
@torch.cuda.amp.autocast()
def main():
    # obtain the Cutie model with default parameters -- skipping hydra configuration
    cutie = get_default_model()
    # Typically, use one InferenceCore per video
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    # ordering is important
    folders = sorted(os.listdir(folder_path))
    timesteps = sorted(os.listdir(f'{folder_path}/{folders[0]}/images'))   
    objects = [255]


    for ti, image_name in enumerate(timesteps):
        # load the image as RGB; normalization is done within the model
        image, mask, final_img = read_images_masks(ti, folder_path=folder_path, 
                                        folders=folders, 
                                        img_name=image_name)
        if ti == 0:
            # if mask is passed in, it is memorized
            # if not all objects are specified, we propagate the unspecified objects using memory
            output_prob = processor.step_multi(image, mask, objects=objects)
        else:
            # otherwise, we propagate the mask from memory
            output_prob = processor.step_multi(image)

        # convert output probabilities to an object mask
        mask = processor.output_prob_to_mask_batch(output_prob)
        out_mask = []
        for idx, mask_i in enumerate(mask):
            print(ti, idx, torch.unique(mask_i))
            Image.fromarray(mask_i.cpu().numpy().astype(np.uint8)).convert('L').save(f'{save_path}/masks_'+'{:03}'.format(ti)+f'_{idx}.png')
        # final_image =  create_final_image(out_mask, final_img)
        # final_image.convert('L').save(f'{save_path}/'+ '{:03}'.format(ti)+'.png')

main()
