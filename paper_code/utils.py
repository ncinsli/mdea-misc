from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

totensor = T.ToTensor()
topil = T.ToPILImage()

def recover_image(image, init_image, mask, background=True):
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)
    print(f"Image shape in prepare_mask_and_masked_image: {image.shape}")
    print(f"Mask shape in prepare_mask_and_masked_image: {mask.shape}")
    print(f"Masked image shape in prepare_mask_and_masked_image: {masked_image.shape}")
    return mask, masked_image

def prepare_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image[0]

def recover_image_add_perturbation_torch(adv_image, init_image, mask_image, cur_mask_image, device='cpu'):
    adv_tensor = totensor(adv_image)
    init_tensor = totensor(init_image)
    mask_tensor = totensor(mask_image)
    cur_mask_tensor = totensor(cur_mask_image)
    topil(init_tensor).save("output/outinit_tensor.png")
    topil(mask_tensor).save("output/outmask.png")
    topil(adv_tensor).save("output/outadv.png")
    topil(cur_mask_tensor).save("output/outcur_mask_image.png")
    temp_tensor = adv_tensor - cur_mask_tensor
    temp_image = topil(temp_tensor)
    temp_image.save("output/out2.png")
    result = init_tensor + temp_tensor
    recovered_image = topil(result)
    recovered_image.save("output/outrecovered_image.png")
    return recovered_image
 
