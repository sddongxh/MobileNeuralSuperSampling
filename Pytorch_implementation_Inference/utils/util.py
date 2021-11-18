import json
from PIL import Image
import torch
from torchvision.transforms import ToPILImage
show = ToPILImage()


if __name__ == '__main__':
    import torchvision.transforms as tf
    trans = tf.ToTensor()


def read_config(json_file):
    with open(json_file) as json_file:
        config = json.load(json_file)
    return config


def prepare_device(n_gpu_use):
    """setup GPU device if available. get gpu device indices which are used for DataParallel"""
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


# data (c, h, w)
def save_image(filename, data):
    with torch.no_grad():
        img = data.clone().detach().numpy()
        img = (img.transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")
        img = Image.fromarray(img)
        img.save(filename)


def unpackage(raba_x: torch.Tensor, rgba_y: torch.Tensor) -> torch.Tensor:
    h = raba_x.shape[1]
    w = raba_x.shape[2]
    bit_shift = torch.tensor([1.0, 1.0 / 256.0, 1.0 / (256.0 * 256.0), 1.0 / (256.0 * 256.0 * 256.0)])
    bit_shift = bit_shift.repeat(h, w, 1).permute(2, 0, 1)
    raba_x = (raba_x*255/256) * bit_shift
    raba_y = (rgba_y*255/256) * bit_shift
    x = raba_x.sum(0).unsqueeze(0) * 2 - 1
    y = raba_y.sum(0).unsqueeze(0) * 2 - 1
    res = torch.cat((x, y/-1), 0)
    return res

def unpackage_depth(depth: torch.Tensor) -> torch.Tensor:
    h = depth.shape[1]
    w = depth.shape[2]
    bit_shift = torch.tensor([1.0, 1.0 / 256.0, 1.0 / (256.0 * 256.0), 1.0 / (256.0 * 256.0 * 256.0)])
    bit_shift = bit_shift.repeat(h, w, 1).permute(2, 0, 1)
    raba_x = (depth * 255 / 256) * bit_shift
    x = raba_x.sum(0).unsqueeze(0)
    return x


