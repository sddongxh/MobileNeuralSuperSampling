import os
import time
import torch
import torch.nn.functional as F
from model.MNSS_Q import MNSS
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from dataloader.sp_data_loader import Mutil_IMG2IMG
from utils.util import prepare_device, read_config, save_image


def BackwardWarping(Image: torch.Tensor, motion_T: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Image B*C*H*W
    motion B*H*W*2
    """
    return F.grid_sample(Image, grid - motion_T, padding_mode='border', mode='bilinear', align_corners=True)


def main(config):
    mode = '1080p'
    if mode == '720p':
        w = 1280
        h = 720
    elif mode == '1080p':
        w = 1920
        h = 1080
    else:
        w = 900
        h = 900

    dataset = Mutil_IMG2IMG(root_dir=config["test_root_dir"], batch_size=config["test_batch_size"])
    valid_data_loader = DataLoader(dataset, batch_size=config["test_batch_size"], shuffle=config["shuffle"],
                             num_workers=config["num_workers"])

    device, device_ids = prepare_device(config['n_gpu'])
    model = MNSS().to(device)

    test_model_path = config['checkpoint']
    cur_state_dict = torch.load(test_model_path)
    model.load_state_dict(cur_state_dict)
    id = 0
    with torch.no_grad():
        previous_x_RGBD = torch.zeros([config["test_batch_size"], 4, h, w]).cuda()
        h_line = torch.linspace(-1, 1, h)
        w_line = torch.linspace(-1, 1, w)
        meshx, meshy = torch.meshgrid([h_line, w_line])
        grid = torch.stack((meshy, meshx), 2)
        grid = grid.unsqueeze(0).cuda()
        tqdm_bar = tqdm(valid_data_loader, desc=f'Testing ', total=int(len(valid_data_loader)))
        for batch_idx, (x_view, x_depth, x_flow, target, image_name) in enumerate(tqdm_bar):
            x_view = x_view.to(device)
            x_depth = x_depth.to(device)
            x_flow = x_flow.to(device)
            x_RGBD = torch.cat((x_view, x_depth), dim=1)

            # upscale Frame 0
            if image_name[0] == '1600.png':
                previous_RGBD = F.interpolate(x_RGBD, scale_factor=3, mode="bilinear", align_corners=False)
                previous_x_RGBD[:, :, 1:h-1, 1:w-1] = previous_RGBD[:, :, 1:h-1, 1:w-1]
                id = id + 1
                continue


            torch.cuda.synchronize()  # GPU sync
            start = time.time()

            # 0 upsampling
            x_RGBD_upsampling = previous_x_RGBD.clone()
            view_RGBD = F.interpolate(x_RGBD, scale_factor=3, mode="bilinear", align_corners=False)
            motion_upsampled = F.interpolate(x_flow, scale_factor=3, mode="bilinear", align_corners=False)

            # aligning
            x_list = [0, 0, 0, 1, 2, 2, 2, 1, 1]
            y_list = [0, 1, 2, 2, 2, 1, 0, 0, 1]
            u = x_list[id % 9]
            v = y_list[id % 9]
            x_RGBD_upsampling[:, :, u:h-2 + u, v:w-2 + v] = view_RGBD[:, :, 1:h-1, 1:w-1]
            motion_T = 2 * motion_upsampled.permute([0, 2, 3, 1])
            previous_RGBD_Warped = BackwardWarping(previous_x_RGBD, motion_T, grid)

            HR_RGBD = model(x_RGBD_upsampling, previous_RGBD_Warped)

            torch.cuda.synchronize()  # GPU sync
            end = time.time()
            time_c = end - start
            print('Frame time:', time_c, 's')

            id += 1

            previous_x_RGBD = torch.cat((HR_RGBD.detach(), x_RGBD_upsampling[:, 3, :, :].unsqueeze(1)), dim=1)
            # save images
            pred = HR_RGBD.clone().cpu()
            for i in range(pred.shape[0]):
                file_name = image_name[i]
                img = pred[i]
                save_name = os.path.join(config['save_dir'], config['checkpoint'][-6:-4])
                if not os.path.exists(save_name):
                    os.makedirs(save_name)
                save_name = os.path.join(save_name, file_name)
                save_image(save_name, img)


if __name__ == '__main__':
    config = read_config('mutil_config.json')
    main(config)
