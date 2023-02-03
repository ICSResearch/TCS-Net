import os
import torch
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

import config
import models


def save_log(recon_root, name_dataset, name_image, psnr, ssim, manner, consecutive=True):
    if not os.path.isfile(f"{recon_root}/Res_{name_dataset}_{manner}.txt"):
        log = open(f"{recon_root}/Res_{name_dataset}_{manner}.txt", 'w')
        log.write("=" * 120 + "\n")
        log.close()
    log = open(f"{recon_root}/Res_{name_dataset}_{manner}.txt", 'r+')
    if consecutive:
        old = log.read()
        log.seek(0)
        log.write(old)
    log.write(
        f"Res {name_image}: PSNR, {round(psnr, 2)}, SSIM, {round(ssim, 4)}\n")
    log.close()


def testing(network, val, manner=config.para.manner, save_img=config.para.save):
    """
    The pre-processing before TCS-Net's forward propagation and the testing platform.
    """
    recon_root = "reconstructed_images"
    if not os.path.isdir(recon_root):
        os.mkdir(recon_root)
    datasets = ["Set11"] if val else ["McM18", "LIVE29", "General100", "OST300"]  # Names of folders (testing datasets)
    with torch.no_grad():
        for one_dataset in datasets:
            if not os.path.isdir(f"{recon_root}/{one_dataset}"):
                os.mkdir(f"{recon_root}/{one_dataset}")

            test_dataset_path = f"dataset/test/{one_dataset}"

            # Grey manner
            if manner == "grey":
                recon_dataset_path_grey = f"{recon_root}/{one_dataset}/grey/"
                recon_dataset_path_grey_rate = f"{recon_root}/{one_dataset}/grey/{config.para.rate}"
                if not os.path.isdir(recon_dataset_path_grey):
                    os.mkdir(recon_dataset_path_grey)
                if not os.path.isdir(recon_dataset_path_grey_rate):
                    os.mkdir(recon_dataset_path_grey_rate)
                sum_psnr, sum_ssim = 0., 0.
                for _, _, images in os.walk(f"{test_dataset_path}/rgb/"):
                    for one_image in images:
                        name_image = one_image.split('.')[0]
                        x = cv.imread(f"{test_dataset_path}/rgb/{one_image}", flags=cv.IMREAD_GRAYSCALE)
                        x_ori = x
                        x = torch.from_numpy(x / 255.).float()
                        h, w = x.size()

                        lack = config.para.block_size - h % config.para.block_size if h % config.para.block_size != 0 else 0
                        padding_h = torch.zeros(lack, w)
                        expand_h = h + lack
                        inputs = torch.cat((x, padding_h), 0)

                        lack = config.para.block_size - w % config.para.block_size if w % config.para.block_size != 0 else 0
                        expand_w = w + lack
                        padding_w = torch.zeros(expand_h, lack)
                        inputs = torch.cat((inputs, padding_w), 1).unsqueeze(0).unsqueeze(0)

                        inputs = torch.cat(torch.split(inputs, split_size_or_sections=config.para.block_size, dim=3), dim=0)
                        inputs = torch.cat(torch.split(inputs, split_size_or_sections=config.para.block_size, dim=2), dim=0)

                        reconstruction, _ = network(inputs)

                        idx = expand_w // config.para.block_size
                        reconstruction = torch.cat(torch.split(reconstruction, split_size_or_sections=1 * idx, dim=0), dim=2)
                        reconstruction = torch.cat(torch.split(reconstruction, split_size_or_sections=1, dim=0), dim=3)
                        reconstruction = reconstruction.squeeze()[:h, :w]

                        x_hat = reconstruction.cpu().numpy() * 255.
                        x_hat = np.rint(np.clip(x_hat, 0, 255))

                        psnr = PSNR(x_ori, x_hat, data_range=255)
                        ssim = SSIM(x_ori, x_hat, data_range=255, multichannel=False)

                        sum_psnr += psnr
                        sum_ssim += ssim

                        if save_img:
                            cv.imwrite(f"{recon_dataset_path_grey_rate}/{name_image}.png", x_hat)
                        save_log(recon_root, one_dataset, name_image, psnr, ssim, f"_{config.para.rate}_{manner}")
                    save_log(recon_root, one_dataset, None,
                             sum_psnr / len(images), sum_ssim / len(images), f"_{config.para.rate}_{manner}_AVG", False)
                    print(
                        f"AVG RES: PSNR, {round(sum_psnr / len(images), 2)}, SSIM, {round(sum_ssim / len(images), 4)}")
                    if val:
                        return round(sum_psnr / len(images), 2), round(sum_ssim / len(images), 4)

            # RGB manner
            elif manner == "rgb":
                recon_dataset_path_rgb = f"{recon_root}/{one_dataset}/rgb/"
                recon_dataset_path_rgb_rate = f"{recon_root}/{one_dataset}/rgb/{config.para.rate}"
                if not os.path.isdir(recon_dataset_path_rgb):
                    os.mkdir(recon_dataset_path_rgb)
                if not os.path.isdir(recon_dataset_path_rgb_rate):
                    os.mkdir(recon_dataset_path_rgb_rate)
                sum_psnr, sum_ssim = 0., 0.
                for _, _, images in os.walk(f"{test_dataset_path}/rgb/"):
                    for one_image in images:
                        name_image = one_image.split('.')[0]
                        x = cv.imread(f"{test_dataset_path}/rgb/{one_image}")
                        x_ori = x
                        r, g, b = cv.split(x)
                        r = torch.from_numpy(np.asarray(r)).squeeze().float() / 255.
                        g = torch.from_numpy(np.asarray(g)).squeeze().float() / 255.
                        b = torch.from_numpy(np.asarray(b)).squeeze().float() / 255.

                        x = torch.from_numpy(x).float()
                        h, w = x.size()[0], x.size()[1]

                        lack = config.para.block_size - h % config.para.block_size if h % config.para.block_size != 0 else 0
                        padding_h = torch.zeros(lack, w)
                        expand_h = h + lack
                        inputs_r = torch.cat((r, padding_h), 0)
                        inputs_g = torch.cat((g, padding_h), 0)
                        inputs_b = torch.cat((b, padding_h), 0)

                        lack = config.para.block_size - w % config.para.block_size if w % config.para.block_size != 0 else 0
                        expand_w = w + lack
                        padding_w = torch.zeros(expand_h, lack)
                        inputs_r = torch.cat((inputs_r, padding_w), 1).unsqueeze(0).unsqueeze(0)
                        inputs_g = torch.cat((inputs_g, padding_w), 1).unsqueeze(0).unsqueeze(0)
                        inputs_b = torch.cat((inputs_b, padding_w), 1).unsqueeze(0).unsqueeze(0)

                        inputs_r = torch.cat(torch.split(inputs_r, split_size_or_sections=config.para.block_size, dim=3),
                                             dim=0)
                        inputs_r = torch.cat(torch.split(inputs_r, split_size_or_sections=config.para.block_size, dim=2),
                                             dim=0)

                        inputs_g = torch.cat(torch.split(inputs_g, split_size_or_sections=config.para.block_size, dim=3),
                                             dim=0)
                        inputs_g = torch.cat(torch.split(inputs_g, split_size_or_sections=config.para.block_size, dim=2),
                                             dim=0)

                        inputs_b = torch.cat(torch.split(inputs_b, split_size_or_sections=config.para.block_size, dim=3),
                                             dim=0)
                        inputs_b = torch.cat(torch.split(inputs_b, split_size_or_sections=config.para.block_size, dim=2),
                                             dim=0)

                        r_hat, _ = network(inputs_r.to(config.para.device))
                        g_hat, _ = network(inputs_g.to(config.para.device))
                        b_hat, _ = network(inputs_b.to(config.para.device))

                        idx = expand_w // config.para.block_size
                        r_hat = torch.cat(torch.split(r_hat, split_size_or_sections=1 * idx, dim=0), dim=2)
                        r_hat = torch.cat(torch.split(r_hat, split_size_or_sections=1, dim=0), dim=3)
                        r_hat = r_hat.squeeze()[:h, :w].cpu().numpy() * 255.

                        g_hat = torch.cat(torch.split(g_hat, split_size_or_sections=1 * idx, dim=0), dim=2)
                        g_hat = torch.cat(torch.split(g_hat, split_size_or_sections=1, dim=0), dim=3)
                        g_hat = g_hat.squeeze()[:h, :w].cpu().numpy() * 255.

                        b_hat = torch.cat(torch.split(b_hat, split_size_or_sections=1 * idx, dim=0), dim=2)
                        b_hat = torch.cat(torch.split(b_hat, split_size_or_sections=1, dim=0), dim=3)
                        b_hat = b_hat.squeeze()[:h, :w].cpu().numpy() * 255.

                        r_hat, g_hat, b_hat = np.rint(np.clip(r_hat, 0, 255)), \
                                              np.rint(np.clip(g_hat, 0, 255)), \
                                              np.rint(np.clip(b_hat, 0, 255))
                        reconstruction = cv.merge([r_hat, g_hat, b_hat])

                        psnr = PSNR(x_ori, reconstruction, data_range=255)
                        ssim = SSIM(x_ori, reconstruction, data_range=255, multichannel=True)

                        sum_psnr += psnr
                        sum_ssim += ssim

                        if save_img:
                            cv.imwrite(f"{recon_dataset_path_rgb_rate}/{name_image}.png",
                                       (reconstruction))
                        save_log(recon_root, one_dataset, name_image, psnr, ssim, f"_{config.para.rate}_{manner}")
                    save_log(recon_root, one_dataset, None,
                             sum_psnr / len(images), sum_ssim / len(images), f"_{config.para.rate}_{manner}_AVG", False)
                    print(
                        f"AVG RES: PSNR, {round(sum_psnr / len(images), 2)}, SSIM, {round(sum_ssim / len(images), 4)}")
                    if val:
                        return round(sum_psnr / len(images), 2), round(sum_ssim / len(images), 4)
            else:
                raise NotImplemented(f"Error manner: {manner}.")


if __name__ == "__main__":
    my_state_dict = config.para.my_state_dict
    device = config.para.device

    net = models.TCS_Net().eval().to(device)
    if os.path.exists(my_state_dict):
        if torch.cuda.is_available():
            trained_model = torch.load(my_state_dict, map_location=device)
        else:
            raise Exception(f"No GPU.")
        net.load_state_dict(trained_model)
    else:
        raise FileNotFoundError(f"Missing trained model of rate {config.para.rate}.")
    testing(net, val=False, manner=config.para.manner, save_img=config.para.save)
