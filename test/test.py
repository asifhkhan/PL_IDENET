import argparse
import logging
import os.path
import sys
import time
import modules
from collections import OrderedDict
from tqdm import tqdm
import lpips

import numpy as np
import torch
from IPython import embed

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr
import wandb
import time
import psutil
wandb.login()
#### options
parser = argparse.ArgumentParser()

# parser.add_argument("-opt", type=str, default='/home/asif/Documents/kernel_est/DAN/DAN_master/codes/config/DANv1/options/setting1/test/test_setting1_x4.yml', help="Path to options YMAL file.")
parser.add_argument("-opt", type=str, default='./options/setting1/test/test_setting1_x2.yml', help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)
loss_fn_alex = lpips.LPIPS(net='alex')
#
# run = wandb.init(
#         # Set the project where this run will be logged
#         project="wiener_filter_SR", id='w1coadjv')
#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))
def measure_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / (1024**3)  # Convert bytes to gigabytes
    return memory_usage_gb
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
#
for test_loader in tqdm(test_loaders):
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results["psnr"] = []
    test_results["ssim"] = []
    test_results["lpips"] = []
    test_results["psnr_y"] = []
    test_results["ssim_y"] = []
    time_per_image = []


    for test_data in test_loader:
        single_img_psnr = []
        single_img_ssim = []
        # single_img_lpips = []
        single_img_psnr_y = []
        single_img_ssim_y = []

        need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
        img_path = test_data["LQ_path"][0]
        img_path_ker = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
        img_name = img_path
        img_name_ker = img_path
        # img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        model.feed_data(test_data["LQ"], test_data["GT"])
        # start_event.record()
        model.test(test_data["LQ"])
        # end_event.record()
        # torch.cuda.synchronize()
        # execution_time = start_event.elapsed_time(end_event) / 1000
        # time_per_image.append(execution_time)

        # print('execution_time per image', execution_time)
        # memory_usage1 = measure_memory_usage()
        visuals = model.get_current_visuals()
        output_SR = visuals["SR"]
        # ker_img = visuals["ker"]
        # rescaled_ker_tensor = (ker_img - ker_img.min()) / (ker_img.max() - ker_img.min())

        if opt['ensemble']:
            img_LR0 = util.augment_img_tensor(test_data['LQ'], mode=0)
            img_LR1 = util.augment_img_tensor(test_data['LQ'], mode=1)
            img_LR2 = util.augment_img_tensor(test_data['LQ'], mode=2)
            img_LR3 = util.augment_img_tensor(test_data['LQ'], mode=3)
            img_LR4 = util.augment_img_tensor(test_data['LQ'], mode=4)
            img_LR5 = util.augment_img_tensor(test_data['LQ'], mode=5)
            img_LR6 = util.augment_img_tensor(test_data['LQ'], mode=6)
            img_LR7 = util.augment_img_tensor(test_data['LQ'], mode=7)

            model.feed_data(img_LR0, test_data["GT"])
            model.test(img_LR0)
            visuals0 = model.get_current_visuals()
            output_SR0 = visuals0["SR"]

            model.feed_data(img_LR1, test_data["GT"])
            model.test(img_LR1)
            visuals1 = model.get_current_visuals()
            output_SR1 = visuals1["SR"]

            model.feed_data(img_LR2, test_data["GT"])
            model.test(img_LR2)
            visuals2 = model.get_current_visuals()
            output_SR2 = visuals2["SR"]

            model.feed_data(img_LR3, test_data["GT"])
            model.test(img_LR3)
            visuals3 = model.get_current_visuals()
            output_SR3 = visuals3["SR"]

            model.feed_data(img_LR4, test_data["GT"])
            model.test(img_LR4)
            visuals4 = model.get_current_visuals()
            output_SR4 = visuals4["SR"]

            model.feed_data(img_LR5, test_data["GT"])
            model.test(img_LR5)
            visuals5 = model.get_current_visuals()
            output_SR5 = visuals5["SR"]

            model.feed_data(img_LR6, test_data["GT"])
            model.test(img_LR6)
            visuals6 = model.get_current_visuals()
            output_SR6 = visuals6["SR"]

            model.feed_data(img_LR7, test_data["GT"])
            model.test(img_LR7)
            visuals7 = model.get_current_visuals()
            output_SR7 = visuals7["SR"]

            output_SR0_ = util.inv_augment_img_tensor(output_SR0, mode=0)
            output_SR1_ = util.inv_augment_img_tensor(output_SR1, mode=1)
            output_SR2_ = util.inv_augment_img_tensor(output_SR2, mode=2)
            output_SR3_ = util.inv_augment_img_tensor(output_SR3, mode=3)
            output_SR4_ = util.inv_augment_img_tensor(output_SR4, mode=4)
            output_SR5_ = util.inv_augment_img_tensor(output_SR5, mode=5)
            output_SR6_ = util.inv_augment_img_tensor(output_SR6, mode=6)
            output_SR7_ = util.inv_augment_img_tensor(output_SR7, mode=7)

            # output_SR = (output_SR0) + (output_SR1_.clamp_(0,255)) +(output_SR2_.clamp_(0,255)) + (output_SR3_.clamp_(0,255)) +(output_SR4_.clamp_(0,255)) +(output_SR5_.clamp_(0,255)) +(output_SR6_.clamp_(0,255)) +(output_SR7_.clamp_(0,255))/8.0

            output_SR = torch.stack((output_SR,output_SR0_, output_SR1_, output_SR2_, output_SR3_,
                           output_SR4_, output_SR5_, output_SR6_,output_SR7_))
            output_SR = torch.mean(output_SR, dim=0)

            # model.feed_data((test_data["LQ"].flip(-1)).flip(-1), test_data["GT"])
            # model.test(test_data["LQ"])
            #
            # visuals = model.get_current_visuals()
            # sr_img_E1 = visuals["SR"]
            #
            # model.feed_data((test_data["LQ"].flip(-2)).flip(-2), test_data["GT"])
            # model.test(test_data["LQ"])
            #
            # visuals = model.get_current_visuals()
            # sr_img_E2 = visuals["SR"]
            #
            # model.feed_data((test_data["LQ"].flip(-1,-2)).flip(-1,-2), test_data["GT"])
            # model.test(test_data["LQ"])
            #
            # visuals = model.get_current_visuals()
            # sr_img_E3 = visuals["SR"]
            #
            # L_t = test_data["LQ"].transpose(-2, -1)
            #
            # model.feed_data((L_t).transpose(-2,-1), test_data["GT"])
            # visuals = model.get_current_visuals()
            # sr_img_E4 = visuals["SR"]
            #
            # model.feed_data(((L_t).flip(-1)).flip(-1).transpose(-2,-1), test_data["GT"])
            # visuals = model.get_current_visuals()
            # sr_img_E5 = visuals["SR"]
            #
            # model.feed_data(((L_t).flip(-2)).flip(-2).transpose(-2, -1), test_data["GT"])
            # visuals = model.get_current_visuals()
            # sr_img_E6 = visuals["SR"]
            #
            # model.feed_data(((L_t).flip(-1,-2)).flip(-1,-2).transpose(-2, -1), test_data["GT"])
            # visuals = model.get_current_visuals()
            # sr_img_E7 = visuals["SR"]
            #
            # sr_img_E = (sr_img_E.clamp_(0,255)) + (sr_img_E1.clamp_(0,255)) +(sr_img_E2.clamp_(0,255)) + (sr_img_E3.clamp_(0,255)) +(sr_img_E4.clamp_(0,255)) +(sr_img_E5.clamp_(0,255)) +(sr_img_E6.clamp_(0,255)) +(sr_img_E7.clamp_(0,255))/8.0

        sr_img = output_SR.clamp_(0,1)
        sr_img_wandb = output_SR.clamp_(0, 1)
        SR_img = visuals["SR"]
        GT_img = visuals["GT"]
        # ker_img = rescaled_ker_tensor.clamp_(0, 1)




        sr_img = util.tensor2img(visuals["SR"]*255.0)  # uint8
        # ker_img = util.tensor2img(rescaled_ker_tensor*255.0)

        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")
            save_img_path_ker = os.path.join(dataset_dir, img_name_ker + ".png")
        util.save_img(sr_img, save_img_path)
        # util.save_img(ker_img, save_img_path_ker)




        # if img_name == 'img_001' or img_name == 'img_003' or  img_name == 'img_004' or  img_name == 'img_013' or  img_name == 'img_028' :
        #
        #     wandb.log({"Test Set [{:s}]...".format(test_set_name): [wandb.Image(sr_img_wandb*255.0, caption="SR_img [{:s}]".format(img_name)),
        #                                ],
        #
        #                "cols": 5})



        if need_GT:
            gt_img = util.tensor2img(test_data["GT"]*255.0)
            gt_img = gt_img / 255.0
            sr_img = sr_img / 255.0

            crop_border = opt["crop_border"] if opt["crop_border"] else opt["scale"]
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[
                    crop_border:-crop_border, crop_border:-crop_border, :
                ]
                cropped_gt_img = gt_img[
                    crop_border:-crop_border, crop_border:-crop_border, :
                ]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
            lpips = modules.PerceptualLoss(model='net-lin', net='alex')  # alex, squeeze, vgg
            lpips_dist = lpips.forward(GT_img, SR_img).item()

            test_results["psnr"].append(psnr)
            test_results["ssim"].append(ssim)
            test_results["lpips"].append(lpips_dist)

            if gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                if crop_border == 0:
                    cropped_sr_img_y = sr_img_y
                    cropped_gt_img_y = gt_img_y
                else:
                    cropped_sr_img_y = sr_img_y[
                        crop_border:-crop_border, crop_border:-crop_border
                    ]
                    cropped_gt_img_y = gt_img_y[
                        crop_border:-crop_border, crop_border:-crop_border
                    ]
                psnr_y = util.calculate_psnr(
                    cropped_sr_img_y * 255, cropped_gt_img_y * 255
                )
                ssim_y = util.calculate_ssim(
                    cropped_sr_img_y * 255, cropped_gt_img_y * 255
                )

                test_results["psnr_y"].append(psnr_y)
                test_results["ssim_y"].append(ssim_y)

                logger.info(
                    "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f};  PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(img_name, psnr, ssim,  psnr_y, ssim_y))
            else:
                logger.info(
                    "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f};.".format(
                        img_name, psnr, ssim
                    )
                )

                test_results["psnr_y"].append(psnr)
                test_results["ssim_y"].append(ssim)
        else:
            logger.info(img_name)

    ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
    ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])

    # wandb.log({f"{test_set_name} Avg_PSNR": ave_psnr, f"{test_set_name} Avg_SSIM": ave_ssim})
    # wandb.log({"Average PSNR/SSIM/ results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f};".format(test_set_name, ave_psnr, ave_ssim)})
    ave_lpips = sum(test_results["lpips"]) / len(test_results["lpips"])
    logger.info(
        "----Average PSNR/SSIM/LPIPS results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f};".format(
            test_set_name, ave_psnr, ave_ssim,ave_lpips
        )
    )
    if test_results["psnr_y"] and test_results["ssim_y"]:
        ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
        ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
        # wandb.log({f"{test_set_name} Avg_PSNR_y": ave_psnr_y, f"{test_set_name} Avg_SSIM_y": ave_ssim_y})
        logger.info(
            "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
                ave_psnr_y, ave_ssim_y
            )
        )
    Average_time_per_image = sum(time_per_image)/len(time_per_image)
    # print(f"Execution time per image: {Average_time_per_image} seconds")
    # print("Memory Usage Model 1:", memory_usage1, "GB")
