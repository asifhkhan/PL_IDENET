import torch
from utils import normalized_tensor
from utils.utils_common import calculate_psnr, calculate_ssim
from imageio import imwrite
import wandb
import numpy as np

from torchvision.utils import make_grid

def crop_forward(model, x, stdn, sf, shave=10, min_size=100000, bic=None):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 1
    scale = sf
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if bic is not None:
        bic_h_size = h_size*scale
        bic_w_size = w_size*scale
        bic_h = h*scale
        bic_w = w*scale

        bic_list = [
            bic[:, :, 0:bic_h_size, 0:bic_w_size],
            bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
            bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
            bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if bic is not None:
                bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)

            sr_batch_temp = model(lr_batch, stdn)

            if isinstance(sr_batch_temp, list):
                sr_batch = sr_batch_temp[-1]
            else:
                sr_batch = sr_batch_temp

            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            crop_forward(model, x=patch, stdn=stdn, shave=shave, min_size=min_size) \
            for patch in lr_list
            ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


@torch.no_grad()
def evaluate(model, testset_loader, lpips_model, epoch, opt, logger, save_test_imgs_path,border,lpips):
    logger.info('===================== start testing =====================')

    model.eval()
    test_results = {'psnr':[], 'ssim':[], 'lpips_dist':[]}
    img_idx = 1
    for i, data in enumerate(testset_loader):
        y, x, sigma = data['LR'], data['HR'], data['sigma']

        window_size = 8
        _, _, h_old, w_old = y.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([y, torch.flip(y, [2])], 2)[:, :, :h_old + h_pad, :]
        y = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

        if opt.cuda:
            y = y.cuda()
            x = x.cuda()
            sigma = sigma.cuda()
            # gt_kr = kernel.cuda()

        y = y.float()
        x = x.float()
        sigma = sigma.float()
        # gt_kr = gt_kr.float()


        # print("test x:", x.shape, x.min(), x.max())
        # print("test y:", y.shape, y.min(), y.max())
        # print("test sigma:", sigma.shape, sigma.min(), sigma.max())
        if opt.use_chop:
            xhat = crop_forward(model, y, opt.upscale_factor)
        if opt.whole:
            outputs, est_kernel = model(y)
            outputs = outputs[..., :h_old * 4, :w_old * 4]
            # xhat = outputs.clamp(0., 255.)
        # print("test xhat:", xhat.shape, xhat.min(), xhat.max())
        if opt.tile is  None:

            # test the image tile by tile
            b, c, h, w = y.size()
            tile = min(opt.tile, h, w)
            assert tile % opt.window_size == 0, "tile size should be a multiple of window_size"
            tile_overlap = opt.tile_overlap
            sf = opt.upscale_factor
            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h * sf, w * sf).type_as(y)
            W = torch.zeros_like(E)
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = y[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch[0])
                    E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch[0])
                    W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)

            outputs = E.div_(W)
        xhat = outputs.clamp(0., 1.)


        # save test images
        to_np_uint8 = lambda img: (img*255.0).cpu().numpy().astype(np.uint8)
        gt = to_np_uint8(x.permute(0, 2, 3, 1))
        LR = to_np_uint8(y.permute(0, 2, 3, 1))
        output = to_np_uint8(xhat.permute(0, 2, 3, 1).detach())
        # est_kernel = to_np_uint8(est_kernel.permute(0, 2, 3, 1).detach())
        # GT_kernel = to_np_uint8(gt_kr.permute(0, 2, 3, 1).detach())


        if img_idx ==3:

            wandb.log({"Test Images": [wandb.Image(gt, caption='GT'),
                                 wandb.Image(LR, caption='LR'), wandb.Image(output, caption='SR')],

                       "cols": 3})

        gt_img_path = save_test_imgs_path + 'img' + repr(img_idx) + '_GT' + '.png'
        LR_img_path = save_test_imgs_path + 'img' + repr(img_idx) + '_LR' + '.png'
        output_img_path = save_test_imgs_path + 'img' + repr(img_idx) + '_SR' + '.png'

        # psnr, ssim, and lpips

        psnr = calculate_psnr(output[0], gt[0], border=border)
        ssim = calculate_ssim(output[0], gt[0], border=border)

        # normalized tensors
        img_x = normalized_tensor(x*255.0)
        pred_xhat = normalized_tensor(xhat*255.0)
        lpips_dist = lpips.forward(img_x, pred_xhat).item()

        # psnr = 0
        # ssim = 0
        # lpips_dist = 0

        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        test_results['lpips_dist'].append(lpips_dist)

        #logger.info('{:->4d}--> {:>10s}, psnr:{:.2f}dB'.format(img_idx, output_img_path, psnr))
        #logger.info('{:->4d}--> {:>10s}, ssim:{:.4f}'.format(img_idx, output_img_path, ssim))
        #logger.info('{:->4d}--> {:>10s}, lpips dist:{:.4f}'.format(img_idx, output_img_path, lpips_dist))
        img_idx += 1

        if epoch % 1 == 0:
            imwrite(gt_img_path, gt[0])
            imwrite(LR_img_path, LR[0])
            imwrite(output_img_path, output[0])

    del y
    del x
    del sigma
    del xhat
    del gt
    del LR
    del output
    torch.cuda.empty_cache()

    avg_psnr = np.mean(test_results['psnr'])
    avg_ssim = np.mean(test_results['ssim'])
    avg_lpips = np.mean(test_results['lpips_dist'])

    wandb.log({'test_psnr':avg_psnr, 'test_ssim':avg_ssim, 'test_lpips': avg_lpips})


    # print("===>test:: Avg. PSNR:{:.2f}".format(avg_psnr))
    # print("===>test:: Avg. SSIM:{:.4f}".format(avg_ssim))
    #logger.info("test:: Epoch[{}]: Avg. PSNR: {:.2f} dB".format(epoch, avg_psnr))
    #logger.info("test:: Epoch[{}]: Avg. SSIM: {:.4f}".format(epoch, avg_ssim))
    #logger.info("test:: Epoch[{}]: Avg. Lpips: {:.6f}".format(epoch, avg_lpips))

    logger.info('===================== end testing =====================')

    return avg_psnr, avg_ssim, avg_lpips
