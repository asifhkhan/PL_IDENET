from data_loader.LRHR_dataset import LRHRDataset
from torch.utils.data import DataLoader


def build(opt, logger):
    # print('===> Loading dataset')
    logger.info('===================== Loading dataset =====================')
    # training dataset

    train_dataset = LRHRDataset(dataroot=opt.imdbTrainPath,
                                is_train=opt.is_train,
                                scale=opt.upscale_factor,
                                patch_size=opt.patch_size,
                                rgb_range=opt.rgb_range,
                                noise_std=opt.train_stdn)

    trainset_loader = DataLoader(train_dataset,
                                 shuffle=True,
                                 batch_size=opt.trainBatchSize,
                                 pin_memory=True,
                                 num_workers=opt.numWorkers,
                                 drop_last=True
                                 )

#    train_size = int(math.ceil(len(train_dataset) / opt.trainBatchSize))
    logger.info('training dataset:{:6d}'.format(len(train_dataset)))
    logger.info('training loaders:{:6d}'.format(len(trainset_loader)))


    # testing dataset
    test_dataset = LRHRDataset(dataroot=opt.imdbTestPath,
                               is_train=False,
                               scale=opt.upscale_factor,
                               patch_size=opt.patch_size,
                               rgb_range=opt.rgb_range,
                               noise_std=opt.test_stdn)

    testset_loader = DataLoader(test_dataset,
                                shuffle=False,
                                batch_size=opt.testBatchSize,
                                pin_memory=True,
                                num_workers=1
                                )

    logger.info('testing dataset:{:6d}'.format(len(test_dataset)))
    logger.info('testing loaders:{:6d}'.format(len(testset_loader)))

    return train_dataset, trainset_loader, test_dataset, testset_loader