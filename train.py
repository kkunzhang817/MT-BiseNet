
import torch.nn as nn

from utils.distributed import *
from utils.log_manager import setup_logger
from model.model_zoo import smp,get_model
import segmentation_models_pytorch.utils.metrics as metrics
from load_data import *
from torch.utils.data import DataLoader
from seg_augmentations import *
import warnings
import gc
import argparse
from utils.torch_utils import *
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')

    # env system set
    parser.add_argument('--system', type=str, default='linux',
                        choices=['windows', 'linux'],
                        help='distributed only can be used by linux')
    # model and dataset
    parser.add_argument('--decoder-1', type=str, default='mt_bisenet',
                        help='decoder name (default: fpn)')
    parser.add_argument('--decoder-2', type=str, default='mt_bisenet',
                        help='decoder name (default: fpn)')
    parser.add_argument('--encoder-1', type=str, default='resnet18',
                        help='backbone name (default: resnet50), others can see encoder storehouse')
    parser.add_argument('--encoder-2', type=str, default='resnet18',
                        help='backbone name (default: resnet50), others can see encoder storehouse')
    parser.add_argument('--encoder-weights', type=str, default='imagenet',
                        help='backbone weights ')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys', 'sbu', 'leaf'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--image-size',  default=224,
                        help='input model image size')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--num-txt',  type=list, default=[1,2])
    # training hyper params
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--val-batch-size', type=int, default=16, metavar='N',
                        help='input val batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='choose optimizer, defeault : adam')
    parser.add_argument('--activation', type=str, default='sigmoid',
                        help='choose optimizer, defeault : sigmoid')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device', default='cuda:0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='./weights',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-dir', default='./runs/logs/',
                        help='Directory for saving checkpoint models')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self,args,classes,logger):
        self.args = args
        self.device = torch.device(self.args.device)
        self.classes = classes
        self.max_iou_score = 0
        self.max_acc_score = 0
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.args.encoder_1, self.args.encoder_weights)
        self.logger = logger
        self.weight = [1] * 2
        self.his_acc = []
        self.his_iou = []
        # todo 可以写数据集库导入
        self.train_dataset = Dataset_interface("JPEGImages", "masks", self.classes, model="train",
                            augmentation=get_training_augmentation(args.image_size,1),augmentation_color=get_training_augmentation(args.image_size,2),preprocessing=get_preprocessing(self.preprocessing_fn),image_size= args.image_size)
        self.val_dataset =  Dataset_interface("JPEGImages", "masks", self.classes, model="val",
                            preprocessing=get_preprocessing(self.preprocessing_fn),image_size= args.image_size)
        self.test_dataset =  Dataset_interface("JPEGImages", "masks", self.classes, model="test",
                            preprocessing=get_preprocessing(self.preprocessing_fn),image_size= args.image_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                       pin_memory=True, num_workers= self.args.workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.args.val_batch_size,
                                      num_workers=self.args.workers, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset,batch_size=self.args.val_batch_size,num_workers= self.args.workers,shuffle=True)

        args.iters_per_epoch = len(self.train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        self.model1 = get_model(self.args.decoder_1,encoder_name= self.args.encoder_1, encoder_weights= self.args.encoder_weights,
                               classes= len(self.classes),activation= self.args.activation,aux_params = {'classes':30, 'activation':'softmax','only_class':False})
        self.model2 = get_model(self.args.decoder_2,encoder_name= self.args.encoder_2, encoder_weights= self.args.encoder_weights,
                               classes= len(self.classes),activation= self.args.activation,aux_params = {'classes':30, 'activation':'softmax','only_class':False})

        # resume model
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'sorry pkl pth need'
                print('Resuming traing, loading {}...'.format(args.resume))
                self.model = torch.load(args.resume)

        # loss function
        self.loss_func=[  smp.utils.losses.DiceLoss(),
                          smp.utils.losses.CrossEntropyLoss(),
                           smp.utils.losses.KLLoss(),
        ]

        # todo 可以写优化器库导入
        self.optimizer = torch.optim.NAdam([dict(params=self.model1.encoder.parameters(), lr=self.args.lr ,weight_decay=0),
                                            dict(params=self.model1.decoder.parameters(), lr=self.args.lr ,weight_decay=0),
                                            dict(params=self.model1.segmentation_head.parameters(),lr=self.args.lr ,weight_decay=0),
                                            dict(params=self.model1.classification_head.parameters(),lr=self.args.lr , weight_decay=0),
                                            dict(params=self.model2.encoder.parameters(), lr=self.args.lr ,weight_decay=0),
                                            dict(params=self.model2.decoder.parameters(), lr=self.args.lr ,weight_decay=0),
                                            dict(params=self.model2.segmentation_head.parameters(),lr=self.args.lr ,weight_decay=0),
                                            dict(params=self.model2.classification_head.parameters(),lr=self.args.lr , weight_decay=0),
                                            ])


        # todo 可以写学习率调整策略导入
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=1,
                                    verbose=True,threshold=0.0001, threshold_mode='rel', cooldown=1, min_lr=1e-10,eps=1e-30)
        # todo 评估指标自定义库导入
        self.metric = [
                        metrics.IoU(threshold=0.5),
                        metrics.Accuracy(threshold=0.5),
                        metrics.Precision(threshold=0.5),
                        metrics.Fscore(threshold=0.5),
                        metrics.Recall(threshold=0.5),
                        ]

    def train(self):

        train_epoch = smp.utils.train.TrainEpoch(self.model1,self.model2, loss = self.loss_func, metrics=self.metric,optimizer=self.optimizer,
                                                 device=self.device,verbose=True,weight_list = self.weight)

        param_groups = self.optimizer.param_groups
        for i in range(0, self.args.epochs):
            print('\nEpoch: {}'.format(i))


            train_logs, class_metrics = train_epoch.run(self.train_loader)

            self.logger.info(
                'Epoch: {} Train: loss_kl_mask : {} , loss_kl_label: {}, loss_mask_1 : {}, loss_mask_2 : {}, loss_label_1 : {}, loss_label_2 : {}'.format(
                    str(i), str(round(train_logs['loss_kl_mask'], 6)), str(round(train_logs['loss_kl_label'], 6)),str(round(train_logs['loss_mask_1'], 6)),
                    str(round(train_logs['loss_mask_2'], 6)),str(round(train_logs['loss_label_1'], 6)),str(round(train_logs['loss_label_2'], 6))))

            self.logger.info('Epoch: {} Train: iou_1 : {} , acc_1: {}, precision_1 : {}, fscore_1 : {}, recall_1 : {}, iou_2 : {} , acc_2: {}, precision_2 : {}, fscore_2 : {}, recall_2 : {}'.format(str(i), str(round(
            train_logs['iou_score_1'], 4)), str(round(train_logs['accuracy_1'], 4)), str(round(train_logs['precision_1'], 4)), str(round(
                                                                                                  train_logs['fscore_1'],4)),str(round(train_logs['recall_1'], 4)),str(round(
            train_logs['iou_score_2'], 4)), str(round(train_logs['accuracy_2'], 4)), str(round(train_logs['precision_2'], 4)), str(round(
                                                                                                  train_logs['fscore_2'],4)),str(round(train_logs['recall_2'], 4))))
            self.logger.info('Epoch: {} Train: acc_1 : {}, recall_1 : {},precision_1 : {},f1_1 : {}, acc_2 : {}, recall_2 : {},precision_2 : {},f1_2 : {}'.format(str(i),
                                                                                             str(round(class_metrics[0],6)),
                                                                                             str(round(class_metrics[1],6)),
                                                                                             str(round(class_metrics[2],6)),
                                                                                             str(round(class_metrics[3],6)),
                                                                                             str(round(class_metrics[4],6)),
                                                                                             str(round(class_metrics[5],6)),
                                                                                             str(round(class_metrics[6],6)),
                                                                                             str(round(class_metrics[7],6))))

            self.lr_scheduler.step(( 9 / 10 * round(train_logs['loss_mask_1'], 6) + 1 / 10 * round(train_logs['loss_label_1'], 6)))
            self.lr_scheduler.step(( 9 / 10 * round(train_logs['loss_mask_2'], 6) + 1 / 10 * round(train_logs['loss_label_2'], 6)))
            print(round(self.optimizer.param_groups[0]['lr'], 6))


            self.valid(i)
    def valid(self,i):
        best_model_path = None
        if self.args.distributed:
            model1 = self.model1.module
            model2 = self.model2.module

        else:
            model1 = self.model1
            model2 = self.model2

        torch.cuda.empty_cache()
        valid_epoch = smp.utils.train.ValidEpoch(model1, model2,loss=self.loss_func, metrics=self.metric, device=self.device,
                                                 verbose=True)
        valid_logs, class_metrics = valid_epoch.run(self.val_loader)

        self.logger.info(
            'Epoch: {} VAL: loss_kl_mask : {} , loss_kl_label: {}, loss_mask_1 : {}, loss_mask_2 : {}, loss_label_1 : {}, loss_label_2 : {}'.format(
                str(i), str(round(valid_logs['loss_kl_mask'], 6)), str(round(valid_logs['loss_kl_label'], 6)),
                str(round(valid_logs['loss_mask_1'], 6)),
                str(round(valid_logs['loss_mask_2'], 6)), str(round(valid_logs['loss_label_1'], 6)),
                str(round(valid_logs['loss_label_2'], 6))))
        self.logger.info(
            'Epoch: {} VAL: iou_1 : {} , acc_1: {}, precision_1 : {}, fscore_1 : {}, recall_1 : {}, iou_2 : {} , acc_2: {}, precision_2 : {}, fscore_2 : {}, recall_2 : {}'.format(
                str(i), str(round(
                    valid_logs['iou_score_1'], 4)), str(round(valid_logs['accuracy_1'], 4)),
                str(round(valid_logs['precision_1'], 4)), str(round(
                    valid_logs['fscore_1'], 4)), str(round(valid_logs['recall_1'], 4)), str(round(
                    valid_logs['iou_score_2'], 4)), str(round(valid_logs['accuracy_2'], 4)),
                str(round(valid_logs['precision_2'], 4)), str(round(
                    valid_logs['fscore_2'], 4)), str(round(valid_logs['recall_2'], 4))))
        self.logger.info(
            'Epoch: {} VAL: acc_1 : {}, recall_1 : {},precision_1 : {},f1_1 : {}, acc_2 : {}, recall_2 : {},precision_2 : {},f1_2 : {}'.format(
                str(i),
                str(round(class_metrics[0], 6)),
                str(round(class_metrics[1], 6)),
                str(round(class_metrics[2], 6)),
                str(round(class_metrics[3], 6)),
                str(round(class_metrics[4], 6)),
                str(round(class_metrics[5], 6)),
                str(round(class_metrics[6], 6)),
                str(round(class_metrics[7], 6))))



        test_epoch = smp.utils.train.ValidEpoch(model1, model2,loss=self.loss_func, metrics=self.metric, device=self.device,
                                                 verbose=True)
        test_logs, class_metrics = test_epoch.run(self.test_loader)

        self.is_best = True
        save_path1 = save_checkpoint(model1, self.args, self.is_best,num_model = 1)
        save_path2 = save_checkpoint(model2, self.args, self.is_best, num_model= 2)

        self.logger.info(
            'Epoch: {} TEST: loss_kl_mask : {} , loss_kl_label: {}, loss_mask_1 : {}, loss_mask_2 : {}, loss_label_1 : {}, loss_label_2 : {}'.format(
                str(i), str(round(test_logs['loss_kl_mask'], 6)), str(round(test_logs['loss_kl_label'], 6)),
                str(round(test_logs['loss_mask_1'], 6)),
                str(round(test_logs['loss_mask_2'], 6)), str(round(test_logs['loss_label_1'], 6)),
                str(round(test_logs['loss_label_2'], 6))))
        self.logger.info(
            'Epoch: {} TEST: iou_1 : {} , acc_1: {}, precision_1 : {}, fscore_1 : {}, recall_1 : {}, iou_2 : {} , acc_2: {}, precision_2 : {}, fscore_2 : {}, recall_2 : {}'.format(
                str(i), str(round(
                    test_logs['iou_score_1'], 4)), str(round(test_logs['accuracy_1'], 4)),
                str(round(test_logs['precision_1'], 4)), str(round(
                    test_logs['fscore_1'], 4)), str(round(test_logs['recall_1'], 4)), str(round(
                    test_logs['iou_score_2'], 4)), str(round(test_logs['accuracy_2'], 4)),
                str(round(test_logs['precision_2'], 4)), str(round(
                    test_logs['fscore_2'], 4)), str(round(test_logs['recall_2'], 4))))
        self.logger.info(
            'Epoch: {} TEST: acc_1 : {}, recall_1 : {},precision_1 : {},f1_1 : {}, acc_2 : {}, recall_2 : {},precision_2 : {},f1_2 : {}'.format(
                str(i),
                str(round(class_metrics[0], 6)),
                str(round(class_metrics[1], 6)),
                str(round(class_metrics[2], 6)),
                str(round(class_metrics[3], 6)),
                str(round(class_metrics[4], 6)),
                str(round(class_metrics[5], 6)),
                str(round(class_metrics[6], 6)),
                str(round(class_metrics[7], 6))))


        if (test_logs['iou_score_1'] + class_metrics[0]) < (test_logs['iou_score_2'] + class_metrics[4]):
            now_max_iou_score = test_logs['iou_score_2']
            now_max_acc_score = class_metrics[4]
            best_model = 2
        else:
            now_max_iou_score = test_logs['iou_score_1']
            now_max_acc_score = class_metrics[0]
            best_model = 1



        # data is full
        if self.max_iou_score + self.max_acc_score < now_max_acc_score + now_max_iou_score :
                self.max_iou_score = now_max_iou_score
                self.max_acc_score = now_max_acc_score
                self.is_best= True

                if best_model == 1:
                    self.logger.info("best model1!")
                    shutil.copy(save_path1,(os.path.join(os.path.dirname(save_path1) ,'{}_butterfly_best_baseline_dml_{}.pth'.format(str(args.decoder_1),str(args.image_size)))))

                else:
                    self.logger.info("best model2!")
                    shutil.copy(save_path2,(os.path.join(os.path.dirname(save_path2),'{}_butterfly_best_baseline_dml_{}.pth'.format(str(args.decoder_1),str(args.image_size)))))



        print("***************************")


if __name__ == '__main__':
    args = parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = None
    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_butterfly_baseline_dml_log.txt'.format(args.decoder_1, args.encoder_1,str(time.time())))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    CLASSES = ['background','butterfly']
    trainer = Trainer(args,CLASSES,logger)
    trainer.train()
    torch.cuda.empty_cache()




