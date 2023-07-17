import sys
import torch
from tqdm import tqdm as tqdm
from torch.cuda.amp import autocast,GradScaler
from .meter import AverageValueMeter
from .dice_score import *
import torch.nn as nn
import torch.nn.functional as F
from . import functional as FUC
from sklearn.metrics import *
class Epoch:
    def __init__(self, model1,model2,loss, metrics, stage_name, device="cpu", verbose=True):
        self.model1 = model1
        self.model2 = model2

        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self._to_device()
    def _to_device(self):
        self.model1.to(self.device)
        self.model2.to(self.device)


        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x1,x2, y, z):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter_list = [AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter(),AverageValueMeter()]
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        metrics_meters_1 = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        train_preds_1 = []
        train_trues = []
        train_preds_2 = []
        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x1,x2, y, z in iterator:
                x1,x2, y, z   = x1.to(self.device),x2.to(self.device),y.to(self.device), z.to(self.device)
         
                loss_dict,pred_list = self.batch_update(x1, x2, y, z)
                # classification

                z_pred_1 = pred_list[2].argmax(dim=1)
                z_pred_2 = pred_list[3].argmax(dim=1)
                train_preds_1.extend(z_pred_1.detach().cpu().numpy())
                train_preds_2.extend(z_pred_2.detach().cpu().numpy())
                train_trues.extend(z.detach().cpu().numpy())
                
                
                # update loss logs
                for idx,value in enumerate(loss_dict.items()):
                    loss_value = value[1].cpu().detach().numpy()
                    loss_meter_list[idx].add(loss_value)
                    loss_logs = {value[0]: loss_meter_list[idx].mean}
                    logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(pred_list[0], y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs1 = {k + "_1": v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs1)
                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(pred_list[1], y).cpu().detach().numpy()
                    metrics_meters_1[metric_fn.__name__].add(metric_value)
                metrics_logs2 = {k + "_2": v.mean for k, v in metrics_meters_1.items()}
                logs.update(metrics_logs2)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
        sklearn_accuracy_1 = accuracy_score(train_trues, train_preds_1)

        sklearn_recall_1 = recall_score(train_trues, train_preds_1, average='weighted')
        sklearn_precision_1 = precision_score(train_trues, train_preds_1, average='weighted')
        sklearn_f1_1 = f1_score(train_trues, train_preds_1, average='weighted')
        sklearn_accuracy_2 = accuracy_score(train_trues, train_preds_2)
        sklearn_recall_2 = recall_score(train_trues, train_preds_2, average='weighted')
        sklearn_precision_2 = precision_score(train_trues, train_preds_2, average='weighted')
        sklearn_f1_2 = f1_score(train_trues, train_preds_2, average='weighted')
        return logs,[sklearn_accuracy_1,sklearn_recall_1,sklearn_precision_1,sklearn_f1_1,
                     sklearn_accuracy_2,sklearn_recall_2,sklearn_precision_2,sklearn_f1_2]


class TrainEpoch(Epoch):
    def __init__(self, model1,model2,loss, metrics, optimizer, device="cpu", verbose=True, weight_list = None):
        super().__init__(
            model1=model1,
            model2=model2,

            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose
            
        )
        self.optimizer = optimizer
        self.scaler = GradScaler()
        self.cre = nn.CrossEntropyLoss()
        self.weight_list = weight_list
    def on_epoch_start(self):
        self.model1.train()
        self.model2.train()

    def graddient_norm(self,x):
        return torch.sqrt(torch.sum(torch.square(x)))
    def batch_update(self, x1,x2, y, z ):
        loss = 0
        self.optimizer.zero_grad()
        # with autocast():
        pred_mask_1, pred_label_1 = self.model1.forward(x1)
        pred_mask_2, pred_label_2 = self.model2.forward(x2)


        self.loss[0] = self.loss[0].to(self.device)
        self.loss[1] = self.loss[1].to(self.device)
        self.loss[2] = self.loss[2].to(self.device)


        loss_mask_1 = self.loss[0](pred_mask_1,y)
        loss_mask_2 = self.loss[0](pred_mask_2, y)

        loss_label_1 = self.loss[1](pred_label_1, z)
        loss_label_2 = self.loss[1](pred_label_2, z)
        loss_kl_mask = self.loss[2](F.softmax(pred_mask_1,dim=1), F.softmax(pred_mask_2,dim=1))
        loss_kl_label = self.loss[2](F.softmax(pred_mask_2,dim=1), F.softmax(pred_mask_1,dim=1))




        loss =       loss_mask_1 +  loss_mask_2 +    loss_label_1  +  loss_label_2 +     0.1*(loss_kl_label +  loss_kl_mask)

        loss.backward(torch.ones_like(loss))
        self.optimizer.step()

        return {'loss_kl_mask': loss_kl_mask, 'loss_kl_label':  loss_kl_label, 'loss_mask_1':  loss_mask_1, 'loss_mask_2':loss_mask_2,
                'loss_label_1':loss_label_1, 'loss_label_2':loss_label_2}, [pred_mask_1, pred_mask_2, F.softmax(pred_label_1,dim= -1 ), F.softmax(pred_label_2,dim= -1 )]


class ValidEpoch(Epoch):
    def __init__(self, model1, model2, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model1=model1,
            model2=model2,

            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,

        )

    def on_epoch_start(self):
        self.model1.eval()
        self.model2.eval()

    def batch_update(self, x1,x2, y, z):
        loss = 0 
        with torch.no_grad():
            pred_mask_1, pred_label_1 = self.model1.forward(x1)
            pred_mask_2, pred_label_2 = self.model2.forward(x2)

            self.loss[0] = self.loss[0].to(self.device)
            self.loss[1] = self.loss[1].to(self.device)
            self.loss[2] = self.loss[2].to(self.device)


            loss_mask_1 = self.loss[0](pred_mask_1, y)
            loss_mask_2 = self.loss[0](pred_mask_2, y)

            loss_label_1 = self.loss[1](pred_label_1, z)
            loss_label_2 = self.loss[1](pred_label_2, z)

            loss_kl_mask = self.loss[2](F.softmax(pred_mask_1, dim=1), F.softmax(pred_mask_2, dim=1))
            loss_kl_label = self.loss[2](F.softmax(pred_mask_2, dim=1), F.softmax(pred_mask_1, dim=1))
            loss = 10 * loss_mask_1 + 10 * loss_mask_2 + loss_label_1 + loss_label_2


        return {'loss_kl_mask': loss_kl_mask, 'loss_kl_label':  loss_label_1, 'loss_mask_1':  loss_mask_1, 'loss_mask_2':loss_mask_2,
                'loss_label_1':loss_label_1, 'loss_label_2':loss_label_2}, [pred_mask_1, pred_mask_2, F.softmax(pred_label_1,dim= -1 ), F.softmax(pred_label_2,dim= -1 )]