import copy
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm

import matplotlib.pyplot as plt

from finetune_evaluator import Evaluator


class Trainer(object):
    def __init__(self, params, data_loader, model, device='cuda'):
        self.params = params
        self.data_loader = data_loader
        self.device = device
        self.best_val_metric = -float('inf')  # start very low
        self.best_model_states = None  

        #self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a']: 
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset in ['SEED-VIG', 'CHONGQING']:
            self.criterion = MSELoss().cuda()

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None

        # Lists to store metrics per epoch for plotting
        epoch_losses = []
        epoch_accs = []

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            correct = 0
            total = 0

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)
                loss.backward()
                losses.append(loss.data.cpu().numpy())

                # Calculate batch accuracy
                preds = pred.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            # Epoch metrics
            epoch_loss = np.mean(losses)
            epoch_acc = correct / total
            epoch_eval_accs = []
            epoch_pr_aucs = []
            epoch_roc_aucs = []
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)

                print(
                    "Epoch {} : Training Loss: {:.5f}, Train Acc: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        epoch_loss,
                        epoch_acc,
                        acc,
                        pr_auc,
                        roc_auc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                if roc_auc > roc_auc_best:
                    print("roc_auc increasing....saving weights !! ")
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        # Load best model
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model saved in " + model_path)

        # Plot training loss and accuracy per epoch
        epochs = range(1, self.params.epochs + 1)
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.plot(epochs, epoch_losses, 'b-', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(epochs, epoch_accs, 'r-', label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


    # def train_for_binaryclass(self):
    #     acc_best = 0
    #     roc_auc_best = 0
    #     pr_auc_best = 0
    #     cm_best = None
    #     for epoch in range(self.params.epochs):
    #         self.model.train()
    #         start_time = timer()
    #         losses = []
    #         for x, y in tqdm(self.data_loader['train'], mininterval=10):
    #             self.optimizer.zero_grad()
    #             x = x.cuda()
    #             y = y.cuda()
    #             pred = self.model(x)

    #             loss = self.criterion(pred, y)

    #             loss.backward()
    #             losses.append(loss.data.cpu().numpy())
    #             if self.params.clip_value > 0:
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
    #                 # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
    #             self.optimizer.step()
    #             self.optimizer_scheduler.step()

    #         optim_state = self.optimizer.state_dict()

    #         with torch.no_grad():
    #             acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
    #             print(
    #                 "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
    #                     epoch + 1,
    #                     np.mean(losses),
    #                     acc,
    #                     pr_auc,
    #                     roc_auc,
    #                     optim_state['param_groups'][0]['lr'],
    #                     (timer() - start_time) / 60
    #                 )
    #             )
    #             print(cm)
    #             if roc_auc > roc_auc_best:
    #                 print("roc_auc increasing....saving weights !! ")
    #                 print("Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
    #                     acc,
    #                     pr_auc,
    #                     roc_auc,
    #                 ))
    #                 best_f1_epoch = epoch + 1
    #                 acc_best = acc
    #                 pr_auc_best = pr_auc
    #                 roc_auc_best = roc_auc
    #                 cm_best = cm
    #                 self.best_model_states = copy.deepcopy(self.model.state_dict())
    #     self.model.load_state_dict(self.best_model_states)
    #     with torch.no_grad():
    #         print("***************************Test************************")
    #         acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
    #         print("***************************Test results************************")
    #         print(
    #             "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
    #                 acc,
    #                 pr_auc,
    #                 roc_auc,
    #             )
    #         )
    #         print(cm)
    #         if not os.path.isdir(self.params.model_dir):
    #             os.makedirs(self.params.model_dir)
    #         model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, pr_auc, roc_auc)
    #         torch.save(self.model.state_dict(), model_path)
    #         print("model save in " + model_path)

    # def train_for_binaryclass(self, epochs=3):
    #     for epoch in range(epochs):
    #         self.model.train()
    #         running_loss = 0.0
    #         for x, y in tqdm(self.data_loader['train'], desc=f"Epoch {epoch+1} Training"): # for test load 'test'
    #             x = x.to(self.device)
    #             y = y.to(self.device)

    #             self.optimizer.zero_grad()
    #             pred = self.model(x)  # [batch, 2]
    #             loss = self.criterion(pred, y.long())
    #             loss.backward()
    #             self.optimizer.step()

    #             running_loss += loss.item() * x.size(0)

    #         epoch_loss = running_loss / len(self.data_loader['train'].dataset)
            
    #         # Evaluate on validation set
    #         acc, pr_auc, roc_auc, cm =  self.val_eval.get_metrics_for_binaryclass(self.model)
            
    #         # Save best model based on balanced accuracy
    #         if acc > self.best_val_metric:
    #             self.best_val_metric = acc
    #             self.best_model_states = copy.deepcopy(self.model.state_dict())
            
    #         print(f"Epoch {epoch+1} : Loss: {epoch_loss:.5f}, acc: {acc:.5f}, pr_auc: {pr_auc:.5f}, roc_auc: {roc_auc}, Best acc: {self.best_val_metric:.5f}")
    #         # x axis - epoch, y axis - loss curve/accuracy curve
    #         # save values in a file

    #     # Load the best model
    #     self.model.load_state_dict(self.best_model_states)

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        loss_best = float('inf')
        #best_epoch = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda().float()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                train_loss = np.mean(losses)
                print(
                    "Epoch {} : Training Loss: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        train_loss,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )

                if train_loss < loss_best:
                    print("Training loss decreasing....saving weights !!")
                    loss_best = train_loss
                    best_f1_epoch = epoch + 1
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )


            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_f1_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)



# def train_for_regression(self): 
#     corrcoef_best = 0 
#     r2_best = 0 
#     rmse_best = 0 
#     for epoch in range(self.params.epochs): 
#         self.model.train() 
#         start_time = timer() 
#         losses = [] 
#         for x, y in tqdm(self.data_loader['train'], mininterval=10): self.optimizer.zero_grad() 
#         x = x.cuda() 
#         y = y.cuda() 
#         pred = self.model(x) 
        
#         loss = self.criterion(pred, y) 
#         loss.backward() losses.append(loss.data.cpu().numpy()) 
#         if self.params.clip_value > 0: 
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value) 
#             #torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value) 
#             self.optimizer.step() 
#             self.optimizer_scheduler.step() 
            
#         optim_state = self.optimizer.state_dict() 
        
#         with torch.no_grad(): 
#             corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model) 
#             print( "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format( 
#                 epoch + 1, 
#                 np.mean(losses), 
#                 corrcoef, 
#                 r2, 
#                 rmse, 
#                 optim_state['param_groups'][0]['lr'], 
#                 (timer() - start_time) / 60 
#                 ) 
#             ) 
#             if r2 > r2_best: 
#                 print("r2 increasing....saving weights !! ") 
#                 print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format( corrcoef, r2, rmse, )) 
#                 best_r2_epoch = epoch + 1 
#                 corrcoef_best = corrcoef 
#                 r2_best = r2 
#                 rmse_best = rmse 
#                 self.best_model_states = copy.deepcopy(self.model.state_dict()) 
            
#         self.model.load_state_dict(self.best_model_states) 
#         with torch.no_grad(): 
#             print("***************************Test************************") 
#             corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model) 
#             print("***************************Test results************************") 
#             print( "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format( 
#                 corrcoef, 
#                 r2, 
#                 rmse, 
#                 ) 
#             ) 
#             if not os.path.isdir(self.params.model_dir): 
#                 os.makedirs(self.params.model_dir) 
#             model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse) 
#             torch.save(self.model.state_dict(), model_path) 
#             print("model save in " + model_path)