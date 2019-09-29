'''
@Time       : 
@Author     : Jingsen Zheng
@File       : trainer
@Brief      : 
'''
import shutil
import tqdm

import numpy as np
import torch

def checkpoint_state(model=None,
                     optimizer=None,
                     best_prec=None,
                     epoch=None,
                     it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        'epoch': epoch,
        'it': it,
        'best_prec': best_prec,
        'model_state': model_state,
        'optimizer_state': optim_state
    }

def save_checkpoint(state,
                    is_best,
                    filename='checkpoint',
                    bestname='model_best'):
    filename = '{}.pth.tar'.format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.pth.tar'.format(bestname))

class Trainer(object):
    r"""
        Reasonably generic trainer for pytorch models

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    eval_frequency : int
        How often to run an eval
    log_name : str
        Name of file to output tensorboard_logger to
    """

    def __init__(self,
                 model,
                 model_fn,
                 optimizer,
                 checkpoint_name="ckpt",
                 best_name="best",
                 lr_scheduler=None,
                 bnm_scheduler=None,
                 eval_frequency=-1,
                 viz=None):
        self.model, self.model_fn, self.optimizer, self.lr_scheduler, self.bnm_scheduler = (
            model, model_fn, optimizer, lr_scheduler, bnm_scheduler)

        self.checkpoint_name, self.best_name = checkpoint_name, best_name
        self.eval_frequency = eval_frequency

        self.training_best, self.eval_best = {}, {}
        self.viz = viz

    @staticmethod
    def _decode_value(v):
        if isinstance(v[0], float):
            return np.mean(v)
        elif isinstance(v[0], tuple):
            if len(v[0]) == 3:
                num = [l[0] for l in v]
                denom = [l[1] for l in v]
                w = v[0][2]
            else:
                num = [l[0] for l in v]
                denom = [l[1] for l in v]
                w = None

            return np.average(
                np.sum(num, axis=0) / (np.sum(denom, axis=0) + 1e-6), weights=w)
        else:
            raise AssertionError("Unknown type: {}".format(type(v)))

    def _train_it(self, it, batch):
        self.model.train()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(it)

        if self.bnm_scheduler is not None:
            self.bnm_scheduler.step(it)

        self.optimizer.zero_grad()
        _, loss, eval_res = self.model_fn(self.model, batch)

        loss.backward()
        self.optimizer.step()

        return eval_res

    def eval_epoch(self, d_loader):
        self.model.eval()

        eval_dict = {}
        total_loss = 0.0
        count = 1.0
        for i, data in tqdm.tqdm(
                enumerate(d_loader, 0),
                total=len(d_loader),
                leave=False,
                desc='val'):
            self.optimizer.zero_grad()

            _, loss, eval_res = self.model_fn(self.model, data, eval=True)

            total_loss += loss.item()
            count += 1
            for k, v in eval_res.items():
                if v is not None:
                    eval_dict[k] = eval_dict.get(k, []) + [v]

        return total_loss / count, eval_dict

    def train(self,
              start_it,
              start_epoch,
              n_epochs,
              train_loader,
              test_loader=None,
              best_loss=0.0):
        r"""
           Call to begin training the model

        Parameters
        ----------
        start_epoch : int
            Epoch to start at
        n_epochs : int
            Number of epochs to train for
        test_loader : torch.utils.data.DataLoader
            DataLoader of the test_data
        train_loader : torch.utils.data.DataLoader
            DataLoader of training data
        best_loss : float
            Testing loss of the best model
        """

        eval_frequency = (self.eval_frequency
                          if self.eval_frequency > 0 else train_loader.iter_num_per_epoch())

        it = start_it
        with tqdm.trange(start_epoch, n_epochs + 1, desc='epochs') as tbar, \
                tqdm.tqdm(total=eval_frequency, leave=False, desc='train') as pbar:

            for epoch in tbar:
                for batch in train_loader:
                    res = self._train_it(it, batch)
                    it += 1

                    pbar.update()
                    pbar.set_postfix(dict(total_it=it))
                    tbar.refresh()

                    if self.viz is not None:
                        self.viz.update('train', it, res)

                    if (it % eval_frequency) == 0:
                        pbar.close()

                        if test_loader is not None:
                            val_loss, res = self.eval_epoch(test_loader)

                            if self.viz is not None:
                                self.viz.update('val', it, res)

                            is_best = val_loss < best_loss
                            best_loss = min(best_loss, val_loss)
                            save_checkpoint(
                                checkpoint_state(self.model, self.optimizer,
                                                 val_loss, epoch, it),
                                is_best,
                                filename=self.checkpoint_name,
                                bestname=self.best_name)

                        pbar = tqdm.tqdm(
                            total=eval_frequency, leave=False, desc='train')
                        pbar.set_postfix(dict(total_it=it))

                    self.viz.flush()

        return best_loss
