import os
import shutil
import torch
from collections import OrderedDict
import glob
from datetime import datetime

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.runname, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        # run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        num=len(os.listdir(self.directory))
        self.experiment_dir = os.path.join(self.directory, '{}_experiment_{}'.format(str(num+1),str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='model_best.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['tr_batch_size']=self.args.tr_batch_size
        p['val_batch_size']=self.args.val_batch_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()