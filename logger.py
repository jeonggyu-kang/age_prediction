from tensorboardX import SummaryWriter
import os
import torch
import copy 
import csv
import glob

def get_logger(save_root, use_cam=False):
    writer = SaveManager(save_root, use_cam=use_cam)
    return writer

class SaveManager:
    def __init__(self, save_root, use_cam):
        self.save_root = save_root
        os.makedirs(self.save_root, exist_ok = True)

        self.writer = SummaryWriter(save_root)

        
        self.best_acc = None
        self.best_model = None

        self.result_summary_path = os.path.join(self.save_root, 'result_summary.txt')
        # gradcam 
        if use_cam:
            self.cam_root = os.path.join(self.save_root, 'gradcam')
            os.makedirs(self.cam_root, exist_ok = True)
        else: 
            self.cam_root = None
        self.cam_idx = 1

        self.csv_path = os.path.join(self.save_root, 'inference_result.csv')
        
        self.csv_fp = open(self.csv_path, 'w')
        self.csv_writer = csv.writer(self.csv_fp)
        self.csv_writer.writerow(['file_name', 'gt', 'pred'])

    def get_cam_dst_path(self):
        cam_path = os.path.join(self.cam_root, '{}.png'.format(self.cam_idx))
        self.cam_idx += 1
        return cam_path

    def export_csv(self, file_name, gt, pred):
        pred = '{:2f}'.format(pred)
        pred = float(pred)
        self.csv_writer.writerow([file_name, gt, pred])

    def add_scalar(self, text:str, value: float, global_step: int):
        self.writer.add_scalar(text, value, global_step)

    def add_image(self, text, image_grid, global_step):
        self.writer.add_image(text, image_grid, global_step)

    def update(self, model, acc):
        if self.best_acc is None:
            self.best_model = copy.deepcopy(model)
            self.best_acc = acc
            return

        if self.best_acc > acc:
            print('best_model update: {}'.format(acc))
            self.best_model = copy.deepcopy(model)
            self.best_acc = acc

    def save(self, model, prefix):
        checkpoint = {}
        checkpoint['weight'] = model.state_dict()

        save_path = os.path.join(self.save_root, str(prefix)+'.pt')
        torch.save(checkpoint, save_path)

    def close(self):
        self.csv_fp.close()
        self.writer.close()

        if self.best_model is not None:
            self.save(self.best_model, 'best')

        if os.path.isfile(self.result_summary_path):
            f_mode = 'a'
        else:
            f_mode = 'wt'

        if self.best_acc is not None:
            with open(self.result_summary_path, f_mode) as f:
                f.write('Best Scoire: '+str(self.best_acc)+'\n')
                print ('Best: Score: {}'.format(self.best_acc))

        # remove pth files

        target_files = glob.glob(os.path.join(self.save_root, '*.pt'))

        for f in target_files:
            if (f.split('/')[-1]).split('.')[0] == 'best':
                continue

            os.remove(f)