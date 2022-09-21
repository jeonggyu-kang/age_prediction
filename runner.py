from tqdm import tqdm
import torch

from evaluation import calc_accuracy, get_confusion_matrix_image, get_mean_squared_error
from evaluation import get_sample_dict, update_hardsample_indice, draw_cam
from evaluation import write_age_hard_sample
import torchvision.utils as vutils
from utils import tensor_rgb2bgr



def trainer(
    max_epoch, 
    model, 
    train_loader, 
    test_loader, 
    loss_mse,
    loss_ce,
    optimizer,
    scheduler,
    meta, 
    writer = None,
):

    save_every = meta['save_every']
    print_every = meta['print_every']
    test_every = meta['test_every']


    for ep in range(1, max_epoch+1):
        train(ep, max_epoch, model, train_loader, loss_mse, loss_ce, optimizer, writer, print_every)
        if scheduler is not None:
            scheduler.step()

        
        if ep % test_every == 0:
            error = test(ep, max_epoch, model, test_loader, writer, loss_mse)
            
            writer.update(model, error)
       
        
        if ep == 1 or ep % save_every == 0:
            writer.save(model, ep)
            
    writer.close()
    

def tester(
    model,
    test_loader,
    writer,
    visualizer,
    confusion_matrix,
    csv = False,
    hard_sample = False,
    age_hard_sample = False,
    age_ratio_thres = 10,
    age_diff_thres = 0.2    
):

    pbar=tqdm(total=len(test_loader))
    print('Dataset length: {}'.format(len(test_loader)))
    acc = test(
        None,None,
        model, test_loader, writer,
        confusion_matrix = confusion_matrix,
        csv = csv,
        hard_sample = hard_sample,
        age_hard_sample = age_hard_sample,
        age_ratio_thres = age_ratio_thres,
        age_diff_thres = age_diff_thres   
    )
    
    writer.close()




def train(ep, max_epoch, model, train_loader, loss_mse, loss_ce, optimizer, writer, _print_every):
    model.train()

    epoch_error = 0.0
    total_loss = 0.0
    mse_loss = 0.0
    ce_loss = 0.0

    print_every = len(train_loader) // _print_every     
    if print_every == 0:
        print_every = 1

    score_dict = {
        'pred_age' : [],
        'gt_age' : [],
        'pred_sex' : [],
        'gt_sex' : [],
    }

    step = 0
    step_cnt = 1

    global_step = (ep - 1) * len(train_loader)
    local_step = 0


    for i, batch in enumerate(train_loader):
        image = batch['image'].cuda()
        gt_age = batch['gt_age'].cuda()
        gt_sex = batch['gt_sex'].cuda()

        output_dict = model(image)

        score_dict['pred_sex'].append(output_dict['sex_hat'].cpu())
        score_dict['gt_sex'].append(gt_sex.cpu()) 

        loss_mse_value = loss_mse(output_dict['age_hat'], gt_age)
        loss_ce_value  = loss_ce(output_dict['sex_hat'], gt_sex)

        loss = loss_mse_value + 0.9 * loss_ce_value

        
        
        # preds.append(output_dict['y_hat'])
        # gt.append(y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        mse_loss += loss_mse_value.item()
        ce_loss += loss_ce_value.item()
        epoch_error += loss_mse_value.item()


        step += 1
        global_step += 1
        local_step += 1


        if (i+1) % print_every == 0:
            total_loss /= step
            mse_loss /= step
            ce_loss /= step

            writer.add_scalar('train/total_loss', total_loss, global_step)
            writer.add_scalar('train/mse_loss', mse_loss, global_step)
            writer.add_scalar('train/ce_loss', ce_loss, global_step)

            print('Epoch [{}/{}] Step[{}/{}] Total-Loss: {:.4f} MSE-Loss: {:.4f} CE-Loss: {:.4f}'.format(
                ep, max_epoch, step_cnt, _print_every, total_loss, mse_loss, ce_loss))
            
            total_loss = 0.0
            mse_loss = 0.0
            ce_loss = 0.0
            step = 0
            step_cnt += 1


    print ('Train Summary[{},{}] : MSE-Error: {:.4f}'.format(ep, max_epoch, epoch_error/local_step))
    writer.add_scalar('train/age-mse-error', epoch_error/local_step, ep)

    preds = torch.cat(score_dict['pred_sex'])
    gt = torch.cat(score_dict['gt_sex'])

    acc = torch.mean((preds.argmax(dim=1) == gt).float())
    print ('Train Summary[{},{}] : Sex-Acc: {:.4f}'.format(ep, max_epoch, acc))
    writer.add_scalar('train/sex-acc', acc, ep)


@torch.no_grad() # stop calculating gradient
def test(ep, max_epoch, model, test_loader, writer, loss_mse=None, confusion_matrix=False, csv = False, hard_sample=False, age_hard_sample=False, age_diff_thres = None, age_ratio_thres = None):
    model.eval()

    epoch_loss = 0.0
    local_step = 0

    if ep is not None:

        global_step = (ep - 1) * len(test_loader)

    else:
        global_step = 0
        ep = 1


    score_dict = {
        'pred_sex' : [],
        'gt_sex' : [],
    }

    hardsample_dict = get_sample_dict(2)
    age_hardsample_list = []


    for i, batch in enumerate(test_loader):
        image = batch['image'].cuda()
        gt_age = batch['gt_age'].cuda()
        gt_sex = batch['gt_sex'].cuda()

        output_dict = model(image)

        score_dict['pred_sex'].append(output_dict['sex_hat'].cpu())
        score_dict['gt_sex'].append(gt_sex.cpu())

        if loss_mse is not None:
            loss_mse_value = loss_mse(output_dict['age_hat'], gt_age)
            epoch_loss += loss_mse_value.item()
            local_step +=1
        else:
            B, _, _, _ = image.shape
            for bi in range(B):
                age_gt  = batch['gt_age_int'][bi].item()
                age_hat = int((output_dict['age_hat'][bi] * 99 + 1. + 0.5).item())
                diff = abs(age_gt - age_hat)
                # print('pred: {},  gt: {}'.format(age_hat, age_gt)) # age
                ratio = diff / age_gt
                if diff > age_diff_thres or ratio > age_ratio_thres:
                    sample = batch['image'][bi].unsqueeze(0)
                    
                    signed_diff = age_gt - age_hat
                    signed_diff_ratio = signed_diff / age_gt
                    age_hardsample_list.append({
                        'signed_diff' : signed_diff,
                        'signed_diff_ratio' : signed_diff_ratio,
                        'image' : batch['image'][bi]
                        
                    })


                epoch_loss += diff 
                local_step +=1



        if hard_sample:
            hardsample_dict = update_hardsample_indice(
                output_dict['sex_hat'].detach().cpu(), 
                batch['gt_sex'], 
                hardsample_dict, 
                batch['image']
            )



        if csv:
            B, _, _, _ = image.shape
            for bi in range(B):
                f_name = batch['f_name'][bi]
                age_gt = int(batch['gt_age_int'][bi].item()) # gt 
                age_hat = (output_dict['age_hat'][bi] * 99 + 1.).item()
                writer.export_csv(f_name, age_gt, age_hat)
        

            

    # mse loss value (return)
    epoch_loss /= local_step
    print ('Test Summary[{}/{}] : MSE-Loss: {:.4f}'.format(ep, max_epoch, epoch_loss))
    writer.add_scalar('test/age-loss', epoch_loss, ep)

    # acc
    preds = torch.cat(score_dict['pred_sex'])
    gt = torch.cat(score_dict['gt_sex'])

    acc = torch.mean((preds.argmax(dim=1) == gt).float())
    print ('Test Summary[{}/{}] : Sex-Acc: {:.4f}'.format(ep, max_epoch, acc))
    writer.add_scalar('test/age-acc', acc, ep)


    if age_hard_sample:
        # write top & bottom N hard sample w.r.t signed age diff.
        age_hardsample_list.sort(key = lambda x:x['signed_diff'])
        write_age_hard_sample(age_hardsample_list, writer, 'diff')

        # write top & bottom N hard sample w.r.t signed age diff ratio.
        age_hardsample_list.sort(key = lambda x:x['signed_diff_ratio'])
        write_age_hard_sample(age_hardsample_list, writer, 'ratio')        

    if hard_sample:
        index2cls_name = {
            0: 'Female',
            1: 'Male',
        }
        for gt_k in hardsample_dict:
            for pred_k, samples in hardsample_dict[gt_k].items():
                if samples:
                    num_sample = len(samples)
                    text = str(index2cls_name[gt_k]) + '/' + str(index2cls_name[pred_k])
                    samples = torch.cat(samples)
                    grid_samples = vutils.make_grid(tensor_rgb2bgr(samples), normalize=True, scale_each=True)
                    writer.add_image(text, grid_samples, 0)
                    print('{} : {}'.format(text, num_sample))

    if confusion_matrix:
        cm_image = get_confusion_matrix_image(preds.detach().cpu(), gt.cpu(), normalize=False)
        writer.add_image('test/unnorm_cm', cm_image, ep)

        cm_image = get_confusion_matrix_image(preds.detach().cpu(), gt.cpu(), normalize=True)
        writer.add_image('test/norm_cm', cm_image, ep)

    return epoch_loss


def grad_cam(model, data_loader, writer, cam, export_csv, n_class, task_type):
    model.eval()
    pbar = tqdm(total=len(data_loader))

    print ('Dataset length: {}'.format(len(data_loader)))

    for idx, batch in enumerate(data_loader):
        x = batch['x']
        y = batch['y']
        f_name = batch['f_name']

        x = x.cuda()
        
        if task_type == 'classification':
            draw_cam(cam, x, y, n_class, writer)

        else:
            raise NotImplementedError

        if export_csv: # csv
            pred = model(x)
            writer.export_csv(f_name, y.cpu().item(), pred.argmax(1).cpu().item())

        pbar.update()

    writer.close()