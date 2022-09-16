from tqdm import tqdm
import torch

from evaluation import calc_accuracy, get_confusion_matrix_image, get_mean_squared_error
from evaluation import get_sample_dict, update_hardsample_indice, draw_cam
import torchvision.utils as vutils
from utils import tensor_rgb2bgr



def trainer_ae(
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
    

def tester_ae(
    model,
    test_loader,
    writer,
    visualizer,
    confusion_matrix,
    csv = False

):
    pbar=tqdm(total=len(test_loader))
    print('Dataset length: {}'.format(len(test_loader)))
    acc = test(
        None,None,
        model, test_loader, writer,
        confusion_matrix = confusion_matrix,
        csv = csv
    )
    
    writer.close()




def train(ep, max_epoch, model, train_loader, loss_mse, loss_ce, optimizer, writer, _print_every):
    model.train()

    epoch_error = 0.0

    total_loss = 0.0
    age_loss = 0.0
    sex_loss = 0.0
    recon_loss = 0.0

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

        score_dict['pred_sex'].append(output_dict['y2_hat'].cpu())
        score_dict['gt_sex'].append(gt_sex.cpu()) 
        
        loss_recon   = loss_mse(output_dict['x_hat'], image)
        loss_age     = loss_mse(output_dict['y1_hat'], gt_age)
        loss_sex     = loss_ce(output_dict['y2_hat'], gt_sex)

        loss = loss_recon + loss_age + loss_sex    # change loss weigth

        
        
        # preds.append(output_dict['y_hat'])
        # gt.append(y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        age_loss += loss_age.item()
        sex_loss += loss_sex.item()
        recon_loss += loss_recon.item()

        epoch_error += loss_age.item()


        step += 1
        global_step += 1
        local_step += 1


        if (i+1) % print_every == 0:
            total_loss /= step
            age_loss /= step
            sex_loss /= step
            recon_loss /= step

            writer.add_scalar('train/total_loss', total_loss, global_step)
            writer.add_scalar('train/age_loss', age_loss, global_step)
            writer.add_scalar('train/sex_loss', sex_loss, global_step)
            writer.add_scalar('train/recon_loss', recon_loss, global_step)

            print('Epoch [{}/{}] Step[{}/{}] Total-Loss: {:.4f} Age-Loss: {:.4f} Sex-Loss: {:.4f} Recon-Loss: {:.4f}'.format(
                ep, max_epoch, step_cnt, _print_every, total_loss, age_loss, sex_loss, recon_loss))
            
            total_loss = 0.0
            age_loss = 0.0
            sex_loss = 0.0
            recon_loss = 0.0
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
def test(ep, max_epoch, model, test_loader, writer, loss_mse=None, confusion_matrix=False, csv = False):
    model.eval()

    epoch_loss = 0.0
    epoch_recon_loss = 0.0
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
     
    loss_mse2 = torch.nn.MSELoss() 
    for i, batch in enumerate(test_loader):
        image = batch['image'].cuda()
        gt_age = batch['gt_age'].cuda()
        gt_sex = batch['gt_sex'].cuda()

        output_dict = model(image)

        epoch_recon_loss += loss_mse2(output_dict['x_hat'], image)

        score_dict['pred_sex'].append(output_dict['y2_hat'].cpu())
        score_dict['gt_sex'].append(gt_sex.cpu())

        if loss_mse is not None:
            loss_mse_value = loss_mse(output_dict['y1_hat'], gt_age)
            epoch_loss += loss_mse_value.item()
            local_step +=1
        else:
            B, _, _, _ = image.shape
            for bi in range(B):
                age_gt  = batch['gt_age_int'][bi].item()
                age_hat = int((output_dict['y1_hat'][bi] * 99 + 1. + 0.5).item())
                diff = abs(age_gt - age_hat)
                # print('pred: {},  gt: {}'.format(age_hat, age_gt)) # age
                epoch_loss += diff 
                local_step +=1

        if csv:
            B, _, _, _ = image.shape
            for bi in range(B):
                f_name = batch['f_name'][bi]
                age_gt = int(batch['gt_age_int'][bi].item()) # gt 
                age_hat = int((output_dict['y1_hat'][bi] * 99 + 1. + 0.5).item())
                writer.export_csv(f_name, age_gt, age_hat)
            

    # mse loss value (return)
    epoch_loss /= local_step
    print ('Test Summary[{}/{}] : MSE-Loss: {:.4f}'.format(ep, max_epoch, epoch_loss))
    writer.add_scalar('test/age-loss', epoch_loss, ep)

    print('Test Summary[{}/{}] : Recon-loss: {:.4f}'.format(
        ep, max_epoch, epoch_recon_loss
    ))
    writer.add_scalar('test/recon-loss', epoch_recon_loss, ep)

    # acc
    preds = torch.cat(score_dict['pred_sex'])
    gt = torch.cat(score_dict['gt_sex'])

    acc = torch.mean((preds.argmax(dim=1) == gt).float())
    print ('Test Summary[{}/{}] : Sex-Acc: {:.4f}'.format(ep, max_epoch, acc))
    writer.add_scalar('test/age-acc', acc, ep)

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