from tqdm import tqdm
import torch

from evaluation import calc_accuracy, get_confusion_matrix_image, get_mean_squared_error
from evaluation import get_sample_dict, update_hardsample_indice, draw_cam
import torchvision.utils as vutils
from utils import tensor_rgb2bgr



def trainer(
    max_epoch, 
    model, 
    train_loader, 
    test_loader, 
    loss_recon,
    loss_ce,
    optimizer,
    scheduler,
    meta, 
    writer = None,
    visualizer = None
):

    save_every = meta['save_every']
    print_every = meta['print_every']
    test_every = meta['test_every']


    for ep in range(1, max_epoch+1):
        train(ep, max_epoch, model, train_loader, loss_recon, loss_ce, optimizer, writer, print_every, visualizer=visualizer)
        if scheduler is not None:
            scheduler.step()


        if ep % test_every == 0:
            loss = test(ep, max_epoch, model, test_loader, writer, loss_recon = loss_recon, loss_ce=loss_ce, visualizer=visualizer)
            if loss_ce is None:
                loss *= -1

            
            writer.update(model, loss)
        
        if ep == 1 or ep % save_every == 0:
            writer.save(model, ep)
            
    writer.close()
    

def tester(
    model,
    test_loader,
    writer,
    visualizer,
    confusion_matrix,

):
    pbar=tqdm(total=len(test_loader))
    acc = test(
        None,None,
        model, test_loader, writer,
        visualizer = visualizer,
        pbar = pbar,
        confusion_matrix = confusion_matrix,
    )
    
    writer.close()




def train(ep, max_epoch, model, train_loader, loss_recon, loss_ce, optimizer, writer, _print_every, visualizer = None):
    model.train()

    epoch_loss = 0.0
    mean_loss = 0.0

    print_every = len(train_loader) // _print_every     
    if print_every == 0:
        print_every = 1

    preds = []
    gt = []

    step = 0
    step_cnt = 1

    global_step = (ep - 1) * len(train_loader)
    local_step = 0

    vis_dict = {
        'latent_code' : [],
        'label' : []
    }

    for i, batch in enumerate(train_loader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        output_dict = model(x)

        loss = 0.0

        if loss_recon is not None:
            loss += loss_recon(output_dict['x_hat'], x)
        if loss_ce is not None:
            loss += loss_ce(output_dict['y_hat'], y)
            preds.append(output_dict['y_hat'])
            gt.append(y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        epoch_loss += loss.item()
        step += 1
        global_step += 1
        local_step += 1

        vis_dict['latent_code'].append(output_dict['latent_code'].detach().cpu())
        vis_dict['label'].append(y.cpu())


        if (i+1) % print_every == 0:
            mean_loss /= step
            writer.add_scalar('train/loss', mean_loss, global_step)
            print('Epoch [{}/{}] Step[{}/{}] Loss: {:.4f}'.format(
                ep, max_epoch, step_cnt, _print_every, mean_loss))
            
            mean_loss = 0.0
            step = 0
            step_cnt += 1

            # add_img
            x_hat = output_dict['x_hat']
            recon_image = vutils.make_grid(x_hat.detach().cpu().clamp(0.0,1.0), normalize=True, scale_each=True)
            gt_image    = vutils.make_grid(x.detach().cpu().clamp(0.0,1.0), normalize=True, scale_each=True)

            if writer is not None:
                writer.add_image('train/recon_img', recon_image, global_step)
                writer.add_image('train/gt_img', gt_image, global_step)

    print ('Train Summary[{},{}] : Loss: {:.4f}'.format(ep, max_epoch, epoch_loss/local_step))


    if loss_ce is not None:
        preds = torch.cat(preds)
        gt = torch.cat(gt)

        acc = torch.mean((preds.argmax(dim=1) == gt).float())
        print ('Train Summary[{},{}] : Acc: {:.4f}'.format(ep, max_epoch, acc))
        writer.add_scalar('train/acc', acc, ep)


@torch.no_grad() # stop calculating gradient
def test(ep, max_epoch, model, test_loader, writer, loss_recon=None, loss_ce = None, visualizer = None,  pbar=None, confusion_matrix = False):
    model.eval()

    epoch_loss = 0.0
    local_step = 0

    if ep is not None:

        global_step = (ep - 1) * len(test_loader)

    else:
        global_step = 0
        ep = 1


    preds = []
    gt = []

    latent_dict = {'latent_code': [], 'label': [] }

    
    for idx, batch in enumerate(test_loader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        output_dict = model(x)

        if loss_recon is not None:
            loss = loss_recon(output_dict['x_hat'], x)
            epoch_loss += loss.item()

        if loss_ce is not None:
            preds.append(output_dict['y_hat'])
            gt.append(y)


        local_step += 1
        
        if idx % 10 ==0:
            x_hat = output_dict['x_hat']
            recon_image = vutils.make_grid(x_hat.detach().cpu().clamp(0.0,1.0), normalize=True, scale_each=True)
            gt_image    = vutils.make_grid(x.cpu().clamp(0.0,1.0), normalize=True, scale_each=True)

            if writer is not None:
                writer.add_image('test/recon_img', recon_image, global_step)
                writer.add_image('test/gt_img', gt_image, global_step)


        if pbar is not None:
            pbar.update()

    epoch_loss /=local_step
    print ('Test Summary[{}/{}] : Loss {:.4f}'.format(ep, max_epoch, epoch_loss))
    writer.add_scalar('test/loss', epoch_loss, (ep-1) * len(test_loader))

    if visualizer is not None:
        umap_img = visualizer(latent_dict)
        writer.add_image('test/umap', umap_img, ep)

    if loss_ce is not None or confusion_matrix:
        preds = torch.cat(preds)
        gt = torch.cat(gt)

        acc = torch.mean((preds.argmax(dim=1) == gt).float())
        print ('Test Summary[{},{}] : Acc: {:.4f}'.format(ep, max_epoch, acc))
        writer.add_scalar('test/acc', acc, ep)

        cm_image = get_confusion_matrix_image(preds.detach().cpu(), gt.cpu(), normalize=False)
        writer.add_image('test/unnorm_cm', cm_image, ep)

        cm_image = get_confusion_matrix_image(preds.detach().cpu(), gt.cpu(), normalize=True)
        writer.add_image('test/norm_cm', cm_image, ep)

        return acc  
       
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