import torch
from torch import nn, optim
from networks.facenet import Facenet
from gan_model import Generator
from tqdm import tqdm
from torchvision.transforms import Resize
from torchvision import utils as uts
import random
import argparse


def resize_img_gen_shape(img_gen, trans):
    t_img = trans(img_gen)
    face_input = t_img.clamp(min=-1, max=1).add(1).div(2)
    return face_input


def mutation(latent, init_strength, cur_step, iter, gen, target_model, t_resize, target_label):
    batch_size = latent.shape[0]
    t     = (cur_step+1)/(iter)
    alpha = max(min(0.6, 1-t), 0.0)
    beta  = max(min(0.2, 1-t), 0.0)
    gamma = max(min(0.0, 1-t), 0.0)
    # eps   = max(1-t, 0.5)
    eps   = 1.0
    t_sim = 2/3*alpha
    t_rad = alpha-t_sim

    tmp1   = torch.randn_like(latent).cuda()*init_strength*eps
    foo    = torch.zeros_like(latent).cuda()
    rand_w = gen.style(tmp1)

    prob = torch.rand(batch_size)

    imgs, _ = gen([latent], input_is_latent=True)
    face_input = resize_img_gen_shape(imgs, t_resize)

    # generate images
    before_norm, outputs1 = target_model.forward_feature(face_input)
    prediction = target_model.forward_classifier(before_norm)
    label      = torch.tensor(target_label).unsqueeze(0).long().cuda()
    # ------------------------------------------

    # calculate loss and ranking
    logitsloss = torch.zeros(batch_size)
    celoss     = torch.zeros(batch_size)
    max_pre,_  = prediction.max(1)
    CE         = nn.CrossEntropyLoss()
    for i in range(batch_size):
        logitsloss[i] = max_pre[i].clone().item()
        celoss[i]     = CE(prediction[i].unsqueeze(0), label).item()
    loss   = celoss
    _, idx = torch.sort(loss)

    # mutation
    for i in range(batch_size):
        if i < 1:
            mut_p = 0.0
        else:
            mut_p = float(i+1)/batch_size+1
        index = idx[i]
        if mut_p >= prob[i]:
            foo[index] = (1-t_rad)*(1-t_sim) * latent[index] + (1-t_rad)*t_sim * latent[idx[0]] + t_rad*rand_w[index]
        elif i != 0:
            foo[index] = latent[index] + beta*torch.randn_like(latent[index])
        else:
            foo[index] = latent[index] + gamma*torch.randn_like(latent[index])
    latent.data = foo


def post_optim(latent, g, target_model, target_label, iter2, save_dir):

    n = g.n_latent
    slice = [2,2,2,2,2,2,n-12]
    latent_mean = latent.detach().mean(0).clone().unsqueeze(0)
    styles_ = []
    for _ in range(len(slice)):
        tmp = latent_mean.unsqueeze(1)
        tmp += torch.randn_like(tmp) * 0.0
        tmp.requires_grad = True
        styles_.append(tmp)

    lr2 = 1e-2
    optimizer2 = optim.Adam(styles_, lr=lr2)

    st_top, st_left = 50, 50
    h, w = 160, 160

    print('start post-optimization!')
    pbar2 = tqdm(range(iter2))
    for i in pbar2:
        t   = (i+1)/iter2
        tjj = []
        for id, j in enumerate(slice):
            tjj.append(styles_[id].repeat(1, j, 1))

        latent2     = torch.cat(tjj, 1)
        latent_dis  = disturb_latent(latent2, 0.1*(1-t))
        imgs_gen, _ = g([latent_dis], input_is_latent=True)
        face_input  = imgs_gen[:,:,st_top:st_top+h, st_left:st_left+w].clamp(min=-1, max=1).add(1).div(2)

        # generate images
        before_norm, outputs1 = target_model.forward_feature(face_input)
        prediction = target_model.forward_classifier(before_norm)
        label      = torch.tensor(target_label).unsqueeze(0).long().cuda()

        # calculate loss
        CE_loss    = nn.CrossEntropyLoss()(prediction, label)
        loss       = CE_loss

        # update
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        # save image
        if (i+1) % iter2 == 0:
            file_name = f'label{target_label}-post{i+1}.jpg'
            uts.save_image(
                imgs_gen,
                save_dir+file_name,
                nrow=1,
                normalize=True,
                range=(-1, 1),
                )

        # print loss
        pbar2.set_description(
            (
                f'CE_loss: {CE_loss.item():.7f};'
            )
        )


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def disturb_latent(latent_in, disturb_strenth):
    latent_n = latent_in
    disturb = torch.randn_like(latent_n) * disturb_strenth
    return latent_n + disturb


def inversion_attack(args, generator, target_model, target_label, latent_in,
                     optimizer, lr_scheduler, latent_std, face_shape):

    print(f'start attack label-{target_label}!')

    post_iter   = args.post_step
    save_dir    = args.save_dir
    attack_step = args.step
    pbar        = tqdm(range(attack_step))
    t_resize    = Resize(face_shape)
    batch       = args.batch
    mut_stren   = 0.5

    for i in pbar:

        # calculate strength of perturbation
        ccoo      = float(i%100)+1
        tmp_dist  = args.disturb * (1-ccoo/100)
        _disturb  = latent_std * tmp_dist

        # perturb and generate images
        latent_n    = disturb_latent(latent_in, _disturb)
        imgs_gen, _ = generator([latent_n], input_is_latent=True)
        face_input  = resize_img_gen_shape(imgs_gen, t_resize)

        # predict
        before_norm, outputs1 = target_model.forward_feature(face_input)
        prediction = target_model.forward_classifier(before_norm)
        label      = torch.tensor(target_label).unsqueeze(0).repeat(batch).long().cuda()

        # calculate loss
        CE_loss    = nn.CrossEntropyLoss()(prediction, label)
        loss       = CE_loss

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # mutation
        if (i+1) % 100 == 0:
            mutation(latent_in, mut_stren, i, attack_step, generator, target_model, t_resize, target_label)

        # print loss
        pbar.set_description(
            (
                f'label-{target_label} CE_loss: {CE_loss.item():.7f};'
            )
        )

    # post-processing
    post_optim(latent_in, generator, target_model, target_label, post_iter, save_dir)


if __name__ == '__main__':
    #---------------------------------------------------
    # arguments
    parser = argparse.ArgumentParser(description='Evolutionary Model Inversion Attack')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_mean_latent', type=int, default=10000)
    parser.add_argument('--img_size', type=int, default=256, help='image size from styleGAN2')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate of Alg.1')
    parser.add_argument('--init_label', type=int, default=0, help='attack start label')
    parser.add_argument('--final_label', type=int, default=1, help='attack ending label')
    parser.add_argument('--step', type=int, default=500, help='total steps of Alg.1')
    parser.add_argument('--disturb', type=float, default=0.03, help='it multiply 16 near the epsilon in Alg.1')
    parser.add_argument('--h', type=int, default=160, help='target model input image height')
    parser.add_argument('--w', type=int, default=160, help='target model input image width')
    parser.add_argument('--batch', type=int, default=8, help='number of w^(i) in Alg.1')
    parser.add_argument('--post_step', type=int, default=500, help='total steps of post-processing')

    parser.add_argument('gan_path', type=str, help='path to styleGAN2')
    parser.add_argument('target_classifier_path', type=str, help='path to target classifier')
    parser.add_argument('save_dir', type=str, help='path to save attacked image')
    parser.add_argument('backbone', type=str, help='backbone of target model: `mobile_net` or `inception_resnetv1` ')
    parser.add_argument('num_classes', type=int, help='total classes of target model')

    args = parser.parse_args()

    #---------------------------------------------------
    device        = args.device
    n_mean_latent = args.n_mean_latent
    img_size      = args.img_size
    gan_ckpt_path = args.gan_path
    facenet_path  = args.target_classifier_path
    num_classes   = args.num_classes
    backbone      = args.backbone
    init_lr       = args.lr
    init_label    = args.init_label
    fina_label    = args.final_label
    face_shape    = [args.h, args.w]
    batch         = args.batch
    #---------------------------------------------------

    # load generator
    g_ema = Generator(img_size, 512, 8, channel_multiplier=1)
    g_ema.load_state_dict(torch.load(gan_ckpt_path, map_location='cpu')['g_ema'], strict=True)
    g_ema.eval()
    g_ema.to(device)

    # load target classifier
    target_model = Facenet(backbone=backbone, num_classes=num_classes)
    target_model.load_state_dict(torch.load(facenet_path, map_location='cpu'), strict=True)
    target_model.eval()
    target_model.to(device)

    # init latent vector w
    with torch.no_grad():
        noise_samples = torch.randn(n_mean_latent, 512, device=device)
        latents       = g_ema.style(noise_samples)
        latent_mean   = latents.mean(0)
        latent_std    = (((latents-latent_mean).pow(2).sum() / n_mean_latent) ** 0.5).item()

    latent_in = torch.zeros((batch, 512)).to(device)
    latent_in.requires_grad = True

    # Adam optimizer
    optimizer    = optim.Adam([latent_in], lr=init_lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)

    # model inversion attack
    print(f'attack from {init_label} to {fina_label-1}')
    for target_label in range(init_label, fina_label):
        with torch.no_grad():
            for i in range(batch):
                j   = random.randint(0, n_mean_latent//3-100)
                tmp = latents[3*j:3*(j+1),:].detach().mean(0).clone()
                latent_in[i,:] = tmp
        # latent_in.requires_grad = True
        optimizer    = optim.Adam([latent_in], lr=init_lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)

        inversion_attack(args, g_ema, target_model, target_label, latent_in, optimizer,
                         lr_scheduler, latent_std, face_shape)

