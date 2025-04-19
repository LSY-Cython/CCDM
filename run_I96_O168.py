import argparse
import random
import torch
import numpy as np
from model import DiffMTS
from data_producer.data_factory import create_mts_loader
from analysis import *

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='DiffMTS')

    # basic config
    parser.add_argument('--status', type=str, default='test', help='experiment type')
    parser.add_argument('--device', default=torch.device("cuda:4" if torch.cuda.is_available() else "cpu"), help='device')

    # data loader
    # parser.add_argument('--data_name', type=str, default='ETTh1', help='dataset name')
    # parser.add_argument('--data_path', type=str, default='dataset/ETT/ETTh1.csv', help='dataset path')
    # parser.add_argument('--num_feat', type=int, default=7, help='num of variates')

    # parser.add_argument('--data_name', type=str, default='Weather', help='dataset name')
    # parser.add_argument('--data_path', type=str, default='dataset/Weather/weather.csv', help='dataset path')
    # parser.add_argument('--num_feat', type=int, default=21, help='num of variates')

    # parser.add_argument('--data_name', type=str, default='Electricity', help='dataset name')
    # parser.add_argument('--data_path', type=str, default='dataset/Electricity/electricity.csv', help='dataset path')
    # parser.add_argument('--num_feat', type=int, default=321, help='num of variates')

    # parser.add_argument('--data_name', type=str, default='Exchange', help='dataset name')
    # parser.add_argument('--data_path', type=str, default='dataset/Exchange/exchange.csv', help='dataset path')
    # parser.add_argument('--num_feat', type=int, default=8, help='num of variates')

    parser.add_argument('--data_name', type=str, default='Traffic', help='dataset name')
    parser.add_argument('--data_path', type=str, default='dataset/Traffic/traffic.csv', help='dataset path')
    parser.add_argument('--num_feat', type=int, default=862, help='num of variates')

    # parser.add_argument('--data_name', type=str, default='Appliance', help='dataset name')
    # parser.add_argument('--data_path', type=str, default='dataset/Appliance/appliance.csv', help='dataset path')
    # parser.add_argument('--num_feat', type=int, default=28, help='num of variates')

    # forecasting task
    parser.add_argument('--cont_len', type=int, default=96, help='context sequence length')
    parser.add_argument('--pred_len', type=int, default=168, help='prediction sequence length')
    parser.add_argument('--task', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, '
                             'd:daily, b:business days, w:weekly, m:monthly]')

    # model instantiation
    parser.add_argument('--use_window_norm', type=bool, default=True, help='use instance-wise normalization and '
                                                                           'denormalization to remove non-stationarity')
    parser.add_argument('--n_emb', type=int, default=2, help='num of input embedding layers')
    parser.add_argument('--cont_hidden_dim', type=int, default=256, help='hidden size of context, options:[128, 256, 512, 768]')
    parser.add_argument('--pred_hidden_dim', type=int, default=256, help='hidden size of predition, options:[128, 256, 512, 768]')
    parser.add_argument('--step_hidden_dim', type=int, default=256, help='hidden size of diffusion step')
    parser.add_argument('--time_hidden_dim', type=int, default=256, help='hidden size of timestamp')
    parser.add_argument('--n_depth', type=int, default=2, help='number of DiT blocks')
    parser.add_argument('--n_heads', type=int, default=8, help='num of attention heads')
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='dropout of attention score')
    parser.add_argument('--mlp_ratio', type=int, default=1, help='dimension of feedforward layer in attention')
    parser.add_argument('--non_attn', type=bool, default=False, help='whether replacing attention by linear modules for ablation study')

    # step sampling trick employed in Stable Diffusion 3
    parser.add_argument('--step_dist', type=str, default="uniform", help='options:[uniform, non-uniform]')
    parser.add_argument('--dev_dist', type=str, default="1.0", help='deviation degree from uniform step')

    # contrastive diffusion
    parser.add_argument('--use_contrast', type=str, default="non-contrast", help='options:[contrast, non-contrast]')
    parser.add_argument('--contrast_weight', type=float, default=0.0, help='weight of contrastive loss')
    parser.add_argument('--n_negatives', type=int, default=64, help='number of negative samples for each mode')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature in contrastive loss')

    # noise schedule
    parser.add_argument('--n_steps', type=int, default=100, help='diffusion steps, options:[50, 100]')
    parser.add_argument('--beta_start', type=int, default=0.0001, help='beta starting value')
    parser.add_argument('--beta_end', type=int, default=0.2, help='beta end value')
    parser.add_argument('--beta_schedule', type=str, default="quad", help='options:[linear, quad, cosine]')

    # argument in TimeDiff, y0-prediction is not effective, thus fixed to noise-prediction
    parser.add_argument('--parameterization', type=str, default="noise", help='parameterize reverse transition, '
                                                                              'options:[noise, y0]')

    # optimization
    parser.add_argument('--train_batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=32, help='testing batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='training epochs')  # 200
    parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--num_workers', type=int, default=10, help='number of data loader workers')

    args = parser.parse_args()

    # data preparation
    data_loader = create_mts_loader(args)

    # model = DiffMTS(configs=args, data_loader=data_loader)
    if args.status == "train":
        # model.train(is_refine=False)

        args.use_contrast = 'contrast'
        args.contrast_weight = 0.001
        args.n_negatives = 64
        args.train_batch_size = 32
        data_loader = create_mts_loader(args)
        model = DiffMTS(configs=args, data_loader=data_loader)

        # two-stage training
        # model.train(is_refine=True, init_epoch=198, n_epochs=20)

        # training from scratch
        model.train(is_refine=False)

    else:
        args.use_contrast = 'contrast'
        args.contrast_weight = 0.0001
        args.test_batch_size = 1
        data_loader = create_mts_loader(args)
        model = DiffMTS(configs=args, data_loader=data_loader)

        model.test(best_epoch=18, n_samples=20, is_out=False)
