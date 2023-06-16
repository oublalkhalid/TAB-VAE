import argparse
import torch
import numpy as np
import random
random.seed(12345)
torch.manual_seed(12345)
np.random.seed(123456)
from exp.exp_model import Exp_Model
parser = argparse.ArgumentParser(description='desaggregation_models')

parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='uk_dale.csv', help='data file')  # change if accordingly
parser.add_argument('--target', type=str, default='target', help='target variable in S or MS task')
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; '
                                                               'M: multivariate predict multivariate, '
                                                               'S: univariate predict univariate, '
                                                               'MS: multivariate predict univariate')
parser.add_argument('--freq', type=str, default='t', help='frequency for time features encoding, '
                                                          'options: [s -- second, t -- minutely, h -- hourly, '
                                                          'd -- daily, b -- business days, w -- week, m -- month], '
                                                          'you can also use more detailed frequency like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--arch_instance', type=str, default='res_mbconv', help='path to the architecture instance')

# load data
parser.add_argument('--sequence_length', type=int, default=8, help='length of input sequence')
parser.add_argument('--prediction_length', type=int, default=8, help='prediction sequence length')
parser.add_argument('--percentage', type=float, default=0.02, help='the percentage of the whole dataset')
parser.add_argument('--target_dim', type=int, default=1, help='dimension of target')
parser.add_argument('--input_dim', type=int, default=7, help='dimension of input')
parser.add_argument('--hidden_size', type=int, default=128, help='dimension of input')
parser.add_argument('--embedding_dimension', type=int, default=64, help='feature embedding dimension')

# diffusion process
parser.add_argument('--diff_steps', type=int, default=1000, help='number of the diff step')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout')
parser.add_argument('--beta_schedule', type=str, default='linear', help='the schedule of beta')
parser.add_argument('--beta_start', type=float, default=0.0, help='start of the beta')
parser.add_argument('--beta_end', type=float, default=0.01, help='end of the beta')
parser.add_argument('--scale', type=float, default=0.1, help='adjust diffusion scale')

parser.add_argument('--psi', type=float, default=0.5, help='trade off parameter psi')
parser.add_argument('--lambda1', type=float, default=1.0, help='trade off parameter lambda')
parser.add_argument('--gamma', type=float, default=0.01, help='trade off parameter gamma')

# Bidirectional VAE
parser.add_argument('--mult', type=float, default=1, help='mult of channels')
parser.add_argument('--num_layers', type=int, default=2, help='num of RNN layers')
parser.add_argument('--num_channels_enc', type=int, default=32, help='number of channels in encoder')
parser.add_argument('--channel_mult', type=int, default=2, help='number of channels in encoder')
parser.add_argument('--num_preprocess_blocks', type=int, default=1, help='number of preprocessing blocks')
parser.add_argument('--num_preprocess_cells', type=int, default=3, help='number of cells per block')
parser.add_argument('--groups_per_scale', type=int, default=2, help='number of cells per block')
parser.add_argument('--num_postprocess_blocks', type=int, default=1, help='number of postprocessing blocks')
parser.add_argument('--num_postprocess_cells', type=int, default=2, help='number of cells per block')
parser.add_argument('--num_channels_dec', type=int, default=32, help='number of channels in decoder')
parser.add_argument('--num_latent_per_group', type=int, default=8, help='number of channels in latent variables per group')

# training settings
parser.add_argument('--num_workers', type=int, default=5, help='data loader num workers')
parser.add_argument('--patience', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiment times')
parser.add_argument('--dim', type=int, default=-1, help='forecasting dims')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0200, help='weight decay')
parser.add_argument('--loss_type', type=str, default='kl',help='loss function')
parser.add_argument('--loss_hyperprams', type=str, default='MMD',help='loss function MMD')
# device
parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0',help='device ids of multiple gpus')

args = parser.parse_args()
args.use_gpu = True

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Exp = Exp_Model
all_mse = []
all_mae = []
for ii in range(0, args.itr):
    # setting record of experiments
    setting = '{}_sl_{}_pl{}_{}_dim{}_scale{}_diffsteps{}'.format(args.data_path, args.sequence_length,
             args.prediction_length, ii, args.dim,  args.scale, args.diff_steps)
    exp = Exp(args) 
    exp.train(setting)
    mae, mse = exp.test(setting)
    all_mae.append(mae)
    all_mse.append(mse)
    torch.cuda.empty_cache()

print(np.mean(np.array(all_mse)), 
      np.std(np.array(all_mse)),
      np.mean(np.array(all_mae)),
      np.std(np.array(all_mae))
      )
