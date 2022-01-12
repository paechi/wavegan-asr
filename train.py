import torch
from torch import autograd
from torch import optim
import json
from utils import save_samples
import numpy as np
import pprint
import pickle
import datetime
from torch.utils.data import DataLoader
from wavegan import *
from utils import *
from logger import *
cuda = True if torch.cuda.is_available() else False


# =============Logger===============
LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)

LOGGER.info('Initialized logger.')
init_console_logger(LOGGER)

# =============Parameters===============
args = parse_arguments()
epochs = args['num_epochs']
batch_size = args['batch_size']
latent_dim = args['latent_dim']
ngpus = args['ngpus']
model_size = args['model_size']
model_dir = make_path(os.path.join(args['output_dir'],
                                   datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
args['model_dir'] = model_dir
# save samples for every N epochs.
epochs_per_sample = args['epochs_per_sample']
# gradient penalty regularization factor.
lmbda = args['lmbda']

# Dir
audio_dir = args['audio_dir']
output_dir = args['output_dir']

# =============Network===============
netG = WaveGANGenerator(model_size=model_size, ngpus=ngpus, latent_dim=latent_dim, upsample=True)
netD = WaveGANDiscriminator(model_size=model_size, ngpus=ngpus)

if cuda:
    netG = torch.nn.DataParallel(netG).cuda()
    netD = torch.nn.DataParallel(netD).cuda()

# "Two time-scale update rule"(TTUR) to update netD 4x faster than netG.
optimizerG = optim.Adam(netG.parameters(), lr=args['learning_rate'], betas=(args['beta1'], args['beta2']))
optimizerD = optim.Adam(netD.parameters(), lr=args['learning_rate'], betas=(args['beta1'], args['beta2']))

# Sample noise used for generated output.
sample_noise = torch.randn(args['sample_size'], latent_dim)
if cuda:
    sample_noise = sample_noise.cuda()
sample_noise_Var = autograd.Variable(sample_noise, requires_grad=False)

# Save config.
LOGGER.info('Saving configurations...')
config_path = os.path.join(model_dir, 'config.json')
with open(config_path, 'w') as f:
    json.dump(args, f)

# Load data.
LOGGER.info('Loading audio data...')
window_length = 16384
audio_path_list = get_all_audio_filepaths(audio_dir)[:2000]
dataset = WaveganDataset(audio_dir, window_length)
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)
print('Training size', dataset.__len__())
TOTAL_TRAIN_SAMPLES = dataset.__len__()
BATCH_NUM = TOTAL_TRAIN_SAMPLES // batch_size

# =============Train===============
history = []
D_costs_train = []
D_wasses_train = []
G_costs = []

start = time.time()
LOGGER.info('Starting training...EPOCHS={}, BATCH_SIZE={}, BATCH_NUM={}'.format(epochs, batch_size, BATCH_NUM))
for epoch in range(1, epochs+1):
    LOGGER.info("{} Epoch: {}/{}".format(time_since(start), epoch, epochs))

    D_cost_train_epoch = []
    D_wass_train_epoch = []
    G_cost_epoch = []
    for batch_idx, (hr_iter, lr_iter) in enumerate(dataloader):
        # Set Discriminator parameters to require gradients.
        for p in netD.parameters():
            p.requires_grad = True
        one = torch.tensor(1, dtype=torch.float).cuda()
        neg_one = one * -1
        if cuda:
            one = one.cuda()
            neg_one = neg_one.cuda()
        #############################
        # (1) Train Discriminator
        #############################
        for iter_dis in range(5):
            netD.zero_grad()

            lr_data_Var = numpy_to_var(lr_iter, cuda)
            hr_data_Var = numpy_to_var(hr_iter, cuda)
            if hr_data_Var.shape[0] != batch_size:
                continue
            # a) compute loss contribution from real training data
            D_real = netD(hr_data_Var)
            D_real = D_real.mean()  # avg loss
            D_real.backward(neg_one)  # loss * -1

            # b) compute loss contribution from generated data, then backprop.
            fake = autograd.Variable(netG(lr_data_Var).data)
            D_fake = netD(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # c) compute gradient penalty and backprop
            gradient_penalty = calc_gradient_penalty(netD, hr_data_Var.data,
                                                     fake.data, batch_size, lmbda,
                                                     use_cuda=cuda)
            gradient_penalty.backward(one)

            # Compute cost * Wassertein loss..
            D_cost_train = D_fake - D_real + gradient_penalty
            D_wass_train = D_real - D_fake

            # Update gradient of discriminator.
            optimizerD.step()

            #############################
            # (2) Compute Valid data
            #############################
            if cuda:
                D_cost_train = D_cost_train.cpu()
                D_wass_train = D_wass_train.cpu()

            # Record costs
            D_cost_train_epoch.append(D_cost_train.data.numpy())
            D_wass_train_epoch.append(D_wass_train.data.numpy())

        #############################
        # (3) Train Generator
        #############################
        # Prevent discriminator update.
        for p in netD.parameters():
            p.requires_grad = False

        # Reset generator gradients
        netG.zero_grad()

        lr_data_Var = numpy_to_var(lr_iter, cuda)
        fake = netG(lr_data_Var)
        # Adversarial loss
        loss_GAN = criterion_GAN(netD(fake), lr_iter)

        # Content loss
        gen_features = feature_extractor(fake)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizerG.step()
        # lr_data_Var = numpy_to_var(lr_iter, cuda)
        # fake = netG(lr_data_Var)
        # G = netD(fake)
        # G = G.mean()
        #
        # # Update gradients.
        # G.backward(neg_one)
        # G_cost = -G
        #
        # optimizerG.step()

        # Record costs
        if cuda:
            G_cost = G_cost.cpu()
        G_cost_epoch.append(G_cost.data.numpy())

        if batch_idx % (BATCH_NUM // 5) == 0:
            LOGGER.info("{} Epoch={} Batch: {}/{} D_c:{:.4f} | D_w:{:.4f} | G:{:.4f}".format(time_since(start), epoch,
                                                                                             batch_idx, BATCH_NUM,
                                                                                             D_cost_train.data.numpy(),
                                                                                             D_wass_train.data.numpy(),
                                                                                             G_cost.data.numpy()))

    # Save the average cost of batches in every epoch.
    D_cost_train_epoch_avg = sum(D_cost_train_epoch) / float(len(D_cost_train_epoch))
    D_wass_train_epoch_avg = sum(D_wass_train_epoch) / float(len(D_wass_train_epoch))
    G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

    D_costs_train.append(D_cost_train_epoch_avg)
    D_wasses_train.append(D_wass_train_epoch_avg)
    G_costs.append(G_cost_epoch_avg)

    LOGGER.info("{} D_cost_train:{:.4f} | D_wass_train:{:.4f} | "
                "G_cost:{:.4f}".format(time_since(start),
                                       D_cost_train_epoch_avg,
                                       D_wass_train_epoch_avg,
                                       G_cost_epoch_avg))

    # # Generate audio samples.
    # if epoch % epochs_per_sample == 0:
    #     LOGGER.info("Generating samples...")
    #     sample_out = netG(sample_noise_Var)
    #     if cuda:
    #         sample_out = sample_out.cpu()
    #     sample_out = sample_out.data.numpy()
    #     save_samples(sample_out, epoch, output_dir)

    # TODO
    # Early stopping by Inception Score(IS)

LOGGER.info('>>>>>>>Training finished !<<<<<<<')

# Save model
LOGGER.info("Saving models...")
netD_path = os.path.join(output_dir, "discriminator.pkl")
netG_path = os.path.join(output_dir, "generator.pkl")
torch.save(netD.state_dict(), netD_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
torch.save(netG.state_dict(), netG_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

# Plot loss curve.
LOGGER.info("Saving loss curve...")
plot_loss(D_costs_train, D_wasses_train, G_costs, output_dir)

LOGGER.info("All finished!")





