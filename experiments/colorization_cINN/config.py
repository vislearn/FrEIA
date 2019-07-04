#################
# Architecture: #
#################

# Image size of L, and ab channels respectively:
img_dims_orig = (256, 256)
img_dims = (img_dims_orig[0] // 4, img_dims_orig[0] // 4)
# Clamping parameter in the coupling blocks (higher = less stable but more expressive)
clamping = 1.5

#############################
# Training hyperparameters: #
#############################

seed = 9287
batch_size = 48
device_ids = [0,1,2]                # GPU ids. Set to [0] for single GPU

log10_lr = -4.0                     # Log learning rate
lr = 10**log10_lr
lr_feature_net = lr                 # lr of the cond. network

n_epochs = 120 * 4
n_its_per_epoch = 32 * 8            # In case the epochs should be cut short after n iterations

weight_decay = 1e-5
betas = (0.9, 0.999)                # concerning adam optimizer

init_scale = 0.030                  # initialization std. dev. of weights (0.03 is approx xavier)
pre_low_lr = 0                      # for the first n epochs, lower the lr by a factor of 20

#######################
# Dataset parameters: #
#######################

dataset = 'imagenet'                # also 'coco' is possible. Todo: places365
validation_images = './imagenet/validation_images.txt'
shuffle_val = False
val_start = 0                       # use a slice [start:stop] of the entire val set
val_stop = 5120

end_to_end = True                   # Whether to leave the cond. net fixed
no_cond_net = False                 # Whether to use a cond. net at all
pretrain_epochs = 0                 # Train only the inn for n epochs before end-to-end

########################
# Display and logging: #
########################

sampling_temperature = 1.0          # latent std. dev. for preview images
loss_display_cutoff = 10            # cut off the loss so the plot isn't ruined
loss_names = ['L', 'lr']
preview_upscale = 256 // img_dims_orig[0]
img_folder = './images'
silent = False
live_visualization = False
progress_bar = False

#######################
# Saving checkpoints: #
#######################

load_inn_only = ''                  # only load the inn part of the architecture
load_feature_net = ''               # only load the cond. net part
load_file = ''                      # load entire architecture (overwrites the prev. 2 options)
filename = 'output/full_model.pt'   # output filename

checkpoint_save_interval = 60
checkpoint_save_overwrite = False   # Whether to overwrite the old checkpoint with the new one
checkpoint_on_error = True          # Wheter to make a checkpoint with suffix _ABORT if an error occurs
