#####################
# Which experiment: #
#####################

# Train to colorize the 'colorized mnist' images,
# instead of conditional generation
colorize = False

#########
# Data: #
#########

data_mean = 0.0
data_std  = 1.0
img_dims = (28, 28)
output_dim = img_dims[0] * img_dims[1]
if colorize:
    output_dim *= 3

add_image_noise = 0.15

##############
# Training:  #
##############

lr = 1e-4
batch_size = 512
decay_by = 0.01
weight_decay = 1e-5
betas = (0.9, 0.999)

do_rev = False
do_fwd = True

n_epochs = 120 * 12
n_its_per_epoch = 2**16

init_scale = 0.03
pre_low_lr = 1

#################
# Architecture: #
#################

# For cond. generation:
n_blocks = 24
internal_width = 512
clamping   = 1.5

# For colorization:
#n_blocks = 7
#n_blocks_conv = 3
#internal_width = 256
#internal_width_conv = 64
#clamping = 1.9
cond_width = 64                             # Output size of conditioning network

fc_dropout = 0.0

####################
# Logging/preview: #
####################

loss_names = ['L', 'L_rev']
preview_upscale = 3                         # Scale up the images for preview
sampling_temperature = 0.8                  # Sample at a reduced temperature for the preview
live_visualization = False                  # Show samples and loss curves during training, using visdom
progress_bar = True                         # Show a progress bar of each epoch

###################
# Loading/saving: #
###################

load_file = 'output/checkpoint.pt'                              # Load pre-trained network
filename = 'output/mnist_cinn.pt'           # Save parameters under this name
cond_net_file = ''                          # Filename of the feature extraction network (only colorization)

checkpoint_save_interval = 120 * 3
checkpoint_save_overwrite = True            # Overwrite each checkpoint with the next one
checkpoint_on_error = True                  # Write out a checkpoint if the training crashes
