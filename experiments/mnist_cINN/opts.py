import argparse
import config as c

def parse(args):

    parser = argparse.ArgumentParser(prog=args[0])

    parser.add_argument('-l', '--lr',        default=c.lr,                dest='lr', type=float)
    parser.add_argument('-d', '--decay',     default=c.decay_by,          dest='decay_by', type=float)

    parser.add_argument('-b', '--batchsize', default=c.batch_size,        dest='batch_size', type=int)
    parser.add_argument('-n', '--batches',   default=c.n_its_per_epoch,   dest='n_its_per_epoch', type=int)
    parser.add_argument('-N', '--epochs',    default=c.n_epochs,          dest='n_epochs', type=int)

    parser.add_argument('-i', '--in',        default=c.load_file,         dest='load_file', type=str)
    parser.add_argument('-o', '--out',       default=c.filename,          dest='filename', type=str)

    parser.add_argument('--init-scale', default=c.init_scale, dest='init_scale', type=float)

    opts = parser.parse_args(args[1:])

    c.lr              = opts.lr
    c.batch_size      = opts.batch_size
    c.decay_by        = opts.decay_by
    c.n_its_per_epoch = opts.n_its_per_epoch
    c.n_epochs        = opts.n_epochs
    c.load_file       = opts.load_file
    c.filename        = opts.filename
    c.init_scale      = opts.init_scale

