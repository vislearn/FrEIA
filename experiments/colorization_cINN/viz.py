from os.path import join

from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np

import config as c
import data

n_imgs = 4
n_plots = 2
figsize = (4,4)
im_width = c.img_dims[1]

class Visualizer:
    def __init__(self, loss_labels):
            self.n_losses = len(loss_labels)
            self.loss_labels = loss_labels
            self.counter = 0

            header = 'Epoch'
            for l in loss_labels:
                header += '\t\t%s' % (l)

            self.config_str = ""
            self.config_str += "==="*30 + "\n"
            self.config_str += "Config options:\n\n"

            for v in dir(c):
                if v[0]=='_': continue
                s=eval('c.%s'%(v))
                self.config_str += "  {:25}\t{}\n".format(v,s)

            self.config_str += "==="*30 + "\n"

            print(self.config_str)
            print(header)

    def update_losses(self, losses, *args):
        print('\r', '    '*20, end='')
        line = '\r%.3i' % (self.counter)
        for l in losses:
            line += '\t\t%.4f' % (l)

        print(line)
        self.counter += 1

    def update_images(self, *img_list):
        w = img_list[0].shape[2]
        k = 0
        k_img = 0

        show_img = np.zeros((3, w*n_imgs, w*n_imgs), dtype=np.uint8)
        img_list_np = []
        for im in img_list:
            im_np = im
            img_list_np.append(np.clip((255. * im_np), 0, 255).astype(np.uint8))

        for i in range(n_imgs):
            for j in range(n_imgs):
                show_img[:, w*i:w*i+w, w*j:w*j+w] = img_list_np[k]

                k += 1
                if k >= len(img_list_np):
                    k = 0
                    k_img += 1

        plt.imsave(join(c.img_folder, '%.4d.jpg'%(self.counter)), show_img.transpose(1,2,0))
        return zoom(show_img, (1., c.preview_upscale, c.preview_upscale), order=0)

    def update_hist(self, *args):
        pass

    def update_running(self, *args):
        pass


if c.live_visualization:
    import visdom

    class LiveVisualizer(Visualizer):
        def __init__(self, loss_labels):
            super().__init__(loss_labels)
            self.viz = visdom.Visdom()#env='mnist')
            self.viz.close()
            self.config_box = self.viz.text('<pre>' + self.config_str + '</pre>')
            self.running_box = self.viz.text('<h1 style="color:red">Running</h1>')

            self.l_plots = self.viz.line(X = np.zeros((1,self.n_losses)),
                                         Y = np.zeros((1,self.n_losses)),
                                         opts = {'legend':self.loss_labels})

            self.imgs = self.viz.image(np.random.random((3, im_width*n_imgs*c.preview_upscale,
                                                            im_width*n_imgs*c.preview_upscale)))

            self.fig, self.axes = plt.subplots(n_plots, n_plots, figsize=figsize)
            self.hist = self.viz.matplot(self.fig)


        def update_losses(self, losses, logscale=False):
            super().update_losses(losses)
            its = min(len(data.train_loader), c.n_its_per_epoch)
            y = np.array([losses])
            if logscale:
                y = np.log10(y)

            self.viz.line(X = (self.counter-1) * np.ones((1,self.n_losses)),
                          Y = y,
                          opts   = {'legend':self.loss_labels},
                          win    = self.l_plots,
                          update = 'append')

        def update_images(self, *img_list):

            show_imgs = super().update_images(*img_list)
            self.viz.image(show_img, win = self.imgs)

            w = img_list[0].shape[2]
            k = 0
            k_img = 0

            show_img = np.zeros((3, w*n_imgs, w*n_imgs), dtype=np.uint8)
            img_list_np = []
            for im in img_list:
                im_np = im
                img_list_np.append(np.clip((255. * im_np), 0, 255).astype(np.uint8))

            for i in range(n_imgs):
                for j in range(n_imgs):
                    show_img[:, w*i:w*i+w, w*j:w*j+w] = img_list_np[k]

                    k += 1
                    if k >= len(img_list_np):
                        k = 0
                        k_img += 1
            show_img = zoom(show_img, (1., c.preview_upscale, c.preview_upscale), order=0)
            self.viz.image(show_img, win = self.imgs)

        def update_hist(self, data):
            for i in range(n_plots):
                for j in range(n_plots):
                    try:
                        self.axes[i,j].clear()
                        self.axes[i,j].hist(data[:, i*n_plots + j], bins=20, histtype='step')
                    except ValueError:
                        pass

            self.fig.tight_layout()
            self.viz.matplot(self.fig, win=self.hist)

        def update_running(self, running=True):
            if running:
                self.viz.text('<h1 style="color:red">Running</h1>', win=self.running_box)
            else:
                self.viz.text('<h1 style="color:green">Done</h1>', win=self.running_box)

        def close(self):
            self.viz.close(win=self.hist)
            self.viz.close(win=self.imgs)
            self.viz.close(win=self.l_plots)
            self.viz.close(win=self.running_box)
            self.viz.close(win=self.config_box)


    visualizer = LiveVisualizer(c.loss_names)
else:
    visualizer = Visualizer(c.loss_names)

def show_loss(losses, logscale=False):
    visualizer.update_losses(losses)

def show_imgs(*imgs):
    visualizer.update_images(*imgs)

def show_hist(data):
    visualizer.update_hist(data.data)

def signal_start():
    visualizer.update_running(True)

def signal_stop():
    visualizer.update_running(False)

def close():
    visualizer.close()

