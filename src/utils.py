import matplotlib.pyplot as plt

class DeNormalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def show_reconstructions(images, dm, ds):
    fig=plt.figure(figsize=(10,5), dpi=150)
    columns = len(images)
    rows = 1


    denorm = DeNormalizer(dm, ds)
    for i, image in enumerate(images):
        ax = fig.add_subplot(rows, columns, i+1)
        plt.axis('off')
        image = denorm(image.squeeze()).permute((1,2,0)).cpu().numpy()
        plt.imshow(image)    

    axes = fig.axes
    for ax, title in zip(axes, ['Original', 'MLP', 'VBMLP']):
        ax.set_title(title)

    fig.tight_layout()
    plt.show()
    plt.close()