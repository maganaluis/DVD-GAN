import os
import torch
from torchvision.utils import save_image
from utils import denorm
from Module.Generator import Generator

device = torch.device("cpu")

class SynthGenerator(object):
    def __init__(self, in_dim, n_class, n_frames, model_path):
        self.in_dim = in_dim
        self.n_class = n_class
        self.n_frames = n_frames
        self.G = Generator(in_dim=in_dim, n_frames=n_frames, n_class=n_class).to(device)
        self.G.load_state_dict(torch.load(model_path))

    def generate(self, class_id, folder_path, prefix=""):
        fixed_z = torch.randn(2, self.n_class).to(device)
        fixed_label = torch.tensor([class_id] * 2).to(device)
        front, back = self.G(fixed_z, fixed_label) # 2 x 48 x 3 x 64 x 64 or B x T x C x H x W
        front, back = denorm(front.data, back.data)
        # save_image(front, "img_front.png")
        front_img = "img_front.png"
        back_img = "img_back.png"
        if prefix:
            front_img = f"{prefix}_img_front.png"
            back_img = f"{prefix}_img_back.png"
        save_image(front, os.path.join(folder_path, front_img))
        save_image(back, os.path.join(folder_path, back_img))

# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y



