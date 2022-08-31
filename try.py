import os

from pythae.models import AutoModel
from pythae.samplers import NormalSampler
# Retrieve the trained model
from torchvision.utils import save_image

os.makedirs("./ouptput",exist_ok=True)

my_trained_vae = AutoModel.load_from_folder(r'my_model\VAE_training_2022-08-31_07-33-05\final_model')
# Define your sampler
my_samper = NormalSampler(model=my_trained_vae)
# Generate samples
gen_data = my_samper.sample(
           num_samples=50,
           batch_size=10,
           output_dir=None,
           return_gen=True
            )

save_image(gen_data.data[:25], "./ouptput/result.png" ,nrow=5, normalize=True)