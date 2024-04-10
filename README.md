# Self-Discovering Interpretable Diffusion Latent Directions for Responsible Text-to-Image Generation

###  [Project Page](https://interpretdiffusion.github.io/) | [Paper](https://arxiv.org/abs/2311.17216)

This is the official code repository for the CVPR 2024 paper titled "Self-Discovering Interpretable Diffusion Latent Directions for Responsible Text-to-Image Generation". Please do not hesitate to reach out for any questions.


## Installation Instructions

This code has been verified with python 3.9 and CUDA version 11.7. To get started, navigate to the `InterpretDiffusion` directory and install the necessary packages using the following commands:

```bash
git clone git@github.com:hangligit/InterpretDiffusion.git
cd InterpretDiffusion
pip install -r requirements.txt
pip install -e diffusers
```


## Model Explanation
Our model is implemented based on the `diffusers` library, and we have adapted these two files specifically for our approach:

- diffusers/src/diffusers/models/unet_2d_condition.py
- diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py


## Demo
To see an example of how to perform inference with our pretrained concept vector, open and run the `demo.ipynb` notebook. We provide a set of pretrained concept vectors and the corresponding dictionary in `checkpoints`.


## Training
The following steps illustrate how to find a concept vector, e.g., female, in a text-to-image diffusion model.

### Data Generation
Run the following script to generate the training data. This takes about 1 hour for 1000 images on a single A100 GPU.

```bash
python data_creation.py
```


### Training
The following script trains the concept vector. Training a single concept vector on 1000 images for 20 epochs takes approximately 2 hours on a single A100 GPU.

Please configure either wandb or tensorboard to monitor the training and validation process. Visualizing the training procedure is crucial as it indicates whether the model has learned or not.

```bash
python train.py --train_data_dir datasets/person/ --output_dir exps/exp_person
```

You can find a typical training procedure in the directory `exps/example_person`.

### Testing
The following script output images for the prompt "a doctor" with the concept vector "female". For additional evaluation options, please refer to `test.py`

```bash
python test.py --train_data_dir datasets/person/ --output_dir exps/exp_person --num_test_samples 10 --prompt "a doctor"
```


## Citing our work
If our work has contributed to your research, we would greatly appreciate an acknowledgement by citing us as follows:
```
@InProceedings{li2024self,
        author    = {Li, Hang and Shen, Chengzhi and Torr, Philip and Tresp, Volker and Gu, Jindong},
        title     = {Self-discovering interpretable diffusion latent directions for responsible text-to-image generation},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2024},
        
    }
