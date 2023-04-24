import sys
import torch
from torch import nn
from typing import List
from diffusion_utils.diffusion_multinomial import index_to_log_onehot,log_onehot_to_index


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_feature_extractor(model_type, **kwargs):
    """ Create the feature extractor for <model_type> architecture. """
    if model_type == 'ddpm':
        print("Creating DDPM Feature Extractor...")
        feature_extractor = FeatureExtractorDDPM(**kwargs)
    # elif model_type == 'mae':
    #     print("Creating MAE Feature Extractor...")
    #     feature_extractor = FeatureExtractorMAE(**kwargs)
    # elif model_type == 'swav':
    #     print("Creating SwAV Feature Extractor...")
    #     feature_extractor = FeatureExtractorSwAV(**kwargs)
    # elif model_type == 'swav_w2':
    #     print("Creating SwAVw2 Feature Extractor...")
    #     feature_extractor = FeatureExtractorSwAVw2(**kwargs)
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return feature_extractor

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out

class FeatureExtractor(nn.Module):
    def __init__(self, model_path: str, input_activations: bool, **kwargs):
        ''' 
        Parent feature extractor class.
        
        param: model_path: path to the pretrained model
        param: input_activations: 
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        print(f"Pretrained model is successfully loaded from {model_path}")
        self.save_hook = save_input_hook if input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, model_path: str, **kwargs):
        pass

class FeatureExtractorDDPM(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        
        # Save decoder activations
        for idx, block in enumerate(self.model._denoise_fn.ups):
            print(idx)
            if idx in blocks:
                block[2].register_forward_hook(self.save_hook)
                self.feature_blocks.append(block[2])
        for idx, block in enumerate([self.model._denoise_fn.mid_block2]):
            print(idx)
            if idx in blocks:
                block.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block)
        

    def _load_pretrained_model(self, model_path, **kwargs):
        from layers.layers import SegmentationUnet
        from diffusion_utils.diffusion_multinomial import MultinomialDiffusion

        # models: UNet and Diffusion 
        # UNet is used to gather activations
        # Diffusion is used to get noisy samples
        unet = SegmentationUnet(
            num_classes=kwargs['num_classes'],
            dim=kwargs['dim'],
            num_steps=kwargs['num_steps'],
            dim_mults=kwargs['dim_mults']
        )

        self.model = MultinomialDiffusion(
            num_classes=kwargs['num_classes'],
            shape=kwargs['shape'],
            denoise_fn=unet,
            timesteps=kwargs['num_steps']
        ).to(kwargs['device'])

        dict = torch.load(model_path)
        self.model.load_state_dict(dict['model'])
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.tensor([t]).to(x.device)
            log_x_start = index_to_log_onehot(x, num_classes=2)
            log_x_t = self.model.q_sample(log_x_start=log_x_start, t=t)
            x_t = log_onehot_to_index(log_x_t)
        
            self.model._denoise_fn(x=x_t, time=t)

            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations
    

def collect_features(args, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = tuple(args['shape'][1:])
    resized_activations = []
    for feats in activations:
        feats = feats[sample_idx][None]
        feats = nn.functional.interpolate(
            feats, size=size, mode=args["upsample_mode"]
        )
        resized_activations.append(feats[0])
    
    return torch.cat(resized_activations, dim=0)