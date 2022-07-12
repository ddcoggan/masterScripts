import torch
import torch.utils.model_zoo

from cornet.cornet_z import CORnet_Z
from cornet.cornet_z import HASH as HASH_Z
from cornet.cornet_r import CORnet_R
from cornet.cornet_r import HASH as HASH_R
from cornet.cornet_rt import CORnet_RT
from cornet.cornet_rt import HASH as HASH_RT
from cornet.cornet_s import CORnet_S
from cornet.cornet_s import HASH as HASH_S
from cornet.cornet_s_varRec import CORnet_S_varRec
from cornet.cornet_s_varRec import HASH as HASH_S_varRec

def get_model(model_letter, pretrained=False, map_location=None, **kwargs):
    if len(model_letter) < 3:
        model_letter = model_letter.upper()
    else:
        model_letter = model_letter[0].upper() + model_letter[1:]
    model_hash = globals()[f'HASH_{model_letter}']
    model = globals()[f'CORnet_{model_letter}'](**kwargs)
    model = torch.nn.DataParallel(model)
    if pretrained:
        url = f'https://s3.amazonaws.com/cornet-models/cornet_{model_letter.lower()}-{model_hash}.pth'
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=map_location)
        model.load_state_dict(ckpt_data['state_dict'])
    return model


def cornet_z(pretrained=False, map_location=None):
    return get_model('z', pretrained=pretrained, map_location=map_location)


def cornet_r(pretrained=False, map_location=None, times=5):
    return get_model('r', pretrained=pretrained, map_location=map_location, times=times)


def cornet_rt(pretrained=False, map_location=None, times=5):
    return get_model('rt', pretrained=pretrained, map_location=map_location, times=times)


def cornet_s(pretrained=False, map_location=None):
    return get_model('s', pretrained=pretrained, map_location=map_location)

def cornet_s_varRec(pretrained=False, map_location=None, times=(2,2,4,2)):
    return get_model('s_varRec', pretrained=pretrained, map_location=map_location, times=times)