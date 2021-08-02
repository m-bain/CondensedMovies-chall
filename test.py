import argparse
import torch
from tqdm import tqdm
import data_loader.data_loader as module_data
import collections
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from sacred import Experiment
import transformers
from trainer.trainer import verbose
import numpy as np
from utils.util import state_dict_data_parallel_fix
import zipfile

ex = Experiment('test')

@ex.main
def run():

    # setup data_loader instances
    config['data_loader']['args']['shuffle'] = False
    data_loader = config.initialize('data_loader', module_data)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # build model architecture
    config['arch']['args']['experts_used'] = data_loader.dataset.experts_used
    model = config.initialize('arch', module_arch)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print('Using random weights')

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    meta_arr = []
    text_embed_arr = []
    vid_embed_arr = []
    print(len(data_loader))
    with torch.no_grad():
        for i, data in tqdm(tqdm(enumerate(data_loader))):
            # leave this for now since not doing anything on the gpu
            if tokenizer is not None:
                data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            data['text'] = {key: val.cuda() for key, val in data['text'].items()}


            _, text_embed, vid_embed = model(data, eval=True)
            text_embed_arr.append(text_embed)
            vid_embed_arr.append(vid_embed)

    text_embeds = torch.cat(text_embed_arr)
    vid_embeds = torch.cat(vid_embed_arr)

    embed_stack = torch.einsum('ted,ved->tve', text_embeds, vid_embeds)
    sims = embed_stack.sum(dim=2) / embed_stack.shape[2]
    #sims = sim_matrix(text_embeds, vid_embeds)
    sims = sims.detach().cpu().numpy()

    # similarity matrix checks
    if sims.min() < 0 or sims.max() > 1:
        ValueError(f"Similarity matrix should be \in [0,1], found {sims.min(), sims.max()}")

    if len(sims.shape) != 2:
        ValueError(f"Similarity matrix should be 2-D, not {sims.shape}")

    if sims.shape[0] != sims.shape[1]:
        ValueError(f"Expects similarity matrix to be square, since num_captions == num_videos, recieved {sims.shape}")


    # save similarity matrix
    if config.resume is not None:
        sim_save_dir = config.resume.parent
    else:
        sim_save_dir = config._save_dir
        if not sim_save_dir.exists():
            sim_save_dir.mkdir()

    sim_save_fp = sim_save_dir / f"sim_matrix_{data_loader.dataset.split}.npy"
    np.save(sim_save_fp, sims)

    txt_save_fp = sim_save_dir / f"txt_embeds__{data_loader.dataset.split}.npy"
    np.save(txt_save_fp, text_embeds.cpu().numpy())

    vid_save_fp = sim_save_dir / f"vid_embeds__{data_loader.dataset.split}.npy"
    np.save(vid_save_fp, vid_embeds.cpu().numpy())

    if data_loader.dataset.split == 'val':
    #if True:
        # load from numpy file
        # sims = np.load(...)
        # DO what happens during evaluation code


        nested_metrics = {}

        for metric in metric_fns:
            metric_name = metric.__name__
            res = metric(sims)
            verbose(epoch=0, metrics=res, name="", mode=metric_name)
            nested_metrics[metric_name] = res
    elif data_loader.dataset.split == 'test':
        # create zip file for submission
        submission_zip = sim_save_fp.parent / 'submission.zip'
        zipfile.ZipFile(submission_zip, mode='w').write(sim_save_fp, sim_save_fp.name)

        print(f"--For test submission, please upload {submission_zip} to the Codalab site.--\n"
              f"https://competitions.codalab.org/competitions/34124#participate-submit_results")

    # if config.config['visualizer']:
    #     meta_arr_cat = {key: [] for key in meta_arr[0]}
    #     for meta in meta_arr:
    #         for key, val in meta.items():
    #             meta_arr_cat[key] += val

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    #args.add_argument('-t', '--test_submission', action='store_true',
    #                  help='whether to evaluate on test data for test submission, else val.')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type default target')
    options = [
        CustomArgs(['--split'], type=str, default='val', target=('data_loader', 'args', 'split')),
        CustomArgs(['--bs', '--batch_size'], type=int, default=16, target=('data_loader', 'args', 'batch_size')),
    ]
    config = ConfigParser(args, options, test=True)

    if config._config['data_loader']['args']['split'] not in ['val', 'test']:
        raise ValueError("Split should be one of either val or test (the latter for submission), not ")

    ex.add_config(config.config)

    ex.run()
