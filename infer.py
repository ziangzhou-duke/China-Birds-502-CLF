import os 
import json
import GPUtil
import argparse 
import torch 
from tqdm import tqdm 
import torch.nn as nn
import kaldiio
import dataset 
import dataloader 
import module.model as module_model


def get_instance(module, cfs, *args):
    return getattr(module, cfs['type'])(*args, **cfs['args'])


def set_device(n_gpu):
    if n_gpu > 0:
        device='cuda'
        deviceIDs = GPUtil.getAvailable(limit=n_gpu, maxMemory=0.8, maxLoad=0.8)
        assert deviceIDs != [], "n_gpu > 0, but no GPUs available!"
        print("Use GPU:", deviceIDs)
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, deviceIDs))
    else:
        print("Use CPU")
        device='cpu'
    return device


def main(config, args):
    device = set_device(config['n_gpu'])

    # g-vector extractor
    model = get_instance(module_model, config['model'])
    chkpt = torch.load(args.resume)
    try:
        model.load_state_dict(chkpt['model'])
    except:
        model.load_state_dict(chkpt)
    model = model.to(device)

    config['dataset']['args']['wav_scp'] = os.path.join(args.data, 'wav.scp')
    testset = get_instance(dataset, config['dataset'])
    testloader = get_instance(dataloader, config['dataloader'], testset)

    model.eval()
    utt2embd = {}
    for i, (utt, data) in enumerate(tqdm(testloader, ncols=80)):
        utt = utt[0]
        # For utterances longer than 30s, we only truncate the first 30s.
        if data.shape[1] > 3000:
            data = data[:,:3000,:]
        data = data.float().to(device)
        with torch.no_grad():
            embd = model.extractor(data)
        embd = embd.squeeze(0).cpu().numpy()
        utt2embd[utt] = embd

    embd_wfile = 'ark,scp:{0}/embedding.ark,{0}/embedding.scp'.format(args.data)
    with kaldiio.WriteHelper(embd_wfile) as writer:
        for utt, embd in utt2embd.items():
            writer(utt, embd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speaker Verification Inference')
    parser.add_argument('-c', '--config', type=str, required=True, 
                        help='config file path')
    parser.add_argument('-r', '--resume', type=str, required=True, 
                        help='path to latest checkpoint')
    parser.add_argument('data', 
                        help='data directory of inputs and outputs.')
    args = parser.parse_args()
    # Read config of the whole system.
    with open(args.config) as rfile:
        config = json.load(rfile)

    main(config, args)
