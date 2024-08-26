import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm
import yaml
from src.datasets.vignettes import Vignettes
from src.models.congealing import Congealing
from src.models.naive import Naive
from src.models.vae import VariationalAutoencoder
from src.utils.paths import DATA_PATH, RESULTS_PATH, CONFIGS_PATH
from src.utils.utils import transform, coerce_to_path_and_check_exist, coerce_to_path_and_create_dir
from src.utils.metrics import Metrics


def train_naive(sample, device, init=None):
    model = Naive(init, sample.to(device))
    loss, recons = model()
    return model.mean, loss.item(), recons.detach().cpu(), model


def train_congealing(config, sample, cat, device, init=None):
    model = Congealing(init, sample.to(device), learn_prototype=cat == 'normal').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['training']['scheduler']['gamma'])
    model.train()
    for i in tqdm(range(config['training']['n_epochs'])):
        optimizer.zero_grad()
        loss, _ = model()
        loss.backward()
        optimizer.step()
        if i == config['training']['scheduler']['lr_step']:
            scheduler.step()
    model.eval()
    loss, recons = model()
    return model.mask, loss.item(), recons.detach().cpu(), model


def train_vae(config, sample, device):
    batch_size = config['training']['batch_size']
    model = VariationalAutoencoder(sample.to(device),
                                   latent_dims=config['model']['latent_dims'],
                                   feature_size=config['model']['feature_size'],
                                   height=sample.shape[2],
                                   width=sample.shape[3]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['training']['scheduler']['gamma'])
    model.train()
    for i in tqdm(range(config['training']['n_epochs'])):
        curr_sample = sample[None].expand(batch_size, -1, -1, -1, -1)
        curr_sample = curr_sample.reshape(batch_size*4, 3, sample.shape[2], sample.shape[3])
        curr_sample_gt = curr_sample[np.random.choice(batch_size*4, batch_size)].to(device)
        curr_sample = transform(curr_sample_gt)
        model.samples = curr_sample
        model.gt_samples = curr_sample_gt
        optimizer.zero_grad()
        loss, _ = model()
        loss = loss + config['training']['kl_alpha'] * model.encoder.kl
        loss.backward()
        optimizer.step()
        if i == config['training']['scheduler']['lr_step']:
            scheduler.step()
    model.eval()
    model.gt_samples = sample.to(device)
    model.samples = sample.to(device)
    loss, recons = model(mode='test')
    return None, loss.item(), recons.detach().cpu(), model


def train(config, sample, device, cat=None, init=None):
    print('   Training...')
    if config['model']['name'] == 'vae':
        return train_vae(config, sample, device)
    elif config['model']['name'] == 'congealing':
        return train_congealing(config, sample, cat, device, init)
    elif config['model']['name'] == 'naive':
        return train_naive(sample, device, init)
    else:
        raise ValueError(f'model name has to be one of `naive`, `congealing` or `vae` not {config["model"]["name"]}.')


def evaluate(config, model, dataset, cat, device, prototype):
    print(f'   Evaluating on {cat}...')
    if config['model']['name'] == 'vae':
        model.eval()
        with torch.no_grad():
            model.samples = dataset.samples[cat][0].unsqueeze(0).to(device)
            loss, recons = model(mode='test')
    elif config['model']['name'] in ['congealing', 'naive']:
        _, loss, recons, _ = train(config, dataset.samples[cat][0].unsqueeze(0), device, cat, prototype)
    else:
        raise ValueError(f'model_name has to be one of `naive`, `congealing` or `vae` not {config["model"]["name"]}.')
    return loss, recons


def main(config, res_dir):
    device = 'cuda'
    glyphs = os.listdir(DATA_PATH)
    glyphs.sort()
    metrics = Metrics(glyphs, config['num_threshold'])
    metrics_changed = Metrics(glyphs, config['num_threshold'])
    for glyph_id, glyph in enumerate(glyphs):

        # Load dataset
        print(glyph)
        dataset = Vignettes(glyph)

        # Train on normal
        prototype, base_loss, recons, model = train(config, dataset.samples['normal'], device, cat='normal')
        if not os.path.exists(os.path.join(res_dir, glyph)):
            os.mkdir(os.path.join(res_dir, glyph))
        if prototype is not None:
            if config['model']['name'] == 'naive':
                prototype_to_save = prototype.permute(1, 2, 0)
            elif config['model']['name'] == 'congealing':
                prototype_to_save = 1 - torch.sigmoid(prototype)
            else:
                raise ValueError(
                    f'model_name has to be one of `naive` or `congealing` not {config["model"]["name"]}.')
            plt.imsave(os.path.join(res_dir, glyph, 'prototype.jpg'),
                       prototype_to_save.detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        max_diff = 0
        for k in range(recons.shape[0]):
            plt.imsave(os.path.join(res_dir, glyph, f'input_normal_{k}.jpg'),
                       dataset.samples['normal'][k].permute(1, 2, 0).numpy())
            plt.imsave(os.path.join(res_dir, glyph, f'recons_normal_{k}.jpg'),
                       torch.clamp(recons[k].permute(1, 2, 0), 0, 1).numpy())
            if ((dataset.samples['normal'][k] - recons[k])**2).mean(0).max() > max_diff:
                max_diff = ((dataset.samples['normal'][k] - recons[k])**2).mean(0).max().item()
        np.save(os.path.join(res_dir, glyph, f'gt_mask.npy'), dataset.label.numpy())

        # Evaluate on changed/unchanged
        predictions = []
        max_diff_changed = max_diff
        for cat in ['changed', 'unchanged']:
            _, recons = evaluate(config, model, dataset, cat, device, prototype)
            plt.imsave(os.path.join(res_dir, glyph, f'input_{cat}.jpg'),
                       dataset.samples[cat][0].permute(1, 2, 0).numpy())
            plt.imsave(os.path.join(res_dir, glyph, f'recons_{cat}.jpg'),
                       torch.clamp(recons.detach().cpu()[0].permute(1, 2, 0), 0, 1).numpy())
            curr_diff_img = ((dataset.samples[cat][0] - recons[0])**2).mean(0).detach().cpu().numpy()
            predictions.append(curr_diff_img)
            if ((dataset.samples[cat][0] - recons[0])**2).mean(0).max() > max_diff:
                max_diff = ((dataset.samples[cat][0] - recons[0])**2).mean(0).max().item()
                if cat == 'changed':
                    max_diff_changed = ((dataset.samples[cat][0] - recons[0])**2).mean(0).max().item()
        ground_truth_mask = np.concatenate([dataset.label.numpy(), dataset.label.numpy()*0], axis=1)
        predictions_all = np.concatenate(predictions, axis=1) / max_diff
        metrics.update(predictions_all, ground_truth_mask, glyph_id)
        metrics_changed.update(predictions[0] / max_diff_changed, dataset.label.numpy(), glyph_id)
        print('   Done!')

    # Compute and save metrics
    metrics = metrics.compute_scores()
    metrics_changed = metrics_changed.compute_scores()
    with open(os.path.join(res_dir, "metrics_changed_unchanged.json"), "w") as f:
        f.write(json.dumps(metrics, indent=4))
    with open(os.path.join(res_dir, "metrics_changed.json"), "w") as f:
        f.write(json.dumps(metrics_changed, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline to train a NN model specified by a YML config")
    parser.add_argument("-c", "--config", nargs="?", type=str, required=True, help="Config file name")
    args = parser.parse_args()

    print(f'Configuration file is {args.config}.')

    assert args.config is not None
    config_path = coerce_to_path_and_check_exist(CONFIGS_PATH / args.config)
    with open(config_path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    curr_res_dir = RESULTS_PATH / cfg['model']['name']
    coerce_to_path_and_create_dir(curr_res_dir)
    print(json.dumps(cfg, indent=2))
    with open(os.path.join(curr_res_dir, "conf.json"), "w") as file:
        file.write(json.dumps(cfg, indent=4))
    main(cfg, curr_res_dir)
