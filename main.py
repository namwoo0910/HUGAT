# w/ update embeddings
import pandas as pd
import numpy as np
import torch
import utils
import random
import os
from model import HUGAT
import argparse
import torch.nn as nn

def seed(seed = 2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def argument():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--device', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=48, help='The representation dimension (defaults to 320)')
    parser.add_argument('--num-heads', type=list, default=[12], help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=2000, help='The number of epochs')
    parser.add_argument('--seed', type=int, default=0, help='The random seed')
    parser.add_argument('--weight', type=tuple, default=(0.1, 0.9, 1.0), help='The random seed')
    args = parser.parse_args()
    return args


def main():
    # seed()
    args=argument()

    # set random seed for reproducibility
    seed(args.seed)
    # load dataset
    data = utils.load_dataset()
    # Regional Attribute Distribution
    src_matrix = torch.tensor(data['src_matrix']).float().to(args['device'])
    dst_matrix = torch.tensor(data['dst_matrix']).float().to(args['device'])
    chk_hell = data['chk_hell'].float().to(args['device'])
    geo_hell = data['geo_hell'].float().to(args['device'])
    # Downstream Applications
    income = data['income']
    poverty = data['poverty']
    popularity = data['popularity']
    labels = data['labels']
    feats = utils.node_feats(args.repr_dims).to(args['device'])  # torch.tensor(data['feats']).float().to(args['device'])
    # heterograph
    g_ = data['heterograph_unified'][0][0].to(args['device'])
    meta_paths = data['meta_paths']
    weight = data['weight']

    model = HUGAT(g_, meta_paths=meta_paths,
                  in_size=feats.shape[1],
                  out_size=args['out_size'],
                  num_heads=args['num_heads'],
                  dropout=args['dropout'])
    model.to(args['device'])

    # params
    total_params = sum(p.numel() for p in model.parameters())
    print("total_params: {:d}".format(total_params))

    # stooper, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)

    for epoch in range(args['num_epochs']):
        model.train()
        optimizer.zero_grad()
        # emb = model.forward(g_, feats)
        mob_loss = model.get_mob_loss(g_, feats, src_matrix, dst_matrix)
        chk_loss = model.get_chk_loss(g_, feats, chk_hell)
        land_loss = model.get_land_loss(g_, feats, geo_hell)
        loss = args.weight[0]*mob_loss + args.weight[1]*chk_loss + args.weight[2]*land_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        #scheduler.step()
        if (epoch + 1) % 10 == 0:
            print("Epoch {:04d} | Mob Loss = {:.4f} | Chk Loss = {:.4f} | Geo Loss = {:.4f}".format(epoch + 1, mob_loss.item(), chk_loss.item(), land_loss.item()))

    with torch.no_grad():
        #loss = model.get_loss(g_, feats, src_matrix, dst_matrix, chk_hell, geo_hell)
        emb = model.forward(g_, feats)
        emb = emb.detach().cpu().numpy()
    print('emb size:', emb.shape)

    mae_income, rmse_income, r2_income = utils.predict_regression(emb, income)
    mae_poverty, rmse_poverty, r2_poverty = utils.predict_regression(emb, poverty)
    mae_crowded, rmse_crowded, r2_crowded = utils.predict_regression(emb, popularity)
    nmi, ars = utils.region_cluetering(emb, labels)
    print(
        '----------------------------------------------------Results-----------------------------------------------------\n')
    print('\n')
    print('income prediction R2:', r2_income.round(3))
    print('\n')
    print('poverty prediction R2:', r2_poverty.round(3))
    print('\n')
    print('crowded prediction R2:', r2_crowded.round(3))
    print('\n')
    print('region clustering NMI:', nmi.round(3))
    print('region clustering ARI:', ars.round(3))
    print('\n')

if __name__ == "__main__":
    main()