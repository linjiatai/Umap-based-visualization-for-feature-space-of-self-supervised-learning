import os
import argparse
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn
import numpy as np
from tool.dataset import ImageDataset
from tool.PDBL import PDBL_net
from tool.eff import EfficientNet
from tool.resnet import resnet50
from tool.shufflenet import shufflenet_v2_x1_0
from sklearn.metrics import accuracy_score,f1_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def load_SL_pretrained_weight(model, load_dir):
    # 加载SL-model
    backbone = torch.load(load_dir)['model']
    model_dict = dict()
    for key, value in model.state_dict().items():
        if key == 'fc.weight' or key == 'fc.bias':
            continue
        model_dict[key] = backbone.pop('module.backbone.'+key)
    model.load_state_dict(model_dict, strict=False)
    model = model.cuda()
    return model

def create_model(model_name, n_class):
    if model_name == 'shuff':
        model = shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model._stage_out_channels[-1], n_class)
    elif model_name == 'eff':
        model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=n_class)
    elif model_name == 'r50':
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(512 * model.block.expansion, n_class)
    return model

def feature_extraction(model,dataloader):
    model.eval()
    steps = len(dataloader)
    dataiter_train = iter(dataloader)
    print('Training phase ---> number of training items is: ', args.n_item_train)
    work_space_in = np.zeros((args.n_item_train,args.n_feature))
    work_space_out = np.zeros((args.n_item_train,args.n_class))
    progress = 0
    for step in tqdm(range(steps)):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        img224, target = next(dataiter_train)
        len_batch = len(target)
        with torch.no_grad():
            img224 = Variable(img224.float().cuda())
        feature1, _ = model(img224)
        feature = feature1
        work_space_in[  progress:(progress+len_batch),  :] = feature.detach().cpu().numpy()
        work_space_out[  progress:(progress+len_batch),  :] = target.detach().cpu().numpy()
        progress = progress+len_batch
    return work_space_in, work_space_out

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def main(args):
    dataset = ImageDataset(data_path = args.datadir, n_class=args.n_class)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    args.n_item_train = len(dataset)
    model_names = ['r50']
    n_features = [3840]
    

    args.model_name = model_names[0]
    args.n_feature = n_features[0]

    print('<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('This model is', args.model_name)
    ##  Load CNN model
    model = create_model(model_name=args.model_name,n_class=args.n_class)
    model = load_SL_pretrained_weight(model, args.SL_pretrained_weight)
    model = model.cuda()

    train_feature, train_label = feature_extraction(model, loader)
    train_label = np.argmax(train_label, axis=1)


    fig = plt.figure(figsize=(40,30), dpi=600)
    result = umap.UMAP(n_neighbors=10,
                        min_dist=0.5,
                        metric='correlation',
                        random_state=16).fit_transform(train_feature)

    x_min, x_max = np.min(result, 0), np.max(result, 0)
    result = (result - x_min) / (x_max - x_min)

    for j in range(result.shape[0]):
        plt.text(result[j, 0], result[j, 1], str(train_label[j]),
                color=plt.cm.Set1(train_label[j] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
    plt.title('Umap')
    plt.savefig('save/feature_visualization_for_SL.png')
    test = 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pyramidal Deep-Broad Learning')
    parser.add_argument('--device',     default='0',        type=str, help='index of GPU')
    parser.add_argument('--save_dir',   default='save/',    type=str, help='Save path of learned PDBL')
    parser.add_argument('--datadir',   default='./dataset/KME/',     type=str, help='Path of training set')
    parser.add_argument('--batch_size', default=10,         type=int, help='Batch size of dataloaders')
    parser.add_argument('--n_class',    default=9,          type=int, help='Number of categories')
    parser.add_argument('--n_workers',  default=0,          type=int, help='Number of workers')
    parser.add_argument('--SL_pretrained_weight', default='checkpoint/checkpoint_ep_11.pth', type = str, help=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)
