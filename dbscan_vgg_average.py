# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import argparse
import logging
import random
import sys
from os import path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from glob import glob
from sklearn.decomposition import PCA, KernelPCA
import umap
#from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import torchvision.models as models
import math
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score

from util import AverageMeter, learning_rate_decay, load_model, Logger
from sklearn.cluster import KMeans

def image_path_to_name(image_path):
    # return np.string_(path.splitext(path.basename(image_path))[0])
    parent, image_name = path.split(image_path)
    image_name = path.splitext(image_name)[0]
    parent = path.split(parent)[1]
    return path.join(parent, image_name)

class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, layer_num, num_labels):
        super(RegLog, self).__init__()
        if layer_num ==1:
            conv = 2
        elif layer_num ==2:
            conv = 4
        elif layer_num ==3:
            conv = 7
        elif layer_num ==4:
            conv = 10
        elif layer_num ==5:
            conv = 13

        self.conv = conv
        if conv==2:
            self.av_pool = nn.AvgPool2d(19, stride=19, padding=2)
            s = 9216
        elif conv==4:
            self.av_pool = nn.AvgPool2d(14, stride=14, padding=0)
            s = 8192
        elif conv==7:
            self.av_pool = nn.AvgPool2d(10, stride=10, padding=2)
            s = 9216
        elif conv==10:
            self.av_pool = nn.AvgPool2d(7, stride=7, padding=0)
            s = 8192
        elif conv==13:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=1)
            s = 8192
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)


def forward(x, model, conv):
    if hasattr(model, 'sobel') and model.sobel is not None:
        x = model.sobel(x)
    count = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
        if isinstance(m, nn.ReLU):
            if count == conv:
                return x
            count = count + 1
    return x

class ListDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_list,
                 transform=None,
                 loader=default_loader):
        self.images_list = images_list
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.images_list[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.images_list)

def EpsDBSCAN(D, k):
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(D)
    distances, indices = nn.kneighbors(D)
    distances = np.delete(distances, 0, 1)
    Dist = distances.max(axis=1)
    Array = sorted(Dist)
    AvgDist = distances.sum(axis=1)/k
    Avg_Array = sorted(AvgDist)
    plt.plot(Avg_Array, 'b')

    num = len(Avg_Array)
    n_Array = [0 for i in range(num)]
    minArray = min(Avg_Array)
    maxArray = max(Avg_Array)

    for i in range(num):
        n_Array[i] = (Avg_Array[i]-minArray)/(maxArray-minArray)*(1.0-0.0)

    bins = np.linspace(0, 1, 10)
    bin_indice = np.digitize(n_Array, bins)
    Eps = []
    Avg_Array = np.array(Avg_Array)
    count_max = 0

    for i in range(10):
        count = len(np.where(bin_indice == i)[0])
        if count >= k:
            #print count
            e = np.sum(Avg_Array[bin_indice == i], axis=0)/count
            plt.hlines(e, xmin=0, xmax=len(Array), colors='r')
            Eps.append(e)

    N = len(Eps)
    Eps_index = []

    for i in range(N):
        for j in range(num):
            if Avg_Array[j] > Eps[i]:
                Eps_index.append(j)
                break

    ave_slope = (maxArray - minArray)/num
    
    #print 'ave slope'
    #print ave_slope
    #print ''
    for i in range(N-1):
        slope = (Eps[i+1] - Eps[i]) / (Eps_index[i+1] - Eps_index[i])
        #print slope
        if slope > ave_slope * 2:
            out = Eps[i]
            break
        else:
            out = Eps[i+1]

    return Eps

def EpsValue(D, k):
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(D)
    distances, indices = nn.kneighbors(D)
    distances = np.delete(distances, 0, 1)
    Dist = distances.max(axis=1)
    AvgDist = distances.sum(axis=1)/k

    out = (max(Dist) - min(AvgDist))/100

    return min(AvgDist), out

def extract_features_to_disk(image_paths, model, batch_size, workers, reglog, layer):
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #print image_paths
    dataset = ListDataset(image_paths,
                          transforms.Compose([
                              transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              normalize,
                          ]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)

    features = {}
    for i, (input_data, paths) in enumerate(tqdm(loader)):
        input_var = torch.autograd.Variable(input_data, volatile=True).cuda()
        # compute conv features
        if layer=='conv':
            current_features = reglog(forward(input_var, model, reglog.conv)).data.cpu().numpy()
        # compute fc features
        elif layer=='fc':
            current_features = model(input_var).data.cpu().numpy()
        #print current_features.shape
        for j, image_path in enumerate(paths):
            features[image_path] = current_features[j]
    feature_shape = features[list(features.keys())[0]].shape
    #logging.info('Feature shape: %s' % (feature_shape, ))
    #logging.info('Outputting features')

    if sys.version_info >= (3, 0):
        string_type = h5py.special_dtype(vlen=str)
    else:
        string_type = h5py.special_dtype(vlen=unicode)  # noqa
    #paths = features.keys()
    paths = image_paths
    #logging.info('Stacking features')
    features_stacked = np.vstack([features[path] for path in paths])
    ##
    #logging.info('Output feature size: %s' % (features_stacked.shape, ))

    ##dim = 64
    ##pca = KernelPCA(n_components=dim, kernel='cosine')
    ##pca.fit(features_stacked)
    ##reduction_result = pca.transform(features_stacked)
    umap_model = umap.UMAP(n_components=2)
    ##reduction_result = umap_model.fit_transform(reduction_result)
    reduction_result = umap_model.fit_transform(features_stacked)

    return reduction_result

def clustering_dbscan(result, MINPTS, images, class_list, ex_class_list, len_images, ex_len_images, filepath, labels):
    min_eps, e_value = EpsValue(result, MINPTS)
    images = np.array(images)
    class_num = len(class_list)

    best_eps = 0.0
    best_alpha = 0.0
    best_fmeasure = 0.0
    best_nmi = 0.0
    best_ami = 0.0

    data_num = 0
    e = min_eps
    N = len(images)

    while data_num < N:
        e = e + e_value
        db = DBSCAN(eps=e, min_samples=MINPTS).fit(result)
        data_num = len(images[db.labels_>=0])

        sum_max = 0.0
        sum_cluster = 0.0
        sum_inverse = 0.0
        sum_cluster_inverse = 0.0
        sum_cls_num = 0.0

        #I_yc = 0.0
        #H_y = 0.0
        #H_c = 0.0

        if (max(db.labels_) >= 0):
            inv_p = [0.0 for i in range(class_num)]
            for i in range(min(db.labels_), max(db.labels_)+1):
                clusters = [0 for j in range(class_num)]
                for k in range(class_num):
                    cls = len([l for l in images[db.labels_ == i] if filepath+class_list[k] in l])
                    clusters[k] = cls
           #         P_yc = float(cls) / N
           #         P_c = float(len_images[k]) / N
           #         P_y = float(len(images[db.labels_ == i])) / N
           #         if P_yc > 0:
           #             I_yc = I_yc + P_yc * math.log(P_yc/(P_c*P_y), 2)
                    if cls > inv_p[k]:
                        inv_p[k] = float(cls)

                cluster_num = float(len(images[db.labels_ == i]))
           #     H_y = H_y - cluster_num / N * math.log(cluster_num/N, 2)
                sum_max = sum_max + max(clusters)

            #for c in range(len(len_images)):
            #    H_c = H_c - len_images[c] / N * math.log(len_images[c]/N, 2)

            #nmi = I_yc / ((H_y + H_c)/2)
            purity = sum_max / N
            inverse = sum(inv_p) / N

            for i in range(0, max(db.labels_)+1):
                ex_clusters = [0 for j in range(class_num-1)]
                for k in range(class_num-1):
                    ex_cls = len([l for l in images[db.labels_ == i] if filepath+ex_class_list[k] in l])
                    ex_clusters[k] = ex_cls

                ex_cluster_num = float(len(images[db.labels_ == i]))
                sum_cluster = sum_cluster + ex_cluster_num
                sum_max = sum_max + max(ex_clusters)
                ex_cls_purity = float(max(ex_clusters)) / ex_cluster_num
                sum_cls_num = sum_cls_num + sum(ex_clusters)

            if purity > 0 and inverse > 0:
                f_measure = 1/(1/purity*0.5+1/inverse*0.5)
            else:
                f_measure = 0.0
            precision = sum_cls_num / len(images[db.labels_ >= 0])
            recall = sum_cls_num / sum(ex_len_images)
            ami = adjusted_mutual_info_score(labels, db.labels_)
            nmi = normalized_mutual_info_score(labels, db.labels_)

            if nmi > best_nmi:
                best_nmi = nmi
                best_purity = purity
                best_inverse = inverse
                best_f_measure = f_measure
                best_class = float(max(db.labels_)+1)
                best_precision = precision
                best_recall = recall
                best_ami = ami

    print 'purity | inverse purity | F-measure | class num | precision | recall | NMI | AMI'
    print '{0:.3f}'.format(best_purity), '|', '{0:.3f}'.format(best_inverse), '|', '{0:.3f}'.format(best_f_measure), '|', '{0:.3f}'.format(best_class), '|', '{0:.3f}'.format(best_precision), '|', '{0:.3f}'.format(best_recall), '|', '{0:.3f}'.format(best_nmi), '|', '{0:.3f}'.format(best_ami)

    return best_f_measure, best_nmi, best_ami

def main():

    parser = argparse.ArgumentParser(description="""Train linear classifier on top
                                 of frozen convolutional layers of an AlexNet.""")

    #parser.add_argument('--dataset', type=str, help='target dataset')
    parser.add_argument('--model', type=str, help='path to model')
    parser.add_argument('--layer', type=str, help='layer type')
    parser.add_argument('--layer_num', default=2, type=int, help='layer number')
    parser.add_argument('--cls_num', default=1000, type=int, help='model class')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='mini-batch size (default: 32)')
    #parser.add_argument('--seed', type=int, default=31, help='random seed')

    global args

    args = parser.parse_args()




    layer = args.layer
    layer_num = args.layer_num

    # load model
    model = load_model(args.model)

    #param = torch.load(args.model)
    #model.load_state_dict(param)
    #print(model)

    new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    model.classifier = new_classifier

    model.cuda()
    cudnn.benchmark = True
    model.eval()

    #filename = args.dataset

    datasets = glob('/faces_83/evaluation/*')

    F_MEASUREs = [0.0] * len(datasets)
    NMIs = [0.0] * len(datasets)
    AMIs = [0.0] * len(datasets)

    num = 0
    for filepath in datasets:
        filepath = filepath + '/'
        print filepath
        #filepath = '/faces_83/evaluation/' + filename + '/'
        class_list = glob(filepath+'*')
        class_list = [os.path.basename(r) for r in class_list]
        class_num = len(class_list)

        ex_class_list = class_list
        ex_class_list.remove('other')

        len_images = []
        images = []
        labels = []
        n = 0
        for class_name in class_list:
            class_images = glob(filepath+class_name+'/*.jpg')
            images.extend(class_images)
            len_images.append(float(len(class_images)))
            label = [n] * len(class_images)
            labels.extend(label)
            n += 1

        ex_len_images = []
        for class_name in ex_class_list:
            class_images = glob(filepath+class_name+'/*.jpg')
            ex_len_images.append(float(len(class_images)))

        # logistic regression
        reglog = RegLog(layer_num, 10000).cuda()

        seed_f_measure = 0.0
        seed_nmi = 0.0
        seed_ami = 0.0

        for seed in range(10):
            #fix random seeds
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            reduction_result=extract_features_to_disk(images, model, args.batch_size,
                                                      args.workers, reglog, layer)

            f_measure, nmi, ami = clustering_dbscan(reduction_result, 10, images, class_list, ex_class_list, len_images, ex_len_images, filepath, labels)
            seed_f_measure = seed_f_measure + f_measure
            seed_nmi = seed_nmi + nmi
            seed_ami = seed_ami + ami

        seed_f_measure = seed_f_measure / 10
        seed_nmi = seed_nmi / 10
        seed_ami = seed_ami / 10
        F_MEASUREs[num] = seed_f_measure
        NMIs[num] = seed_nmi
        AMIs[num] = seed_ami
        num += 1
        print 'class average f_measure | NMI | AMI'
        print '{0:.3f}'.format(seed_f_measure), '|', '{0:.3f}'.format(seed_nmi), '|', '{0:.3f}'.format(seed_ami)
        print ''

    print 'average f_measure | NMI | AMI'
    print '{0:.3f}'.format(sum(F_MEASUREs)/len(F_MEASUREs)), '|', '{0:.3f}'.format(sum(NMIs)/len(NMIs)), '|', '{0:.3f}'.format(sum(AMIs)/len(AMIs))

if __name__ == '__main__':
    main()
