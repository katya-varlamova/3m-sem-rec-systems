import pandas as pd
from libreco.data import random_split, DatasetPure
from libreco.algorithms import UserCF, ItemCF
from libreco.evaluation import evaluate
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

def ucf(sim_type, k_sim):
    userCf = UserCF(
        "ranking",
        dataInfo,
        sim_type=sim_type,
        k_sim=k_sim
    )
    start_time = time.time()
    userCf.fit(
        trainData,
        neg_sampling=True,
        verbose=2,
        eval_data=evalData,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"]
    )
    elapsed_time = time.time() - start_time

    evaluation = evaluate(
        model=userCf,
        data=testData,
        neg_sampling=True,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"]
    )
    return evaluation, elapsed_time, userCf

def icf(sim_type, k_sim):
    itemCf = ItemCF(
        "ranking",
        dataInfo,
        sim_type=sim_type,
        k_sim=k_sim
    )
    start_time = time.time()
    itemCf.fit(
        trainData,
        neg_sampling=True,
        verbose=2,
        eval_data=evalData,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"]
    )
    elapsed_time = time.time() - start_time

    evaluation = evaluate(
        model=itemCf,
        data=testData,
        neg_sampling=True,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"]
    )
    return evaluation, elapsed_time, itemCf

def compare_algos():
    num_similar_items = []
    itemcf_times = defaultdict(list)
    itemcf_roc_auc = defaultdict(list)
    usercf_times = defaultdict(list)
    usercf_roc_auc = defaultdict(list)
    metrics = ['cosine', 'pearson', 'jaccard']
    for k_sim in range(1, 31):
        num_similar_items.append(k_sim)
        for st in metrics:
            evaluation, elapsed_time, _ = ucf(st, k_sim)
            usercf_times[st].append(elapsed_time)
            usercf_roc_auc[st].append(evaluation["roc_auc"])
            evaluation, elapsed_time, _ = icf(st, k_sim)
            itemcf_times[st].append(elapsed_time)
            itemcf_roc_auc[st].append(evaluation["roc_auc"])

    plt.figure(figsize=(12, 6))


    plt.subplot(1, 2, 1)
    for metric in metrics:
        plt.plot(num_similar_items, itemcf_times[metric], label=metric)
    plt.title('ItemCF: Время обучения')
    plt.xlabel('Количество похожих объектов')
    plt.ylabel('Время обучения (с)')
    plt.legend()
    plt.grid()

    # UserCF
    plt.subplot(1, 2, 2)
    for metric in metrics:
        plt.plot(num_similar_items, usercf_times[metric], label=metric)
    plt.title('UserCF: Время обучения')
    plt.xlabel('Количество похожих объектов')
    plt.ylabel('Время обучения (с)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("time.png")

    # График 2: ROC AUC
    plt.figure(figsize=(12, 6))

    # ItemCF
    plt.subplot(1, 2, 1)
    for metric in metrics:
        plt.plot(num_similar_items, itemcf_roc_auc[metric], label=metric)
    plt.title('ItemCF: ROC AUC')
    plt.xlabel('Количество похожих объектов')
    plt.ylabel('ROC AUC')
    plt.legend()
    plt.grid()

    # UserCF
    plt.subplot(1, 2, 2)
    for metric in metrics:
        plt.plot(num_similar_items, usercf_roc_auc[metric], label=metric)
    plt.title('UserCF: ROC AUC')
    plt.xlabel('Количество похожих объектов')
    plt.ylabel('ROC AUC')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("roc_auc.png")
    
def make_predictions():
    _, _, ucfModel = ucf('cosine', 20)
    _, _, icfModel = icf('cosine', 20)

    print(ucfModel.recommend_user(1425, 30))
    print(icfModel.recommend_user(1425, 30))
    
data = pd.read_csv("sample_movielens_rating.dat", sep="::", names=["user", "item", "label", "time"])
data.drop('time', axis=1, inplace=True)
trainData, evalData, testData= random_split(data, multi_ratios=[0.8, 0.1, 0.1])

trainData, dataInfo = DatasetPure.build_trainset(trainData)
evalData = DatasetPure.build_evalset(evalData)
testData = DatasetPure.build_testset(testData)


