import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from libreco.data import random_split, DatasetPure
from libreco.evaluation import evaluate
from libreco.algorithms import UserCF, ItemCF
import matplotlib.pyplot as plt
import time
import warnings
from collections import defaultdict
from sklearn.metrics import roc_auc_score, precision_score, recall_score
warnings.filterwarnings("ignore")

def ucf(sim_type, k_sim):
    start_time = time.time()
    userCf = UserCF(
        "ranking",
        dataInfo,
        sim_type=sim_type,
        k_sim=k_sim
    )
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

def make_prediction(user, n):
    _, _, ucfModel = ucf('cosine', 20)
    return ucfModel.recommend_user(user, n)[user]

class CollaborativeFiltering:
    def __init__(self, n_factors=10):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}
        self.item_mapping = {}

    def fit(self, trainData):
        users = trainData['user'].unique()
        items = trainData['item'].unique()

        self.user_mapping = {user: idx for idx, user in enumerate(users)}
        self.item_mapping = {item: idx for idx, item in enumerate(items)}

        n_users = len(users)
        n_items = len(items)

        interaction_matrix = np.zeros((n_users, n_items))

        for _, row in trainData.iterrows():
            user_idx = self.user_mapping[row['user']]
            item_idx = self.item_mapping[row['item']]
            interaction_matrix[user_idx, item_idx] = row['label']

        svd = TruncatedSVD(n_components=self.n_factors)
        self.user_factors = svd.fit_transform(interaction_matrix)
        self.item_factors = svd.components_.T

    def predict(self, user_idx, item_idx):
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])

    def evaluate(self, testData):
        y_true = []
        y_pred = []

        for _, row in testData.iterrows():
            user = row['user']
            item = row['item']
            actual_rating = row['label']

            if user in self.user_mapping and item in self.item_mapping:
                user_idx = self.user_mapping[user]
                item_idx = self.item_mapping[item]

                predicted_rating = self.predict(user_idx, item_idx)

                y_true.append(actual_rating)
                y_pred.append(predicted_rating)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        loss = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error

        result = {
            'loss': loss
        }

        return result
    def get_by_value(self, d, val):
        return list(d.keys())[list(d.values()).index(val)]
    
    def recommend(self, user_id, n=10):
        user_idx = self.user_mapping[user_id]
        predicted_ratings = np.dot(self.user_factors[user_idx, :], self.item_factors.T)
        recommended_items_idx = np.argsort(predicted_ratings)[-n:][::-1]
        return [self.get_by_value(self.item_mapping, idx) for idx in recommended_items_idx]

def convert_to_pandas(train_data):
    user_ids = []
    item_ids = []
    ratings = []
    for user, item, rating in train_data:
        user_ids.append(user)
        item_ids.append(item)
        ratings.append(rating)

    df = pd.DataFrame({
        'user': user_ids,
        'item': item_ids,
        'label': ratings
    })
    return df


ratings = pd.read_csv("sample_movielens_rating.dat", sep="::", names=["user", "item", "label", "time"])
ratings.drop('time', axis=1, inplace=True)
trainData, evalData, testData= random_split(ratings, multi_ratios=[0.8, 0.1, 0.1])

trainData, dataInfo = DatasetPure.build_trainset(trainData)
evalData = DatasetPure.build_evalset(evalData)
testData = DatasetPure.build_testset(testData)



##true_items = make_prediction(1425, 500)
##precision, recall = evaluate_recommendations(user_id, true_items, cf_model, n=500)
##print(f'Precision: {precision}, Recall: {recall}')
##
##user_id = 1425  # Замените нужным id пользователя
##recommended_items = cf_model.recommend(user_id, n=500)
##print("Рекомендованные товары для пользователя:", recommended_items)

n_factors_list = range(5, 100, 5)
results = {}

evaluation, elapsed_time, _ = ucf('cosine', 100)

eval_self = []
eval_lib = [evaluation["loss"]] * len(n_factors_list)
time_self = []
time_lib = [elapsed_time] * len(n_factors_list)
for n in n_factors_list:
    pd_train = convert_to_pandas(trainData)
    start_time = time.time()
    cf_model = CollaborativeFiltering(n_factors=n)
    cf_model.fit(pd_train)
    elapsed_time_self = time.time() - start_time
    evaluation_self = cf_model.evaluate(convert_to_pandas(testData))

    eval_self.append(evaluation_self["loss"])
    time_self.append(elapsed_time_self)
    print(elapsed_time_self)
    

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(n_factors_list, eval_lib, label='UCF')
plt.plot(n_factors_list, eval_self, label='my UCF + matrix factorization')
plt.xlabel('Number of factors')
plt.ylabel('MSE')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_factors_list, time_lib, label='UCF')
plt.plot(n_factors_list, time_self, label='my UCF + matrix factorization')
plt.xlabel('Number of factors')
plt.ylabel('Time (s)')
plt.title('Fit Time')
plt.legend()

plt.tight_layout()
plt.savefig("res.png")
plt.show()
    


