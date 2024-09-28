import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
from pyECLAT import ECLAT
from fpgrowth_py import fpgrowth
import numpy as np
import json
import time
import memory_profiler as mp
import pandas as pd
import multiprocessing

ctx = multiprocessing.get_context('spawn')
if __name__ == '__main__':
##    dataset = pd.read_csv("short.csv", header=None) # Market_Basket_Optimisation
##
##    transactions = []
##    for i in range(0, dataset.shape[0]):
##        transactions.append([str(dataset.values[i, j]) for j in range(0, dataset.shape[1])])
##
##    with open("transactions", "w") as fp:
##        json.dump(transactions, fp)

    transactions = []
    with open("transactions", "r") as fp:
        transactions = json.load(fp)

    times = [[], [], []]
    transesForFp = [list(filter(lambda x: x != 'nan', it)) for it in transactions]
    df = pd.DataFrame(transesForFp)
    print(df.stack().nunique())
    eclat = ECLAT(pd.DataFrame(transesForFp))
    
    def daHeck(results):
        firstProd = [tuple(result[2][0][0])[0] for result in results]
        secondProd = [tuple(result[2][0][1])[0] for result in results]
        supports = [result[1] for result in results]
        confidences = [result[2][0][2] for result in results]
        lifts = [result[2][0][3] for result in results]
        return list(zip(firstProd, secondProd, supports, confidences, lifts))


    x = np.concatenate((np.arange(0.001, 0.01, 0.001), np.arange(0.01, 0.2, 0.01)))
    for i in x:
        print(i)
        start = time.time()
        res = apriori(transactions=transesForFp, min_support=i, min_confidence=0.2, min_lift=3, min_length=2, verbose=False
                )
##        results = pd.DataFrame(daHeck(list(res)), columns=["Rule start", "Rule end", "Support", "Confidence", "Lift"])
##        print(results.nlargest(columns = "Support", n=100))
        end = time.time() - start
        times[0].append(end)


        start = time.time()
        ruleIndices, ruleSupports = eclat.fit(min_support=i, min_combination=2, max_combination=2, verbose=False)
##        result = pd.DataFrame(ruleSupports.items(), columns=['Item', 'Support'])
##        result.sort_values(by=['Support'], ascending=False)
##        print(result)
        end = time.time() - start
        times[1].append(end)

        start = time.time()
        freqList, rules = fpgrowth(transesForFp, minSupRatio=i, minConf=0.2)
##        print(rules)
        end = time.time() - start
        times[2].append(end)

    plt.plot(x, times[0], label="Apriori")
    plt.plot(x, times[1], label="ECLAT")
    plt.plot(x, times[2], label="FP-Growth")
    plt.legend()
    plt.ylabel("Время исполнения, сек")
    plt.xlabel("Значение параметра минимальной поддержки")
    plt.savefig("time.png")
    plt.clf()

    memoryUsage = [[], [], []]

    for i in x:
        print(i)
        memoryUsage[0].append(
            mp.memory_usage((apriori, (transactions,), { 'verbose': False, 'min_support':i, 'min_confidence':0.2, 'min_lift':3, 'min_length':2 }), max_usage=True)
        )

        memoryUsage[1].append(
            mp.memory_usage((eclat.fit, (), {'verbose': False, 'min_support':i, 'min_combination':2, 'max_combination':2}), max_usage=True)
        )

        memoryUsage[2].append(
            mp.memory_usage((fpgrowth, (transesForFp, i, 0.2)), max_usage=True)
        )


    plt.plot(x, memoryUsage[0], label="Apriori")
    plt.plot(x, memoryUsage[1], label="ECLAT")
    plt.plot(x, memoryUsage[2], label="FP-Growth")
    plt.legend()
    plt.ylabel("Максимально занятая память процессом, MiB")
    plt.xlabel("Значение параметра минимальной поддержки")
    plt.savefig("mem.png")
