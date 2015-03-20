import pandas as pd
from os import listdir
from os.path import isfile, join
import scipy as sp
import operator
import csv

mypath = "./Data/pred"
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

resultDF = pd.read_csv('./Data/result.csv')

hash = {}
for file in onlyfiles:
    df = pd.read_csv(mypath + '/' + file)

    p = []
    r = []

    for index, row in resultDF.iterrows():
        for index2, row2 in df.loc[df['id'] == row['id']].iterrows():
            if row['id'] != row2['id']:
                continue
            else:
                p.append(row2['pred'])
                r.append(row['result'])
                break

    hash[file] = llfun(r, p)

sorted_x = sorted(hash.items(), key=operator.itemgetter(1))

index = 1

with open('./readme.md', 'wb') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Please see exact score at [Kaggle March Machine Learning Mania 2015 Leaderboard. ](https://www.kaggle.com/c/march-machine-learning-mania-2015/leaderboard)'])
    writer.writerow(['index|filename|score'])
    writer.writerow(['-----|-----|-----'])
    for row in sorted_x:
        writer.writerow([str(index) + '|' + str(row[0]) + '|' + str(row[1])])
        index = index + 1