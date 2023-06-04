import sys
sys.path.append("aste")
from pathlib import Path
from data_utils import Data, Sentence, SplitEnum
from wrapper import SpanModel

#@title K fold cross validation con dataset movil
import json
import statistics
import time
import random
random_seed = random.randint(0, 100)
path_corpus = "aste/data/corpus.txt"
path_dev = "aste/data/k_cross/dev.txt"
path_train = "aste/data/k_cross/train.txt"
path_test = "aste/data/k_cross/test.txt"
save_dir = f"outputs/k_cross/seed_{random_seed}"
file_results = open("results.txt", 'w')
k = 5
results = []

with open(path_corpus, 'r') as fp:
  corpus = [line.rstrip() for line in fp]
  corpus_size = len(corpus)
  fold_size = corpus_size//k
  print(fold_size)
  for fold in range(0, k-1):
    save_dir = f"outputs/k_cross/seed_{random_seed}_fold_{fold}"
    path_dev = f"aste/data/k_cross/fold_{fold}/dev.txt"
    path_train = f"aste/data/k_cross/fold_{fold}/train.txt"
    path_test = f"aste/data/k_cross/fold_{fold}/test.txt"
    train_file = open(path_train, 'w')
    test_file = open(path_test, 'w')
    dev_file = open(path_dev, 'w')
    for i in range(fold*fold_size, (fold+1)*fold_size):
      test_file.write(corpus[i])
      test_file.write('\n')
    for i in range((fold+1)*fold_size, (fold+2)*fold_size):
      dev_file.write(corpus[i])
      dev_file.write('\n')
    for i in range(0, fold*fold_size):
      train_file.write(corpus[i])
      train_file.write('\n')
    for i in range((fold+2)*fold_size, corpus_size):
      train_file.write(corpus[i])
      train_file.write('\n')
    train_file.close()
    test_file.close()
    dev_file.close()

    model = SpanModel(save_dir=save_dir, random_seed=random_seed)
    model.fit(path_train, path_dev)

    path_pred = "pred.txt"
    model.predict(path_in=path_test, path_out=path_pred)
    result = model.score(path_pred, path_test)
    print(result)
    file_results.write(result)
    print('-'*60)
    print(results)
    results.append(result)
    print(f"Modelo entrenado fold: {fold}")

file_results.close()
precisions = [result['precision'] for result in results]
recalls = [result['recall'] for result in results]
f1_scores = [result['score'] for result in results]
print("Precisions:")
print(precisions)
print(f"Precision promedio = {statistics.mean(precisions)}")
print("Recalls:")
print(recalls)
print(f"Recall promedio = {statistics.mean(recalls)}")
print("f1 scores:")
print(f1_scores)
print(f"F1 Score promedio = {statistics.mean(f1_scores)}")

file_stats = open("stats.txt", 'w')
file_stats.write("Precisions:\n")
file_stats.write(str(precisions))
file_stats.write("\n")
file_stats.write(f"Precision promedio = {statistics.mean(precisions)}")
file_stats.write("\n")
file_stats.write("Recalls:\n")
file_stats.write(str(recalls))
file_stats.write("\n")
file_stats.write(f"Recall promedio = {statistics.mean(recalls)}")
file_stats.write("\n")
file_stats.write("f1 scores:\n")
file_stats.write(str(f1_scores))
file_stats.write("\n")
file_stats.write(f"F1 Score promedio = {statistics.mean(f1_scores)}")
file_stats.write("\n")
file_stats.close()
