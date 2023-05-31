import sys
sys.path.append("aste")
from pathlib import Path
from data_utils import Data, Sentence, SplitEnum
from wrapper import SpanModel
import json

random_seed = 17
data_name = "monchi"
path_train = f"aste/data/corpus.txt"
path_dev = f"aste/data/corpus_dev.txt"
path_test = f"aste/data/corpus_dev.txt"
save_dir = f"outputs/{data_name}/seed_{random_seed}"

model = SpanModel(save_dir=save_dir, random_seed=random_seed)
model.fit(path_train, path_dev)

path_pred = "pred.txt"
model.predict(path_in=path_test, path_out=path_pred)
results = model.score(path_pred, path_test)
print(json.dumps(results, indent=2))
