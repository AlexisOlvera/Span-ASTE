# Use pretrained SpanModel weights for prediction
import sys
from pathlib import Path
from aste.data_utils import Data, Sentence, SplitEnum
from aste.wrapper import SpanModel

def predict_sentence(text: str, model: SpanModel) -> Sentence:
    path_in = "temp_in.txt"
    path_out = "temp_out.txt"
    sent = Sentence(tokens=text.split(), triples=[], pos=[], is_labeled=False, weight=1, id=0)
    data = Data(root=Path(), data_split=SplitEnum.test, sentences=[sent])
    data.save_to_path(path_in)
    model.predict(path_in, path_out)
    data = Data.load_from_full_path(path_out)
    return data.sentences[0]

model_dir = '/content/outputs/resESP/seed_17'
text = "No me gusto el pan y los asientos eran incomodos "
model = SpanModel(save_dir=model_dir, random_seed=0)
sent = predict_sentence(text, model)

for t in sent.triples:
    target = " ".join(sent.tokens[t.t_start:t.t_end+1])
    opinion = " ".join(sent.tokens[t.o_start:t.o_end+1])
    print()
    print(dict(target=target, opinion=opinion, sentiment=t.label))