import re
from nltk import regexp_tokenize
# Module for accent delete
from unicodedata import normalize
import sys
sys.path.append("aste")
from pathlib import Path
from data_utils import Data, Sentence, SplitEnum
from wrapper import SpanModel


class ModelPredict:
    def predict_sentence(self, text: str, model: SpanModel) -> Sentence:
        path_in = "temp_in.txt"
        path_out = "temp_out.txt"
        sent = Sentence(tokens=text.split(), triples=[], pos=[], is_labeled=False, weight=1, id=0)
        data = Data(root=Path(), data_split=SplitEnum.test, sentences=[sent])
        data.save_to_path(path_in)
        model.predict(path_in, path_out)
        data = Data.load_from_full_path(path_out)
        return data.sentences[0]


    def accents(self, doc):
        # -> NFD
        doc_str = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1",
                normalize( "NFD", doc), 0, re.I)

        # -> NFC
        doc_str = normalize( 'NFC', doc_str)
        return doc_str

    def replace_wrd(self, text):
        # no emojis
        text = text.replace(u"[\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF]",'')
        text = text.replace('must', 'recomendado')
        return text

    def clean_tokens(self, text):
        text = text.lower()      # minus
        text = self.accents(self, text)     # no acentos
        text = self.replace_wrd(self, text) # reemplazo de palabras

        # Expresión regular para tokenización

        pattern = r'''(?x)                 # set flag to allow verbose regexps
                    [$.!?¿¡$,]             # signos de puntuación
                    | \w+(?:-\w+)*
                    | [0-9]+
        '''
        # NLTK tokenization module
        return regexp_tokenize(text, pattern)


    def preprocesar(self, text):
        return "RESTAURANTE " + ' '.join(self.clean_tokens(self, text)) + " RESTAURANTE"

    @classmethod
    def predecir(self, text):
        model_dir = 'monchi/seed_17'
        model = SpanModel(save_dir=model_dir, random_seed=0)
        text = self.preprocesar(self, text)
        sent = self.predict_sentence(self, text, model)
        res = []
        for t in sent.triples:
            target = " ".join(sent.tokens[t.t_start:t.t_end+1])
            opinion = " ".join(sent.tokens[t.o_start:t.o_end+1])
            res.append(dict(
                aspect=target,
                opinion=opinion,
                sentiment=t.label,
                positions=dict(
                aspect=[t.t_start, t.t_end],
                opinion=[t.o_start, t.o_end]
                )))

        return dict(review = text, triplets = res)
