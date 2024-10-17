import sys
import json
import pdb

from transformers import LlamaTokenizer
from openai import OpenAI

tokenizer = None

def token_count(sentence: str):
    global tokenizer
    if not tokenizer:
        tokenizer = LlamaTokenizer.from_pretrained("LLM360/Amber")    
    return len(tokenizer(sentence, return_tensors="pt")["input_ids"])


def diff_count():
    pass

def blank2end(sentence: str):
    pass

def rename_entities(sentence: str):
    pass

def fix_sent(sent: str):
    # Count tokens.
    n_tokens = token_count(sent)

    # Rephrase blank to end.
    sent_rephrased = blank2end(sent)

    # Rename entities.
    sent_group = rename_entities(sent_rephrased)


def create_wino_pairs(sent1: str, sent2: str):
    count = 0
    sample = None
    for qid, l_data in sent_pairs.items():
        if len(l_data) > 1:
            count += 1

            # Count tokens before modifications.
            n_tokens1 = token_count(sample[0]["sentence"])
            n_tokens2 = token_count(sample[1]["sentence"])

            # Rephrase blank to end.
            sent1_rephrased = blank2end(sample[0]["sentence"])

            # Rename entities.
            sent1_group = rename_entities(sent1_rephrased)

        
        pdb.set_trace()



def load_pairs(data_path: str):
    sent_pairs = {}

    with open(data_path) as din:
        for line in din.readlines():
            wino_data = json.loads(line)
            qid, index = wino_data["qID"].split("-")

            sentence = wino_data["sentence"]
            option1 = wino_data["option1"]
            option2 = wino_data["option2"]
                        
            answer, wrong_answer = (option1, option2) if wino_data["answer"] == "1" else (option2, option1)

            d = {
                "sentence": sentence,
                "answer": answer,
                "wrong_answer": wrong_answer, 
            }

            try:
                sent_pairs[qid].append(d)
            except KeyError:
                sent_pairs[qid] = [d]

    return sent_pairs


if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    sent_pairs = load_pairs(input_file)

    count = 0
    for qid, l_data in sent_pairs.items():
        if len(l_data) > 1:
            count += 1

            fix_sent(l_data[0]["sentence"])

