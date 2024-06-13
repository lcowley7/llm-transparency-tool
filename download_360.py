
from transformers import LlamaTokenizer, LlamaForCausalLM

import sys

model_name = sys.argv[1]
revision = sys.argv[2]

if not model_name.startswith("LLM360")
    raise Exception("Not an LLM360 model")

tokenizer = LlamaTokenizer.from_pretrained("LLM360/Amber", revision="ckpt_356")
model = LlamaForCausalLM.from_pretrained("LLM360/Amber", revision="ckpt_356")
