from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
import llm_transparency_tool.routes.graph as lmttg


# Empirically, single sentence is around 0.02, contrast setences threshold is around 0.01

def build_graph(model, sentence, threshold=0.02):
    model.run(sentence)

    graph = lmttg.build_full_graph(
        model,
        renormalizing_threshold=threshold,
    )
    
    return graph

sentences = [
    "In the hotel laundry room, Emma burned Mary's shirt while ironing it, so the manager scolded",
    "In the hotel laundry room, Emma burned Mary's shirt while ironing it, so the manager refunded",
]

print("Build graph for sentence")

model = TransformerLensTransparentLlm("LLM360/amber")

graph = build_graph(model, sentences[0])

import pdb
pdb.set_trace()