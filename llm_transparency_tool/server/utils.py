# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from typing import List, Optional, Tuple

import networkx as nx
import statistics
import streamlit as st
import torch
import transformers


import llm_transparency_tool.routes.graph
from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
from llm_transparency_tool.models.transparent_llm import TransparentLlm

GPU = "gpu"
CPU = "cpu"

# This variable is for expressing the idea that batch_id = 0, but make it more
# readable than just 0.
B0 = 0


def possible_devices() -> List[str]:
    devices = []
    if torch.cuda.is_available():
        devices.append("gpu")
    devices.append("cpu")
    return devices


def load_dataset(filename) -> List[str]:
    with open(filename) as f:
        dataset = [s.strip("\n") for s in f.readlines()]
    print(f"Loaded {len(dataset)} sentences from {filename}")
    return dataset


@st.cache_resource(
    hash_funcs={
        TransformerLensTransparentLlm: id
    }
)
def load_model(
    model_name: str,
    _device: str,
    _model_path: Optional[str] = None,
    _dtype: torch.dtype = torch.float32,
    supported_model_name: Optional[str] = None,
) -> TransparentLlm:
    """
    Returns the loaded model along with its key. The key is just a unique string which
    can be used later to identify if the model has changed.
    """
    assert _device in possible_devices()


    # have adjusted to now allow locally stored models, there is a downside though
    # this will cause other problems later on if the model isn't supported by TransformerLens or HookedTransformer
    if _model_path:
        causal_lm = transformers.AutoModelForCausalLM.from_pretrained(_model_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(_model_path)
    else:
        causal_lm = None
        tokenizer = None

    

    tl_lm = TransformerLensTransparentLlm(
        model_name=model_name,
        hf_model=causal_lm,
        tokenizer=tokenizer,
        device=_device,
        dtype=_dtype,
        supported_model_name=supported_model_name,
    )

    return tl_lm


def run_model(model: TransparentLlm, sentence: str) -> None:
    print(f"Running inference for '{sentence}'")
    model.run([sentence])


def load_model_with_session_caching(
    **kwargs,
) -> Tuple[TransparentLlm, str]:
    return load_model(**kwargs)

def run_model_with_session_caching(
    _model: TransparentLlm,
    model_key: str,
    sentence: str,
):
    LAST_RUN_MODEL_KEY = "last_run_model_key"
    LAST_RUN_SENTENCE = "last_run_sentence"
    state = st.session_state

    if (
        state.get(LAST_RUN_MODEL_KEY, None) == model_key
        and state.get(LAST_RUN_SENTENCE, None) == sentence
    ):
        return

    run_model(_model, sentence)
    state[LAST_RUN_MODEL_KEY] = model_key
    state[LAST_RUN_SENTENCE] = sentence


@st.cache_resource(
    hash_funcs={
        TransformerLensTransparentLlm: id
    }
)
def get_contribution_graph(
    model: TransparentLlm,  # TODO bug here
    model_key: str,
    tokens: List[str],
    threshold: float,
) -> nx.Graph:
    """
    The `model_key` and `tokens` are used only for caching. The model itself is not
    hashed, hence the `_` in the beginning.
    """
    return llm_transparency_tool.routes.graph.build_full_graph(
        model,
        B0,
        threshold,
    )

@st.cache_resource(
    hash_funcs={
        TransformerLensTransparentLlm: id
    }
)
def get_contribution_graph_contrast(
    base_model: TransparentLlm,
    contrast_model: TransparentLlm,
    model_key: str,
    tokens: List[str],    
    threshold: float,    
) -> nx.Graph:
    """Get the graph by using the contrast of the two models.
    
    Use object id for models, and added model_key and tokens for hashing purposes

    Args:
        base_model (TransparentLlm): Model 1, the one to be contrast
        contrast_model (TransparentLlm): Model 2, the one to compare
        threshold (float): Threshold to keep the edge.

    Returns:
        nx.Graph: Resulting graph.
    """
    
    return llm_transparency_tool.routes.graph.build_full_graph_with_contrast(
        base_model,
        contrast_model,
        B0,
        threshold,
    ) 


def st_placeholder(
    text: str,
    container=st,
    border: bool = True,
    height: Optional[int] = 500,
):
    empty = container.empty()
    empty.container(border=border, height=height).write(f'<small>{text}</small>', unsafe_allow_html=True)
    return empty


def contrast_graphs(graph1: nx.Graph, graph2: nx.Graph) -> nx.Graph:
    """
    No longer used, this was my attempt at implementing Hector's method, I'm now just using his original
    """

    
    def median_edge_weight(graph: nx.Graph) -> float:
        # Extract all edge weights into a list
        weights = [data["weight"] for _, _, data in graph.edges(data=True)]
        
        # Calculate and return the median of the edge weights
        return statistics.median(weights) if weights else 0.0

    def find_attention_edge_difference(graph1: nx.Graph, graph2: nx.Graph) -> nx.Graph:
        """
        Finds the difference in attention edges between two graphs, focusing only on edges
        involving attention nodes ('A' nodes). If an edge exists, its weight is adjusted,
        otherwise it is created with the appropriate weight.
        
        graph1: The first graph (e.g., the reference or base graph).
        graph2: The second graph (e.g., the graph to compare against the first).
        
        Returns:
            A new graph with the same nodes as the input graphs and edges that represent
            the difference in attention weights between the two graphs, but only for edges
            involving attention nodes.
        """
        # Initialize a new directed graph for the difference
        diff_graph = nx.DiGraph()
        
        # Add all nodes from graph1 (and graph2, since they should have the same nodes)
        diff_graph.add_nodes_from(graph1.nodes())
        
        # Function to determine if a node is an attention node
        def is_attention_or_resid_node(node):
            return (node.startswith("A") or node.startswith("I"))

        # Iterate over edges in graph1 and calculate the difference with graph2
        for u, v, data in graph1.edges(data=True):
            # Only consider edges involving attention nodes
            if is_attention_or_resid_node(u) and is_attention_or_resid_node(v):
                weight1 = data["weight"]
                if graph2.has_edge(u, v):
                    weight2 = graph2[u][v]["weight"]
                    weight_diff = weight1 - weight2
                    if weight_diff >= 0:
                        # If edge exists in diff_graph, update the weight
                        if diff_graph.has_edge(u, v):
                            diff_graph[u][v]["weight"] += weight_diff
                        else:
                            diff_graph.add_edge(u, v, weight=weight_diff)
                else:
                    # Edge only in graph1
                    if diff_graph.has_edge(u, v):
                        diff_graph[u][v]["weight"] += weight1 # was previously weight1 / was 0
                    else:
                        diff_graph.add_edge(u, v, weight=weight1) # was previously weight1 / was 0 
        
        # Iterate over edges in graph2 that are not in graph1
        for u, v, data in graph2.edges(data=True):
            # Only consider edges involving attention nodes
            if is_attention_or_resid_node(u) and is_attention_or_resid_node(v):
                if not graph1.has_edge(u, v):
                    weight2 = data["weight"]
                    # Add the edge with negative weight to indicate it was only in graph2
                    if diff_graph.has_edge(u, v):
                        diff_graph[u][v]["weight"] -= weight2
                    else:
                        diff_graph.add_edge(u, v, weight=-weight2)
        
        return diff_graph

    diff_graph = find_attention_edge_difference(graph1, graph2)
    print(f"meidan edge weight: {median_edge_weight(diff_graph)}")
    return diff_graph
