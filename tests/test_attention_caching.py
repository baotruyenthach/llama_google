import fire
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # Add the parent directory of the current file to sys.path
from llama import Llama
from typing import List
import timeit
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

"""
Script to visualize attention maps for a given input text using the LLAMA model.
"""

def plot_attention_maps(attentions, batch_idx, layer_indices, head=0, show_fig=False, fig_save_path=None):
    fig, axes = plt.subplots(1, len(layer_indices), figsize=(20, 5))
    for idx, layer_idx in enumerate(layer_indices):
        attention = attentions[layer_idx][batch_idx][head][0:-1,0:-1]
        ax = axes[idx]
        im = ax.imshow(attention, cmap='coolwarm', vmin=0, vmax=1)  # Set vmax to 1
        ax.set_title(f'Layer {layer_idx} Head {head}')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Sequence Position')
        
        # Set integer ticks for both axes
        ax.set_xticks(np.arange(attention.shape[-1]))
        ax.set_yticks(np.arange(attention.shape[-2]))
        
        # fig.colorbar(im, ax=ax)

    plt.tight_layout()
    if show_fig:
        plt.show()
    if fig_save_path is not None:
        plt.savefig(fig_save_path)


def plot_attention_maps_2(attentions, batch_idx, positions, heads, show_fig=False, fig_save_path=None):
    num_heads = len(heads)
    fig, axes = plt.subplots(len(positions), 1, figsize=(20, len(positions) * 4))
    if len(positions) == 1:
        axes = [axes]  # Ensure axes is iterable

    for idx, pos in enumerate(positions):
        if pos >= attentions[-1].shape[-1]:
            raise ValueError(f"Position {pos} is out of range for the input sequence.")
        
        attention = np.stack([attentions[-1][batch_idx][head, pos, 1:] for head in heads])  # Select attention for the specific heads and position, excluding the first token
        threshold = 1 / pos  # Define the threshold for significant attention scores
        attention = np.where(attention < threshold, attention, 1)    
        
        ax = axes[idx]
        im = ax.imshow(attention, cmap='Greens', aspect='auto')
        ax.set_title(f'Attention map at position {pos}')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Attention Head')
        
        # Set integer ticks for x-axis
        ax.set_xticks(np.arange(0, attention.shape[1], step=10))
        ax.set_xticklabels(np.arange(1, attention.shape[1] + 1, step=10))
        
        # Set integer ticks for y-axis
        ax.set_yticks(np.arange(num_heads))
        ax.set_yticklabels(heads)

        # fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    if show_fig:
        plt.show()
    if fig_save_path is not None:
        plt.savefig(fig_save_path)


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


    with open(os.path.join(script_dir, 'input_text.txt'), 'r') as file:
        prompts = [line.strip() for line in file.readlines()] 

    start_time = timeit.default_timer()
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        output_attentions=True,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

    print("Time taken by generator: ", timeit.default_timer() - start_time)

    attentions = generator.model.get_attention_matrices()
    if attentions is not None:
        batch_idx = 0

        layer_indices = [0, 9, 16, 23, 30]
        plot_attention_maps(attentions, batch_idx, layer_indices, head=10, fig_save_path=os.path.join(script_dir, 'attention_maps_1.png'))

        heads = [0, 1, 2, 3, 4]
        positions = np.arange(1, attentions[0].shape[-1]-1, 50)
        plot_attention_maps_2(attentions, batch_idx, positions, heads, fig_save_path=os.path.join(script_dir, 'attention_maps_2.png'))

    print("Time taken by generator + matplotlib: ", timeit.default_timer() - start_time)

if __name__ == "__main__":
    fire.Fire(main)
