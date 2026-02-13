#!/usr/bin/env python3
"""
Simple runner script for refusal direction transfer analysis

Usage:
    python run_analysis.py --source llama2 --target vicuna
    python run_analysis.py --source llama2 --target vicuna --use-gcg --layers 10 20
    python run_analysis.py --help
"""

import argparse
import sys
from pathlib import Path

# Model path mappings
MODEL_PATHS = {
    'llama2': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'vicuna': 'lmsys/vicuna-7b-v1.5',
    'vicuna-13b': 'lmsys/vicuna-13b-v1.5',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
    'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
}


def main():
    parser = argparse.ArgumentParser(
        description='Analyze refusal direction transfer between models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=list(MODEL_PATHS.keys()),
        help='Source model to extract refusal direction from'
    )

    parser.add_argument(
        '--target',
        type=str,
        required=True,
        choices=list(MODEL_PATHS.keys()),
        help='Target model to transfer refusal direction to'
    )

    parser.add_argument(
        '--source-path',
        type=str,
        default=None,
        help='Custom path for source model (overrides --source)'
    )

    parser.add_argument(
        '--target-path',
        type=str,
        default=None,
        help='Custom path for target model (overrides --target)'
    )

    # Data arguments
    parser.add_argument(
        '--num-wikitext',
        type=int,
        default=520,
        help='Number of WikiText examples to use'
    )

    parser.add_argument(
        '--num-harmbench',
        type=int,
        default=520,
        help='Number of HarmBench examples to use'
    )

    parser.add_argument(
        '--use-gcg',
        action='store_true',
        help='Include GCG-attacked prompts in analysis'
    )

    parser.add_argument(
        '--num-gcg',
        type=int,
        default=100,
        help='Number of GCG examples to use'
    )

    parser.add_argument(
        '--gcg-path',
        type=str,
        default='../outputs/advbench_suffixes_all_models_fixed.csv',
        help='Path to GCG prompts CSV'
    )

    # Layer arguments
    parser.add_argument(
        '--layers',
        type=int,
        nargs=2,
        metavar=('START', 'END'),
        default=None,
        help='Layer range to analyze (e.g., --layers 10 25)'
    )

    parser.add_argument(
        '--specific-layers',
        type=int,
        nargs='+',
        metavar='LAYER',
        default=None,
        help='Specific layers to analyze (e.g., --specific-layers 10 15 20)'
    )

    # Processing arguments
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for embedding extraction'
    )

    parser.add_argument(
        '--position',
        type=str,
        default='last',
        choices=['last', 'first', 'mean'],
        help='Token position to use for embeddings'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./refusal_analysis_output',
        help='Directory to save results'
    )

    parser.add_argument(
        '--save-embeddings',
        action='store_true',
        help='Save extracted embeddings to disk'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )

    # Method arguments
    parser.add_argument(
        '--direction-method',
        type=str,
        default='mean_diff',
        choices=['mean_diff', 'pca', 'contrastive'],
        help='Method for computing refusal direction'
    )

    parser.add_argument(
        '--alignment-method',
        type=str,
        default='procrustes',
        choices=['procrustes', 'cca', 'both'],
        help='Method for aligning embedding spaces'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.source == args.target:
        print("Warning: Source and target are the same model. This is mainly useful for sanity checking.")

    if args.use_gcg and not Path(args.gcg_path).exists():
        print(f"Error: GCG path not found: {args.gcg_path}")
        sys.exit(1)

    if args.layers and args.specific_layers:
        print("Error: Cannot specify both --layers and --specific-layers")
        sys.exit(1)

    # Import here to avoid loading transformers if just showing help
    from refusal_direction_transfer import (
        ModelConfig, ExperimentConfig, EmbeddingExtractor,
        load_wikitext, load_harmbench, load_gcg_prompts,
        analyze_layer_wise_transfer, plot_analysis_results
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Set up configs
    source_path = args.source_path if args.source_path else MODEL_PATHS[args.source]
    target_path = args.target_path if args.target_path else MODEL_PATHS[args.target]

    source_config = ModelConfig(name=args.source, path=source_path)
    target_config = ModelConfig(name=args.target, path=target_path)

    config = ExperimentConfig(
        num_wikitext=args.num_wikitext,
        num_harmbench=args.num_harmbench,
        num_gcg=args.num_gcg,
        use_gcg=args.use_gcg,
        gcg_path=args.gcg_path,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_embeddings=args.save_embeddings
    )

    if args.layers:
        config.layer_range = tuple(args.layers)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration
    import json
    run_config = vars(args)
    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    print("=" * 80)
    print("REFUSAL DIRECTION TRANSFER ANALYSIS")
    print("=" * 80)
    print(f"\nSource model: {args.source} ({source_path})")
    print(f"Target model: {args.target} ({target_path})")
    print(f"Output directory: {output_dir}")
    print(f"\nDataset sizes:")
    print(f"  WikiText: {config.num_wikitext}")
    print(f"  HarmBench: {config.num_harmbench}")
    if config.use_gcg:
        print(f"  GCG: {config.num_gcg}")
    print(f"\nAlignment method: {args.alignment_method}")
    print(f"Direction method: {args.direction_method}")

    # Load datasets
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    wikitext = load_wikitext(config.num_wikitext)
    harmbench = load_harmbench(config.num_harmbench)

    datasets = {
        'clean': wikitext,
        'harm': harmbench
    }

    if config.use_gcg:
        # Determine model index based on source model
        model_index_map = {'llama2': 0, 'llama2-13b': 0}  # Adjust as needed
        model_index = model_index_map.get(args.source, 0)

        gcg_prompts = load_gcg_prompts(config.gcg_path, model_index=model_index, num_samples=config.num_gcg)
        datasets['gcg'] = gcg_prompts

    # Load models
    print("\n" + "=" * 80)
    print("LOADING MODELS")
    print("=" * 80)

    print(f"\nLoading source model: {args.source}")
    source_model = AutoModelForCausalLM.from_pretrained(
        source_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    source_tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)
    if source_tokenizer.pad_token is None:
        source_tokenizer.pad_token = source_tokenizer.eos_token

    print(f"\nLoading target model: {args.target}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    target_tokenizer = AutoTokenizer.from_pretrained(target_path, trust_remote_code=True)
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token

    # Initialize extractors
    source_extractor = EmbeddingExtractor(source_model, source_tokenizer, source_config.device)
    target_extractor = EmbeddingExtractor(target_model, target_tokenizer, target_config.device)

    print(f"\nSource model layers: {source_extractor.num_layers}")
    print(f"Target model layers: {target_extractor.num_layers}")

    # Determine layers to analyze
    if args.specific_layers:
        layers = args.specific_layers
    elif config.layer_range:
        layers = list(range(config.layer_range[0], config.layer_range[1]))
    else:
        layers = list(range(min(source_extractor.num_layers, target_extractor.num_layers)))

    print(f"Analyzing {len(layers)} layers: {layers}")

    # Extract embeddings
    print("\n" + "=" * 80)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 80)

    source_embeddings = {}
    target_embeddings = {}

    for dataset_name, texts in datasets.items():
        print(f"\n--- {dataset_name.upper()} dataset ---")

        print(f"Extracting from source model...")
        source_embeddings[dataset_name] = source_extractor.extract_embeddings(
            texts, layers=layers, batch_size=config.batch_size, position=args.position
        )

        print(f"Extracting from target model...")
        target_embeddings[dataset_name] = target_extractor.extract_embeddings(
            texts, layers=layers, batch_size=config.batch_size, position=args.position
        )

    # Save embeddings if requested
    if config.save_embeddings:
        print("\n" + "=" * 80)
        print("SAVING EMBEDDINGS")
        print("=" * 80)

        embeddings_dir = output_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)

        for dataset_name in datasets.keys():
            torch.save(
                source_embeddings[dataset_name],
                embeddings_dir / f"source_{dataset_name}_embeddings.pt"
            )
            torch.save(
                target_embeddings[dataset_name],
                embeddings_dir / f"target_{dataset_name}_embeddings.pt"
            )

        print(f"Saved embeddings to {embeddings_dir}")

    # Analyze layer-wise transfer
    print("\n" + "=" * 80)
    print("ANALYZING LAYER-WISE TRANSFER")
    print("=" * 80)

    results_df = analyze_layer_wise_transfer(
        source_embeddings,
        target_embeddings,
        layers,
        output_dir
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nTop 5 layers by cosine similarity:")
    top_layers = results_df.nlargest(5, 'cosine_similarity')[
        ['layer', 'cosine_similarity', 'angular_distance', 'procrustes_disparity']
    ]
    print(top_layers.to_string(index=False))

    print("\nTop 5 layers by alignment quality (lowest Procrustes disparity):")
    best_alignment = results_df.nsmallest(5, 'procrustes_disparity')[
        ['layer', 'procrustes_disparity', 'cosine_similarity']
    ]
    print(best_alignment.to_string(index=False))

    if 'gcg_cosine_similarity' in results_df.columns:
        print("\nTop 5 layers for GCG direction transfer:")
        top_gcg = results_df.nlargest(5, 'gcg_cosine_similarity')[
            ['layer', 'gcg_cosine_similarity', 'gcg_angular_distance']
        ]
        print(top_gcg.to_string(index=False))

    # Create visualizations
    if not args.no_plots:
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)

        plot_analysis_results(results_df, output_dir)

    # Save summary
    best_layer = int(results_df.loc[results_df['cosine_similarity'].idxmax(), 'layer'])
    summary = {
        'source_model': args.source,
        'target_model': args.target,
        'best_layer': best_layer,
        'best_cosine_similarity': float(results_df['cosine_similarity'].max()),
        'mean_cosine_similarity': float(results_df['cosine_similarity'].mean()),
        'median_cosine_similarity': float(results_df['cosine_similarity'].median()),
        'best_alignment_layer': int(results_df.loc[results_df['procrustes_disparity'].idxmin(), 'layer']),
        'best_alignment_disparity': float(results_df['procrustes_disparity'].min()),
        'num_layers_analyzed': len(layers)
    }

    import json
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nBest layer for transfer: {best_layer}")
    print(f"Cosine similarity at best layer: {summary['best_cosine_similarity']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Layer-wise analysis: layer_wise_analysis.csv")
    if not args.no_plots:
        print(f"  - Visualizations: analysis_plots.png, metrics_heatmap.png")
    print(f"  - Summary: summary.json")
    if config.save_embeddings:
        print(f"  - Embeddings: embeddings/")


if __name__ == "__main__":
    main()