#!/usr/bin/env python3
"""
computational efficiency analysis for harmonization methods.

measures and compares:
- inference time (per-slice and per-volume)
- memory usage (peak gpu/cpu memory)
- model complexity (parameter count, flops)
- throughput (samples per second)

this analysis is important for demonstrating practical
applicability of the harmonization method.
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sys

# add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class EfficiencyMetrics:
    """container for efficiency metrics."""
    method_name: str
    # timing metrics
    inference_time_per_slice_ms: float
    inference_time_per_volume_ms: float
    total_processing_time_s: float
    # memory metrics
    peak_memory_mb: float
    model_size_mb: float
    # model complexity
    parameter_count: int
    estimated_flops: float
    # throughput
    throughput_slices_per_sec: float
    throughput_volumes_per_hour: float


def count_parameters(model) -> int:
    """count trainable parameters in pytorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, input_shape: Tuple[int, ...]) -> float:
    """
    estimate flops for a single forward pass.

    this is a rough estimate based on layer types.
    """
    try:
        from thop import profile
        import torch
        dummy_input = torch.randn(1, *input_shape)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return flops
    except ImportError:
        # fallback: rough estimate based on parameter count
        n_params = count_parameters(model)
        # assume ~2 flops per parameter per forward pass
        return n_params * 2


def measure_inference_time(
    model,
    input_data: np.ndarray,
    device: str = 'cuda',
    n_warmup: int = 5,
    n_trials: int = 20
) -> Tuple[float, float]:
    """
    measure inference time with warmup.

    args:
        model: pytorch model
        input_data: numpy array of input
        device: device to run on
        n_warmup: warmup iterations
        n_trials: measurement iterations

    returns:
        (mean_time_ms, std_time_ms)
    """
    import torch

    model.eval()
    model = model.to(device)

    # convert input to tensor
    if isinstance(input_data, np.ndarray):
        input_tensor = torch.from_numpy(input_data).float()
    else:
        input_tensor = input_data

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
            if device == 'cuda':
                torch.cuda.synchronize()

    # measure
    times = []
    with torch.no_grad():
        for _ in range(n_trials):
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_tensor)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # convert to ms

    return float(np.mean(times)), float(np.std(times))


def measure_peak_memory(
    model,
    input_data: np.ndarray,
    device: str = 'cuda'
) -> float:
    """
    measure peak memory usage during inference.

    args:
        model: pytorch model
        input_data: numpy array of input
        device: device string

    returns:
        peak memory in mb
    """
    import torch

    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

        model.eval()
        model = model.to(device)

        input_tensor = torch.from_numpy(input_data).float().to(device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            _ = model(input_tensor)
            torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # convert to mb
        return float(peak_memory)

    else:
        # cpu memory measurement using tracemalloc
        import tracemalloc
        tracemalloc.start()

        model.eval()
        input_tensor = torch.from_numpy(input_data).float()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            _ = model(input_tensor)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return float(peak / (1024 ** 2))


def get_model_size(model) -> float:
    """
    get model size in mb.

    args:
        model: pytorch model

    returns:
        model size in mb
    """
    import torch
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / (1024 ** 2)
        os.unlink(f.name)

    return float(size)


def analyze_model_efficiency(
    model,
    input_shape: Tuple[int, ...],
    method_name: str,
    device: str = 'cuda',
    n_slices_per_volume: int = 155,
    n_test_samples: int = 100
) -> EfficiencyMetrics:
    """
    comprehensive efficiency analysis for a model.

    args:
        model: pytorch model
        input_shape: shape of single input (c, h, w)
        method_name: name of the method
        device: device to run on
        n_slices_per_volume: slices per mri volume
        n_test_samples: number of samples to test

    returns:
        EfficiencyMetrics dataclass
    """
    import torch

    print(f'[efficiency] analyzing {method_name}...')

    # create random input
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # measure inference time
    mean_time_ms, std_time_ms = measure_inference_time(model, input_data, device)

    # measure peak memory
    peak_memory = measure_peak_memory(model, input_data, device)

    # model complexity
    n_params = count_parameters(model)
    flops = estimate_flops(model, input_shape)
    model_size = get_model_size(model)

    # compute derived metrics
    time_per_volume = mean_time_ms * n_slices_per_volume
    throughput_slices = 1000.0 / mean_time_ms if mean_time_ms > 0 else 0
    throughput_volumes_hour = (3600 * 1000) / time_per_volume if time_per_volume > 0 else 0

    # total processing time for n_test_samples
    total_time = (mean_time_ms * n_test_samples) / 1000

    return EfficiencyMetrics(
        method_name=method_name,
        inference_time_per_slice_ms=mean_time_ms,
        inference_time_per_volume_ms=time_per_volume,
        total_processing_time_s=total_time,
        peak_memory_mb=peak_memory,
        model_size_mb=model_size,
        parameter_count=n_params,
        estimated_flops=flops,
        throughput_slices_per_sec=throughput_slices,
        throughput_volumes_per_hour=throughput_volumes_hour
    )


def analyze_baseline_efficiency(
    method_name: str,
    input_shape: Tuple[int, ...],
    n_samples: int = 100
) -> Dict:
    """
    analyze efficiency of classical baseline methods.

    args:
        method_name: name of baseline method
        input_shape: shape of input (h, w) or (c, h, w)
        n_samples: number of samples to test

    returns:
        efficiency metrics dictionary
    """
    # import baseline methods using path manipulation
    baseline_path = Path(__file__).parent.parent / '08_additional_baselines'
    sys.path.insert(0, str(baseline_path))
    from baseline_methods import (
        ZScoreNormalizer, HistogramMatcher, NyulNormalizer, WhiteStripeNormalizer
    )

    print(f'[efficiency] analyzing {method_name}...')

    # create test data
    if len(input_shape) == 3:
        test_data = np.random.randn(input_shape[1], input_shape[2]).astype(np.float32)
    else:
        test_data = np.random.randn(*input_shape).astype(np.float32)

    # initialize method
    if method_name == 'zscore':
        method = ZScoreNormalizer()
        process_func = lambda x: method.normalize(x)
    elif method_name == 'histogram_matching':
        method = HistogramMatcher()
        reference = np.random.randn(*test_data.shape).astype(np.float32)
        method.fit(reference)
        process_func = lambda x: method.transform(x)
    elif method_name == 'nyul':
        method = NyulNormalizer()
        training_images = [np.random.randn(*test_data.shape).astype(np.float32) for _ in range(10)]
        method.learn_standard(training_images)
        process_func = lambda x: method.normalize(x)
    elif method_name == 'whitestripe':
        method = WhiteStripeNormalizer()
        process_func = lambda x: method.normalize(x)
    else:
        raise ValueError(f"unknown method: {method_name}")

    # warmup
    for _ in range(5):
        _ = process_func(test_data)

    # measure timing
    times = []
    for _ in range(n_samples):
        start = time.perf_counter()
        _ = process_func(test_data)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_time = np.mean(times)
    std_time = np.std(times)

    return {
        'method_name': method_name,
        'inference_time_per_slice_ms': float(mean_time),
        'inference_time_std_ms': float(std_time),
        'inference_time_per_volume_ms': float(mean_time * 155),
        'throughput_slices_per_sec': float(1000.0 / mean_time) if mean_time > 0 else 0,
        'peak_memory_mb': 0.0,  # negligible for classical methods
        'model_size_mb': 0.0,
        'parameter_count': 0
    }


def create_comparison_table(metrics_list: List[Dict], output_path: Path):
    """
    create latex comparison table for efficiency metrics.

    args:
        metrics_list: list of efficiency metrics dictionaries
        output_path: path to save table
    """
    latex = r"""\begin{table}[htbp]
\centering
\caption{Computational Efficiency Comparison}
\label{tab:efficiency_comparison}
\begin{tabular}{lcccccc}
\toprule
Method & Time/Slice & Time/Vol. & Throughput & Memory & Params & Size \\
 & (ms) & (s) & (vol/hr) & (MB) & (M) & (MB) \\
\midrule
"""

    for m in metrics_list:
        name = m.get('method_name', 'Unknown')
        time_slice = m.get('inference_time_per_slice_ms', 0)
        time_vol = m.get('inference_time_per_volume_ms', 0) / 1000  # convert to seconds
        throughput = m.get('throughput_volumes_per_hour', 0)
        memory = m.get('peak_memory_mb', 0)
        params = m.get('parameter_count', 0) / 1e6  # convert to millions
        size = m.get('model_size_mb', 0)

        latex += f"{name} & {time_slice:.2f} & {time_vol:.2f} & {throughput:.0f} & "
        latex += f"{memory:.1f} & {params:.2f} & {size:.1f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'[table] saved efficiency table to {output_path}')


def plot_efficiency_comparison(metrics_list: List[Dict], output_dir: Path):
    """
    create efficiency comparison figures.

    args:
        metrics_list: list of efficiency metrics dictionaries
        output_dir: output directory
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    methods = [m['method_name'] for m in metrics_list]
    times = [m.get('inference_time_per_slice_ms', 0) for m in metrics_list]
    throughputs = [m.get('throughput_volumes_per_hour', 0) for m in metrics_list]
    memories = [m.get('peak_memory_mb', 0) for m in metrics_list]

    # create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # inference time
    ax = axes[0]
    colors = ['#2E86AB' if 'SA-CycleGAN' in m else '#A23B72' for m in methods]
    bars = ax.bar(methods, times, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Inference Time (ms/slice)')
    ax.set_title('(a) Inference Time per Slice')
    ax.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{t:.1f}', ha='center', va='bottom', fontsize=8)

    # throughput
    ax = axes[1]
    bars = ax.bar(methods, throughputs, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Throughput (volumes/hour)')
    ax.set_title('(b) Processing Throughput')
    ax.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, throughputs):
        if t > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{t:.0f}', ha='center', va='bottom', fontsize=8)

    # memory
    ax = axes[2]
    bars = ax.bar(methods, memories, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('(c) Peak GPU Memory Usage')
    ax.tick_params(axis='x', rotation=45)
    for bar, m in zip(bars, memories):
        if m > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{m:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_efficiency_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'fig_efficiency_comparison.png', bbox_inches='tight')
    plt.close()

    print(f'[fig] saved efficiency comparison to {output_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='computational efficiency analysis for harmonization methods'
    )
    parser.add_argument('--model-path', type=str,
                       help='path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='device for inference (cuda/cpu/mps)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='output directory for results')
    parser.add_argument('--include-baselines', action='store_true',
                       help='include baseline method analysis')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    # analyze sa-cyclegan if model provided
    if args.model_path and Path(args.model_path).exists():
        try:
            import torch
            from neuroscope.models.architectures.sa_cyclegan_25d import (
                SACycleGAN25D, SACycleGAN25DConfig
            )

            print('[efficiency] loading sa-cyclegan model...')
            config = SACycleGAN25DConfig()
            model = SACycleGAN25D(config)

            checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # handle dataparallel wrapper
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)

            # analyze efficiency
            input_shape = (12, 240, 240)  # 3 slices * 4 modalities
            metrics = analyze_model_efficiency(
                model.generator_ab if hasattr(model, 'generator_ab') else model,
                input_shape,
                'SA-CycleGAN-2.5D',
                args.device
            )
            all_metrics.append(asdict(metrics))

        except Exception as e:
            print(f'[efficiency] error loading model: {e}')

    # analyze baseline methods
    if args.include_baselines:
        input_shape = (240, 240)
        for method in ['zscore', 'histogram_matching', 'nyul', 'whitestripe']:
            try:
                metrics = analyze_baseline_efficiency(method, input_shape)
                all_metrics.append(metrics)
            except Exception as e:
                print(f'[efficiency] error analyzing {method}: {e}')

    # save results
    with open(output_dir / 'efficiency_results.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # create comparison table and figures
    if all_metrics:
        create_comparison_table(all_metrics, output_dir / 'table_efficiency.tex')
        plot_efficiency_comparison(all_metrics, output_dir)

    print('=' * 60)
    print('[efficiency] analysis complete')
    print(f'[efficiency] results saved to {output_dir}')


if __name__ == '__main__':
    main()
