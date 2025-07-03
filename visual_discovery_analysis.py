#!/usr/bin/env python3
"""
Visual Discovery Analysis: Configuration-Driven Validation
==========================================================

This script uses external JSON configuration files to run main.py with different settings
and then saves the results. The plotting functionality has been moved to
visual_plotter.py for better separation of concerns.

Key Features:
1. External JSON configuration files define all experiment parameters
2. Support for parameter grids (automatic combinatorial experiments)
3. Calls main.py via subprocess for each configuration
4. Parses results from main.py output
5. Saves comprehensive results with config for later analysis
6. No plotting - use visual_plotter.py to generate plots

Config Format:
- Manual experiments: List each experiment individually (current approach)
- Parameter grids: Automatically generate all combinations of parameters
- Hybrid: Mix both approaches in the same config

Available Config Files (in validation_configs/ directory):
- validation_config_users_num.json: User count sensitivity analysis
- validation_config_clip_radius.json: Clip radius sensitivity analysis

Usage:
  python visual_discovery_analysis.py --list-configs
  python visual_discovery_analysis.py --config validation_config_users_num.json
  python visual_discovery_analysis.py --config validation_config_clip_radius.json
"""

import json
import subprocess
import re
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import argparse
from itertools import product

# ════════════════════════════════════════════════════════════════
# CONFIGURATION MANAGEMENT (External Files + Parameter Grids)
# ════════════════════════════════════════════════════════════════

def expand_parameter_grid(config):
    """Expand parameter_grid into individual experiments"""
    if 'parameter_grid' not in config:
        return config
    
    grid = config['parameter_grid']
    if not grid:
        return config
    
    print(f"🔧 Expanding parameter grid...")
    
    # Get parameter names and values
    param_names = list(grid.keys())
    param_values = [grid[name] if isinstance(grid[name], list) else [grid[name]] for name in param_names]
    
    # Generate all combinations
    combinations = list(product(*param_values))
    
    print(f"   📊 Parameters: {param_names}")
    for name, values in zip(param_names, param_values):
        print(f"   • {name}: {values}")
    print(f"   🔢 Total combinations: {len(combinations)}")
    
    # Generate experiments from combinations
    grid_experiments = []
    for i, combo in enumerate(combinations):
        # Create experiment name
        name_parts = []
        exp_dict = {}
        
        for param_name, param_value in zip(param_names, combo):
            # Handle special parameter names for naming
            if param_name == 'target-epsilon':
                name_parts.append(f"eps_{param_value}")
            elif param_name == 'dp-layer':
                layer_name = param_value.replace(',', '_').replace('conv', 'c')
                name_parts.append(f"layer_{layer_name}")
            elif param_name == 'users':
                name_parts.append(f"u{param_value}")
            elif param_name == 'clip-radius':
                name_parts.append(f"clip_{param_value}")
            elif param_name == 'noise_strategy':
                if param_value == 'positive':
                    exp_dict['positively_correlated_noise'] = True
                    name_parts.append("pos")
                elif param_value == 'negative':
                    exp_dict['negatively_correlated_noise'] = True
                    name_parts.append("neg")
                continue  # Don't add noise_strategy directly to exp_dict
            else:
                name_parts.append(f"{param_name}_{param_value}")
            
            # Add parameter to experiment (except noise_strategy which is handled above)
            if param_name != 'noise_strategy':
                exp_dict[param_name] = param_value
        
        exp_dict['name'] = "_".join(name_parts)
        exp_dict['description'] = f"Grid experiment: {', '.join(f'{k}={v}' for k, v in zip(param_names, combo))}"
        
        grid_experiments.append(exp_dict)
    
    # Combine with existing manual experiments
    manual_experiments = config.get('experiments', [])
    all_experiments = manual_experiments + grid_experiments
    
    print(f"   ✅ Generated {len(grid_experiments)} experiments from parameter grid")
    if manual_experiments:
        print(f"   📝 Plus {len(manual_experiments)} manual experiments")
    print(f"   🎯 Total experiments: {len(all_experiments)}")
    
    # Update config
    config['experiments'] = all_experiments
    
    return config

def load_config(filename):
    """Load configuration from JSON file and expand parameter grids"""
    # If filename doesn't include path, look in validation_configs directory
    if not os.path.dirname(filename):
        config_path = os.path.join('validation_configs', filename)
    else:
        config_path = filename
    
    if not os.path.exists(config_path):
        print(f"❌ Config file {config_path} not found.")
        print(f"📄 Available config files:")
        config_dir = 'validation_configs'
        if os.path.exists(config_dir):
            config_files = [f for f in os.listdir(config_dir) if f.startswith('validation_config_') and f.endswith('.json')]
            for i, config_file in enumerate(config_files, 1):
                print(f"   {i}. {config_file}")
        else:
            print(f"   No validation_configs directory found")
        raise FileNotFoundError(f"Configuration file {config_path} does not exist")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"📄 Loaded configuration: {config_path}")
    
    # Expand parameter grids if present
    config = expand_parameter_grid(config)
    
    return config

def build_command(config, experiment):
    """Build the main.py command for a specific experiment"""
    cmd_parts = [config["base_command"]]
    
    # Add common arguments
    for key, value in config["common_args"].items():
        if isinstance(value, bool) and value:
            cmd_parts.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd_parts.append(f"--{key}")
            cmd_parts.append(str(value))
    
    # Add experiment-specific arguments
    for key, value in experiment.items():
        if key in ["name", "description"]:
            continue
        if isinstance(value, bool) and value:
            # Special handling for noise correlation arguments - keep underscores
            if key in ["positively_correlated_noise", "negatively_correlated_noise"]:
                cmd_parts.append(f"--{key}")
            else:
                cmd_parts.append(f"--{key.replace('_', '-')}")
        elif not isinstance(value, bool):
            cmd_parts.append(f"--{key.replace('_', '-')}")
            cmd_parts.append(str(value))
    
    return " ".join(cmd_parts)

def parse_main_output(output_text):
    """Parse the output from main.py to extract accuracy results and MIA results"""
    results = {}
    
    # Look for the accuracy summary section
    accuracy_pattern = r"📊\s+Accuracy summary.*?\n(.*?)(?=\n\n|\n💾|\nFisher vs|\Z)"
    accuracy_match = re.search(accuracy_pattern, output_text, re.DOTALL)
    
    if accuracy_match:
        accuracy_section = accuracy_match.group(1)
        
        # Extract individual accuracies
        patterns = {
            'baseline': r'baseline\s*:\s*(\d+\.?\d*)%',
            'fisher_dp': r'Fisher DP\s*:\s*(\d+\.?\d*)%',
            'vanilla_dp': r'Vanilla DP\s*:\s*(\d+\.?\d*)%',
            'dp_sat': r'DP-SAT\s*:\s*(\d+\.?\d*)%'
        }
        
        for name, pattern in patterns.items():
            match = re.search(pattern, accuracy_section)
            if match:
                results[name] = float(match.group(1))
    
    # Extract improvement comparisons
    improvement_patterns = {
        'fisher_vs_vanilla': r'Fisher vs Vanilla:\s*([+-]?\d+\.?\d*)%',
        'fisher_vs_dp_sat': r'Fisher vs DP-SAT\s*:\s*([+-]?\d+\.?\d*)%',
        'dp_sat_vs_vanilla': r'DP-SAT vs Vanilla:\s*([+-]?\d+\.?\d*)%'
    }
    
    for name, pattern in improvement_patterns.items():
        match = re.search(pattern, output_text)
        if match:
            results[name] = float(match.group(1))
    
    # Extract MIA results: Confidence Attack AUCs
    confidence_section_pattern = r"🎯 Confidence Attack AUC:(.*?)(?=\n\n|🕶️|📊|\Z)"
    confidence_match = re.search(confidence_section_pattern, output_text, re.DOTALL)
    
    if confidence_match:
        confidence_section = confidence_match.group(1)
        confidence_patterns = {
            'baseline_confidence_auc': r'\s+Baseline:\s*(\d+\.?\d*)',
            'fisher_dp_confidence_auc': r'\s+Fisher DP:\s*(\d+\.?\d*)',
            'vanilla_dp_confidence_auc': r'\s+Vanilla DP:\s*(\d+\.?\d*)',
            'dp_sat_confidence_auc': r'\s+DP-SAT:\s*(\d+\.?\d*)',
            'l2_baseline_confidence_auc': r'\s+L2 Baseline:\s*(\d+\.?\d*)'
        }
        
        for name, pattern in confidence_patterns.items():
            match = re.search(pattern, confidence_section)
            if match:
                results[name] = float(match.group(1))
    
    # Extract MIA results: Shadow Attack AUCs
    shadow_section_pattern = r"🕶️\s+Shadow Attack AUC:(.*?)(?=\n\n|🧮|📊|\Z)"
    shadow_match = re.search(shadow_section_pattern, output_text, re.DOTALL)
    
    if shadow_match:
        shadow_section = shadow_match.group(1)
        shadow_patterns = {
            'baseline_shadow_auc': r'\s+Baseline:\s*(\d+\.?\d*)',
            'fisher_dp_shadow_auc': r'\s+Fisher DP:\s*(\d+\.?\d*)',
            'vanilla_dp_shadow_auc': r'\s+Vanilla DP:\s*(\d+\.?\d*)',
            'dp_sat_shadow_auc': r'\s+DP-SAT:\s*(\d+\.?\d*)',
            'l2_baseline_shadow_auc': r'\s+L2 Baseline:\s*(\d+\.?\d*)'
        }
        
        for name, pattern in shadow_patterns.items():
            match = re.search(pattern, shadow_section)
            if match:
                results[name] = float(match.group(1))
    
    # Extract MIA results: Worst-case AUCs
    worst_case_pattern = r"📊 Worst-case AUC:(.*?)(?=\n\n|🏆|✅|⚠️|\Z)"
    worst_case_match = re.search(worst_case_pattern, output_text, re.DOTALL)
    
    if worst_case_match:
        worst_case_section = worst_case_match.group(1)
        worst_case_patterns = {
            'baseline_worst_auc': r'\s+•\s+Baseline:\s*(\d+\.?\d*)',
            'fisher_dp_worst_auc': r'\s+•\s+Fisher DP:\s*(\d+\.?\d*)',
            'vanilla_dp_worst_auc': r'\s+•\s+Vanilla DP:\s*(\d+\.?\d*)',
            'dp_sat_worst_auc': r'\s+•\s+DP-SAT:\s*(\d+\.?\d*)',
            'l2_baseline_worst_auc': r'\s+•\s+L2 Baseline:\s*(\d+\.?\d*)'
        }
        
        for name, pattern in worst_case_patterns.items():
            match = re.search(pattern, worst_case_section)
            if match:
                results[name] = float(match.group(1))
    
    return results

def run_experiment(config, experiment):
    """Run a single experiment by calling main.py"""
    print(f"\n🚀 Running experiment: {experiment['name']}")
    print(f"📝 Description: {experiment['description']}")
    
    command = build_command(config, experiment)
    print(f"💻 Command: {command}")
    
    try:
        # Run the command with real-time output streaming
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Collect output while showing progress
        stdout_lines = []
        stderr_lines = []
        
        print(f"📊 Streaming output from main.py:")
        print("-" * 60)
        
        # Create a simple progress indicator
        progress_chars = "|/-\\"
        progress_idx = 0
        
        # Read output line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                stdout_lines.append(output.strip())
                
                # Show important progress lines
                line = output.strip()
                if any(keyword in line.lower() for keyword in [
                    'training', 'epoch', 'fisher', 'dp-sgd', 'baseline', 
                    'accuracy', 'privacy', 'computing', 'calibration'
                ]):
                    # Show progress character and important lines
                    print(f"{progress_chars[progress_idx % len(progress_chars)]} {line}")
                    progress_idx += 1
                elif '📊' in line or '✅' in line or '🎯' in line:
                    # Show result lines
                    print(f"  {line}")
        
        # Get any remaining stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            stderr_lines.extend(stderr_output.split('\n'))
        
        # Wait for process to complete
        return_code = process.wait()
        
        print("-" * 60)
        
        if return_code != 0:
            print(f"❌ Experiment failed with return code {return_code}")
            if stderr_lines:
                print(f"Error output: {' '.join(stderr_lines)}")
            return None
        
        # Parse the output
        full_stdout = '\n'.join(stdout_lines)
        parsed_results = parse_main_output(full_stdout)
        
        # Add metadata
        parsed_results.update({
            'experiment_name': experiment['name'],
            'users': experiment.get('users', 0),
            'noise_strategy': 'positive' if experiment.get('positively_correlated_noise') else 'negative',
            'command': command,
            'stdout': full_stdout if config.get("output_settings", {}).get("save_logs", False) else "",
            'success': True
        })
        
        print(f"✅ Experiment completed successfully")
        if 'fisher_dp' in parsed_results:
            print(f"   📊 Fisher DP: {parsed_results['fisher_dp']:.2f}%")
        if 'vanilla_dp' in parsed_results:
            print(f"   📊 Vanilla DP: {parsed_results['vanilla_dp']:.2f}%")
        if 'dp_sat' in parsed_results:
            print(f"   📊 DP-SAT: {parsed_results['dp_sat']:.2f}%")
        if 'fisher_vs_vanilla' in parsed_results:
            print(f"   📊 Fisher vs Vanilla: {parsed_results['fisher_vs_vanilla']:+.2f}%")
        
        return parsed_results
        
    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        return None

def run_all_experiments(config):
    """Run all experiments defined in the configuration"""
    results = []
    
    # Create output directories
    os.makedirs(config["output_settings"]["results_dir"], exist_ok=True)
    
    print(f"🎯 Starting validation with {len(config['experiments'])} experiments")
    print(f"📁 Results will be saved to: {config['output_settings']['results_dir']}")
    
    # Show experiment overview
    print(f"\n📋 Experiment Overview:")
    for i, exp in enumerate(config["experiments"], 1):
        noise_type = "Positive" if exp.get('positively_correlated_noise') else "Negative"
        # Handle different experiment types flexibly
        if 'users' in exp:
            param_desc = f"{exp['users']} users"
        elif 'clip-radius' in exp:
            param_desc = f"clip={exp['clip-radius']}"
        else:
            param_desc = exp.get('name', f"exp_{i}")
        print(f"   {i:2d}. {param_desc} - {noise_type} correlation")
    
    print(f"\n" + "="*80)
    
    # Run experiments with enhanced progress tracking
    with tqdm(total=len(config["experiments"]), desc="🔬 Overall Progress", position=0) as main_pbar:
        for i, experiment in enumerate(config["experiments"], 1):
            # Update main progress bar description
            noise_type = "Positive" if experiment.get('positively_correlated_noise') else "Negative"
            # Handle different experiment types flexibly
            if 'users' in experiment:
                param_desc = f"{experiment['users']} users"
            elif 'clip-radius' in experiment:
                param_desc = f"clip={experiment['clip-radius']}"
            else:
                param_desc = experiment.get('name', f"exp_{i}")
            main_pbar.set_description(f"🔬 Exp {i}/{len(config['experiments'])}: {param_desc} - {noise_type}")
            
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {i}/{len(config['experiments'])}")
            print(f"{'='*80}")
            
            result = run_experiment(config, experiment)
            if result:
                results.append(result)
                status = "✅ SUCCESS"
            else:
                status = "❌ FAILED"
            
            # Update progress bar with current status
            main_pbar.update(1)
            main_pbar.set_postfix({
                'completed': len(results),
                'failed': i - len(results),
                'current': status
            })
            
            print(f"\n📊 Progress Summary: {len(results)}/{i} experiments completed successfully")
    
    print(f"\n" + "="*80)
    print(f"🎉 VALIDATION BATCH COMPLETE")
    print(f"📊 Completed {len(results)}/{len(config['experiments'])} experiments successfully")
    
    if len(results) < len(config['experiments']):
        failed_count = len(config['experiments']) - len(results)
        print(f"⚠️  {failed_count} experiments failed - check logs above for details")
    
    # Get the random seed for filename
    from config import get_random_seed
    seed = get_random_seed()
    
    # Save results with config for visual_plotter.py
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(config["output_settings"]["results_dir"], f"validation_results_{timestamp}_seed_{seed}.json")
    
    # Save both results and config in the new format
    output_data = {
        'results': results,
        'config': config,
        'metadata': {
            'timestamp': timestamp,
            'seed': seed,
            'total_experiments': len(config['experiments']),
            'successful_experiments': len(results),
            'failed_experiments': len(config['experiments']) - len(results)
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results and config saved to: {results_file}")
    print(f"🎲 Random seed: {seed}")
    print(f"📊 Use visual_plotter.py to generate plots from these results")
    
    return results, results_file

def analyze_results(results, config):
    """Analyze the experimental results (no plotting, just summary)"""
    if not results:
        print("❌ No results to analyze")
        return
    
    print(f"\n📊 Analyzing {len(results)} experimental results...")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Detect what parameter is varying and get its values
    varying_param, param_values = detect_varying_parameter(results, config)
    
    # Print basic analysis summary
    print_basic_analysis_summary(results, config, varying_param, param_values)

def detect_varying_parameter(results, config):
    """Detect which parameter is being varied across experiments"""
    if not results:
        return 'users', []
    
    # Check common parameters that might vary
    param_candidates = ['users', 'target-epsilon', 'clip-radius', 'dp-layer', 'k']
    
    for param in param_candidates:
        values = []
        
        # Check if this parameter varies across experiments
        for result in results:
            # Try to get value from experiment metadata or parsed fields
            if param == 'target-epsilon':
                # For epsilon experiments, check experiment_name or direct field
                if 'experiment_name' in result and 'eps_' in result['experiment_name']:
                    exp_name = result['experiment_name']
                    parts = exp_name.split('_')
                    for i, part in enumerate(parts):
                        if part == 'eps' and i + 1 < len(parts):
                            try:
                                values.append(float(parts[i + 1]))
                                break
                            except ValueError:
                                pass
                elif param in result:
                    values.append(result[param])
            elif param == 'clip-radius':
                # For clip radius experiments
                if 'experiment_name' in result and 'clip_' in result['experiment_name']:
                    exp_name = result['experiment_name']
                    parts = exp_name.split('_')
                    for i, part in enumerate(parts):
                        if part == 'clip' and i + 1 < len(parts):
                            try:
                                values.append(float(parts[i + 1]))
                                break
                            except ValueError:
                                pass
                elif param in result:
                    values.append(result[param])
            else:
                # For other parameters, try direct access
                if param in result:
                    values.append(result[param])
        
        # Check if this parameter actually varies
        unique_values = list(set(values))
        if len(unique_values) > 1:
            return param, sorted(unique_values)
    
    # Fallback to users if no clear varying parameter detected
    user_values = [r.get('users', 0) for r in results]
    return 'users', sorted(set(user_values))

def print_basic_analysis_summary(results, config, varying_param, param_values):
    """Print a basic analysis summary without plotting - parameter agnostic"""
    print(f"\n" + "="*60)
    print(f"CONFIGURATION-DRIVEN VALIDATION SUMMARY")
    print(f"="*60)
    
    print(f"📊 Experiment Configuration:")
    print(f"   • Fisher k: {config['common_args']['k']}")
    
    # Handle epsilon specially - it might be varying or fixed
    if varying_param == 'target-epsilon':
        print(f"   • Privacy: ε=VARYING ({param_values}), δ={config['common_args']['delta']}")
    else:
        epsilon_val = config['common_args'].get('target-epsilon', 'N/A')
        print(f"   • Privacy: ε={epsilon_val}, δ={config['common_args']['delta']}")
    
    print(f"   • Epochs: {config['common_args']['epochs']}")
    print(f"   • Layers: {config['common_args']['dp-layer']}")
    
    # Show what parameter is being varied
    if varying_param == 'users':
        print(f"   • Users tested: {param_values}")
    elif varying_param == 'target-epsilon':
        print(f"   • Epsilon values tested: {param_values}")
    elif varying_param == 'clip-radius':
        print(f"   • Clip radius values tested: {param_values}")
        print(f"   • Users: {config['common_args']['users']}")
    else:
        print(f"   • Varying parameter: {varying_param} = {param_values}")
    
    # Group results by noise strategy
    positive_results = [r for r in results if r.get('noise_strategy') == 'positive']
    negative_results = [r for r in results if r.get('noise_strategy') == 'negative']
    
    print(f"\n🎯 Key Findings:")
    print(f"   📊 Experiments completed:")
    print(f"     • Positive noise correlation: {len(positive_results)} experiments")
    print(f"     • Negative noise correlation: {len(negative_results)} experiments")
    
    # Calculate wins for each parameter value
    wins = 0
    total_comparisons = 0
    
    for param_val in param_values:
        # Get results for this parameter value
        pos_data = []
        neg_data = []
        
        for r in positive_results:
            if get_param_value_from_result(r, varying_param) == param_val:
                pos_data.append(r)
        
        for r in negative_results:
            if get_param_value_from_result(r, varying_param) == param_val:
                neg_data.append(r)
        
        if pos_data and neg_data:
            pos_fisher = pos_data[0].get('fisher_dp', 0)
            neg_fisher = neg_data[0].get('fisher_dp', 0)
            if pos_fisher > neg_fisher + 0.5:  # Significant advantage threshold
                wins += 1
            total_comparisons += 1
            
            # Format parameter value for display
            if varying_param == 'target-epsilon':
                param_display = f"ε={param_val}"
            elif varying_param == 'clip-radius':
                param_display = f"clip={param_val}"
            elif varying_param == 'users':
                param_display = f"{param_val} users"
            else:
                param_display = f"{varying_param}={param_val}"
            
            print(f"   {param_display}:")
            print(f"     • Positive Fisher DP: {pos_fisher:.2f}%")
            print(f"     • Negative Fisher DP: {neg_fisher:.2f}%")
            print(f"     • Advantage: {pos_fisher - neg_fisher:+.2f}%")
    
    if total_comparisons > 0:
        print(f"\n📈 Overall Summary:")
        print(f"   • Positively correlated noise wins: {wins}/{total_comparisons} conditions")
        if wins > 0:
            print(f"   ✅ Discovery confirmed under multiple conditions")
        else:
            print(f"   ❌ Positively correlated advantage not observed")
    
    print(f"\n📊 Next Steps:")
    print(f"   • Use visual_plotter.py to generate comprehensive plots")
    print(f"   • Example: python visual_plotter.py --latest")
    print(f"\n✅ Validation analysis complete")

def get_param_value_from_result(result, param_name):
    """Extract parameter value from result for analysis"""
    if param_name == 'target-epsilon':
        # Try experiment name parsing first
        if 'experiment_name' in result and 'eps_' in result['experiment_name']:
            exp_name = result['experiment_name']
            parts = exp_name.split('_')
            for i, part in enumerate(parts):
                if part == 'eps' and i + 1 < len(parts):
                    try:
                        return float(parts[i + 1])
                    except ValueError:
                        pass
        # Fallback to direct field access
        return result.get('target-epsilon', result.get('target_epsilon', 0))
    elif param_name == 'clip-radius':
        # Try experiment name parsing first
        if 'experiment_name' in result and 'clip_' in result['experiment_name']:
            exp_name = result['experiment_name']
            parts = exp_name.split('_')
            for i, part in enumerate(parts):
                if part == 'clip' and i + 1 < len(parts):
                    try:
                        return float(parts[i + 1])
                    except ValueError:
                        pass
        return result.get('clip-radius', result.get('clip_radius', 0))
    else:
        return result.get(param_name, 0)

def main():
    """Main validation function with command line argument support"""
    parser = argparse.ArgumentParser(
        description='Configuration-Driven Validation Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Configuration Files (in validation_configs/):
  validation_config_users_num.json      - User count sensitivity analysis
  validation_config_clip_radius.json    - Clip radius sensitivity analysis

Examples:
  %(prog)s --config validation_config_users_num.json       # Run user count experiments
  %(prog)s --config validation_config_clip_radius.json     # Run clip radius experiments
  %(prog)s --list-configs                                   # List available config files
        """
    )
    
    parser.add_argument('--config', type=str, 
                       help='Configuration file name (will look in validation_configs/ directory)')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configuration files and exit')
    
    args = parser.parse_args()
    
    print("🎨 CONFIGURATION-DRIVEN VALIDATION ANALYSIS")
    print("=" * 60)
    print("This script uses external JSON configuration files to run main.py")
    print("with different settings, ensuring perfect consistency and eliminating drift.")
    print("Results are saved for later plotting with visual_plotter.py")
    print("=" * 60)
    
    # Handle list configs option
    if args.list_configs:
        print(f"\n📄 Available Configuration Files (validation_configs/):")
        config_dir = 'validation_configs'
        if os.path.exists(config_dir):
            config_files = [f for f in os.listdir(config_dir) if f.startswith('validation_config_') and f.endswith('.json')]
            if config_files:
                for i, config_file in enumerate(config_files, 1):
                    config_path = os.path.join(config_dir, config_file)
                    try:
                        with open(config_path, 'r') as f:
                            config_data = json.load(f)
                        exp_name = config_data.get('experiment_name', 'Unknown')
                        exp_count = len(config_data.get('experiments', []))
                        print(f"   {i}. {config_file}")
                        print(f"      • Experiment: {exp_name}")
                        print(f"      • Total experiments: {exp_count}")
                    except Exception as e:
                        print(f"   {i}. {config_file} (Error reading: {e})")
            else:
                print("   No configuration files found matching pattern 'validation_config_*.json'")
        else:
            print("   validation_configs/ directory not found")
        return None
    
    # Require config file
    if not args.config:
        print("❌ Error: --config argument is required")
        print("\n📄 Available configuration files (validation_configs/):")
        config_dir = 'validation_configs'
        if os.path.exists(config_dir):
            config_files = [f for f in os.listdir(config_dir) if f.startswith('validation_config_') and f.endswith('.json')]
            for i, config_file in enumerate(config_files, 1):
                print(f"   {i}. {config_file}")
        else:
            print("   validation_configs/ directory not found")
        print(f"\nUsage: python {os.path.basename(__file__)} --config <config_file>")
        print(f"   or: python {os.path.basename(__file__)} --list-configs")
        return None
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None
    
    # Run all experiments
    results, results_file = run_all_experiments(config)
    
    # Analyze results (basic summary only)
    analyze_results(results, config)
    
    print(f"\n🎉 Validation analysis complete!")
    print(f"📁 Results: {results_file}")
    print(f"📊 Generate plots: python visual_plotter.py --latest")
    
    return results

if __name__ == "__main__":
    results = main() 