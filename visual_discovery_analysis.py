#!/usr/bin/env python3
"""
Visual Discovery Analysis: Configuration-Driven Validation
==========================================================

This script uses a configuration file to run main.py with different settings
and then saves the results. The plotting functionality has been moved to
visual_plotter.py for better separation of concerns.

Key Features:
1. JSON configuration file defines all experiment parameters
2. Calls main.py via subprocess for each configuration
3. Parses results from main.py output
4. Saves comprehensive results with config for later analysis
5. No plotting - use visual_plotter.py to generate plots
"""

import json
import subprocess
import re
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# ════════════════════════════════════════════════════════════════
# CONFIGURATION TEMPLATE
# ════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "experiment_name": "positively_correlated_noise_validation",
    "base_command": "uv run main.py",
    "common_args": {
        "k": 2048,
        "epochs": 50,
        "dataset-size": 50000,
        "target-epsilon": 2.0,
        "delta": 1e-5,
        "clip-radius": 2.0,
        "dp-layer": "conv1,conv2",
        "mps": True,
        "clean": True,
        "compare-others": True,
        "run-mia": True
    },
    "experiments": [
        {
            "name": "25_users_positive",
            "users": 25,
            "positively_correlated_noise": True,
            "description": "25 users with positively correlated noise"
        },
        {
            "name": "25_users_negative", 
            "users": 25,
            "negatively_correlated_noise": True,
            "description": "25 users with negatively correlated noise (default)"
        },
        {
            "name": "50_users_positive",
            "users": 50,
            "positively_correlated_noise": True,
            "description": "50 users with positively correlated noise"
        },
        {
            "name": "50_users_negative",
            "users": 50,
            "negatively_correlated_noise": True,
            "description": "50 users with negatively correlated noise (default)"
        },
        {
            "name": "100_users_positive",
            "users": 100,
            "positively_correlated_noise": True,
            "description": "100 users with positively correlated noise"
        },
        {
            "name": "100_users_negative",
            "users": 100,
            "negatively_correlated_noise": True,
            "description": "100 users with negatively correlated noise (default)"
        },
        {
            "name": "200_users_positive",
            "users": 200,
            "positively_correlated_noise": True,
            "description": "200 users with positively correlated noise"
        },
        {
            "name": "200_users_negative",
            "users": 200,
            "negatively_correlated_noise": True,
            "description": "200 users with negatively correlated noise (default)"
        },
        {
            "name": "400_users_positive",
            "users": 400,
            "positively_correlated_noise": True,
            "description": "400 users with positively correlated noise"
        },
        {
            "name": "400_users_negative",
            "users": 400,
            "negatively_correlated_noise": True,
            "description": "400 users with negatively correlated noise (default)"
        }
    ],
    "output_settings": {
        "results_dir": "validation_results",
        "plots_dir": "validation_plots",
        "save_logs": True,
        "figure_dpi": 300
    }
}

def create_config_file(filename="validation_config.json"):
    """Create the default configuration file"""
    with open(filename, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(f"📄 Created configuration file: {filename}")
    return filename

def load_config(filename="validation_config.json"):
    """Load configuration from JSON file"""
    if not os.path.exists(filename):
        print(f"⚠️  Config file {filename} not found. Creating default...")
        filename = create_config_file(filename)
    
    with open(filename, 'r') as f:
        config = json.load(f)
    
    print(f"📄 Loaded configuration: {filename}")
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
        print(f"   {i:2d}. {exp['users']:3d} users - {noise_type} correlation")
    
    print(f"\n" + "="*80)
    
    # Run experiments with enhanced progress tracking
    with tqdm(total=len(config["experiments"]), desc="🔬 Overall Progress", position=0) as main_pbar:
        for i, experiment in enumerate(config["experiments"], 1):
            # Update main progress bar description
            noise_type = "Positive" if experiment.get('positively_correlated_noise') else "Negative"
            main_pbar.set_description(f"🔬 Exp {i}/{len(config['experiments'])}: {experiment['users']} users ({noise_type})")
            
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
    
    # Group by user count and experiment type
    user_counts = sorted(df['users'].unique())
    
    # Print basic analysis summary
    print_basic_analysis_summary(results, config, user_counts)

def print_basic_analysis_summary(results, config, user_counts):
    """Print a basic analysis summary without plotting"""
    print(f"\n" + "="*60)
    print(f"CONFIGURATION-DRIVEN VALIDATION SUMMARY")
    print(f"="*60)
    
    print(f"📊 Experiment Configuration:")
    print(f"   • Fisher k: {config['common_args']['k']}")
    print(f"   • Privacy: ε={config['common_args']['target-epsilon']}, δ={config['common_args']['delta']}")
    print(f"   • Epochs: {config['common_args']['epochs']}")
    print(f"   • Layers: {config['common_args']['dp-layer']}")
    print(f"   • User counts tested: {user_counts}")
    
    # Group results by noise strategy
    positive_results = [r for r in results if r.get('noise_strategy') == 'positive']
    negative_results = [r for r in results if r.get('noise_strategy') == 'negative']
    
    print(f"\n🎯 Key Findings:")
    print(f"   📊 Experiments completed:")
    print(f"     • Positive noise correlation: {len(positive_results)} experiments")
    print(f"     • Negative noise correlation: {len(negative_results)} experiments")
    
    # Calculate wins for each user count
    wins = 0
    total_comparisons = 0
    
    for uc in user_counts:
        pos_data = [r for r in positive_results if r.get('users') == uc]
        neg_data = [r for r in negative_results if r.get('users') == uc]
        
        if pos_data and neg_data:
            pos_fisher = pos_data[0].get('fisher_dp', 0)
            neg_fisher = neg_data[0].get('fisher_dp', 0)
            if pos_fisher > neg_fisher + 0.5:  # Significant advantage threshold
                wins += 1
            total_comparisons += 1
            
            print(f"   {uc:3d} users:")
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

def main():
    """Main validation function"""
    print("🎨 CONFIGURATION-DRIVEN VALIDATION ANALYSIS")
    print("=" * 60)
    print("This script uses JSON configuration to run main.py with different")
    print("settings, ensuring perfect consistency and eliminating drift.")
    print("Results are saved for later plotting with visual_plotter.py")
    print("=" * 60)
    
    # Load or create configuration
    config = load_config()
    
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