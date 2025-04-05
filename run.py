import logging
from pathlib import Path
from itertools import product
import json
from train import ContractTrainer

class ExperimentRunner:
    def __init__(self, base_config):
        """
        Initialize runner with base configuration
        base_config: dict containing base parameters
        """
        self.base_config = base_config
        self.setup_logging()
        self.create_directories()
        
    def setup_logging(self):
        """Setup logging for the runner"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_directories(self):
        """Create necessary directories"""
        Path(self.base_config['output_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.base_config['log_dir']).mkdir(parents=True, exist_ok=True)

    def get_cache_dirs(self):
        """Interactively get cache directories from user"""
        cache_dirs = []
        print("\nEnter cache directories (one per line).")
        print("Press Enter twice to finish input:")
        
        while True:
            cache_dir = input().strip()
            if not cache_dir:
                break
            
            path = Path(cache_dir)
            if not path.exists():
                print(f"Warning: Directory {cache_dir} does not exist. Include anyway? (y/n)")
                if input().lower() != 'y':
                    continue
            
            cache_dirs.append(cache_dir)
        
        if not cache_dirs:
            raise ValueError("At least one cache directory is required.")
        
        return cache_dirs

    def get_thresholds(self):
        """Interactively get similarity thresholds from user"""
        while True:
            try:
                print("\nEnter similarity thresholds (comma-separated, between 0 and 1)")
                print("Example: 0.7,0.8,0.9")
                thresholds_input = input().strip()
                thresholds = [float(t) for t in thresholds_input.split(',')]
                
                if any(t < 0 or t > 1 for t in thresholds):
                    print("Error: Thresholds must be between 0 and 1")
                    continue
                    
                return thresholds
                
            except ValueError:
                print("Error: Invalid input. Please enter numbers separated by commas.")

    def run_experiments(self, interactive=True):
        """Run all experiments"""
        if interactive:
            cache_dirs = self.get_cache_dirs()
            similarity_thresholds = self.get_thresholds()
        else:
            cache_dirs = ['cache']
            similarity_thresholds = [0.7, 0.8, 0.9]
            self.logger.info("Running with default values in non-interactive mode")

        # Save experiment configuration
        config = {
            **self.base_config,
            'cache_dirs': cache_dirs,
            'similarity_thresholds': similarity_thresholds,
        }
        
        # Show configuration summary
        print("\nConfiguration Summary:")
        print(json.dumps(config, indent=2))
        
        if interactive:
            print("\nProceed with experiments? (y/n)")
            if input().lower() != 'y':
                print("Experiment cancelled.")
                return

        # Initialize trainer
        trainer = ContractTrainer(config)
        
        # Run experiments
        results = []
        experiment_id = 0
        successful_experiments = 0
        total_experiments = len(cache_dirs) * len(similarity_thresholds)
        
        for cache_dir, threshold in product(cache_dirs, similarity_thresholds):
            experiment_id += 1
            self.logger.info(f"\nStarting experiment {experiment_id}/{total_experiments}")
            self.logger.info(f"Cache dir: {cache_dir}")
            self.logger.info(f"Similarity threshold: {threshold}")
            
            # Update config for this experiment
            experiment_config = {**config, 'similarity_threshold': threshold}
            trainer.config = experiment_config
            
            # Run training
            success, files = trainer.train(cache_dir, experiment_id)
            if success:
                successful_experiments += 1
                results.append({
                    'experiment_id': experiment_id,
                    'cache_dir': cache_dir,
                    'threshold': threshold,
                    'files': files
                })
        
        # Log summary
        self.logger.info("\nExperiment Summary:")
        self.logger.info(f"Total experiments: {total_experiments}")
        self.logger.info(f"Successful experiments: {successful_experiments}")
        self.logger.info(f"Failed experiments: {total_experiments - successful_experiments}")
        
        return results

def main():
    # Base configuration
    base_config = {
        'output_dir': 'experiments',
        'log_dir': 'logs',
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001
    }
    
    # Create runner and run experiments
    runner = ExperimentRunner(base_config)
    results = runner.run_experiments(interactive=True)
    
    # Process results as needed
    print("\nExperiment Results:")
    for result in results:
        print(f"\nExperiment {result['experiment_id']}:")
        print(f"Cache dir: {result['cache_dir']}")
        print(f"Threshold: {result['threshold']}")
        print(f"Output files: {result['files']}")

if __name__ == "__main__":
    main()