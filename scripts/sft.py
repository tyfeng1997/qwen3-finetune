#python scripts/sft.py --config receipes/Qwen3-0.6B-qlora.yaml

from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
import logging
import os
import re
from typing import Optional, List
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import SFTTrainer, TrlParser, ModelConfig, SFTConfig, get_peft_config
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM

# Initialize wandb
import wandb

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    spectrum_config_path: Optional[str] = None
    sample_size: Optional[int] = None  # Added parameter to control sample size
    wandb_project: Optional[str] = None  # Added wandb project name
    wandb_run_name: Optional[str] = None  # Added wandb run name
    wandb_tags: Optional[List[str]] = None  # Added wandb tags

########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

########################
# Helper functions
########################

def get_checkpoint(training_args: SFTConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def setup_model_for_spectrum(model, spectrum_config_path):
    """Configure model for Spectrum training by freezing/unfreezing specific parameters."""
    unfrozen_parameters = []
    with open(spectrum_config_path, "r") as fin:
        yaml_parameters = fin.read()

    # get the unfrozen parameters from the yaml file
    for line in yaml_parameters.splitlines():
        if line.startswith("- "):
            unfrozen_parameters.append(line.split("- ")[1])

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # unfreeze Spectrum parameters
    trainable_param_count = 0
    total_param_count = 0
    
    for name, param in model.named_parameters():
        total_param_count += param.numel()
        if any(re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters):
            param.requires_grad = True
            trainable_param_count += param.numel()
    
    # Log trainable parameter information
    logger.info(f"Total parameters: {total_param_count:,}")
    logger.info(f"Trainable parameters: {trainable_param_count:,} ({100 * trainable_param_count / total_param_count:.2f}%)")
            
    return model

def setup_wandb(model_args, script_args, training_args):
    """Initialize and configure wandb for experiment tracking."""
    if script_args.wandb_project:
        # Setup wandb configuration
        wandb_config = {
            "model_name": model_args.model_name_or_path,
            "dataset": script_args.dataset_id_or_path,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
            "epochs": training_args.num_train_epochs,
            "max_seq_length": training_args.max_seq_length,
            "lora_r": model_args.lora_r if hasattr(model_args, "lora_r") else None,
            "lora_alpha": model_args.lora_alpha if hasattr(model_args, "lora_alpha") else None,
            "use_peft": model_args.use_peft,
            "load_in_4bit": model_args.load_in_4bit,
        }
        
        # Initialize wandb
        run_name = script_args.wandb_run_name or f"{model_args.model_name_or_path.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M')}"
        wandb.init(
            project=script_args.wandb_project,
            name=run_name,
            config=wandb_config,
            tags=script_args.wandb_tags
        )
        logger.info(f"Initialized wandb run: {run_name}")
        return True
    return False

###########################################################################################################

def train_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig):
    """Main training function."""
    #########################
    # Log parameters
    #########################
    logger.info(f'Model parameters {model_args}')
    logger.info(f'Script parameters {script_args}')
    logger.info(f'Training/evaluation parameters {training_args}')

    # Setup wandb
    using_wandb = setup_wandb(model_args, script_args, training_args)

    ###############
    # Load datasets
    ###############
    if script_args.dataset_id_or_path.endswith('.json'):
        train_dataset = load_dataset('json', data_files=script_args.dataset_id_or_path, split='train')
    else:
        train_dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_splits)
    
    # Apply sample size if specified
    if script_args.sample_size:
        train_dataset = train_dataset.select(range(min(script_args.sample_size, len(train_dataset))))
    
    logger.info(f'Loaded dataset with {len(train_dataset)} samples and the following features: {train_dataset.features}')
    
    # Log dataset statistics to wandb if enabled
    if using_wandb:
        wandb.log({"dataset_size": len(train_dataset)})
        # Log a few example samples
        if len(train_dataset) > 0:
            examples = train_dataset.select(range(min(3, len(train_dataset))))
            for i, example in enumerate(examples):
                wandb.log({f"example_{i}": example})
    
    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    # Log tokenizer details
    logger.info(f"Tokenizer: {tokenizer.__class__.__name__}")
    logger.info(f"Vocab size: {len(tokenizer)}")
    
    #######################
    # Load pretrained model
    #######################
    
    # define model kwargs
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype if model_args.torch_dtype in ['auto', None] else getattr(torch, model_args.torch_dtype),
        use_cache=False if training_args.gradient_checkpointing else True,
        low_cpu_mem_usage=True if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) else None,
    )
    
    # Check which training method to use and if 4-bit quantization is needed
    if model_args.load_in_4bit: 
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
            bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
        )
    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
        logger.info(f"PEFT config: {peft_config}")
    else:
        peft_config = None
    
    # Load the model with our kwargs
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    start_time = datetime.now()
    
    if training_args.use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        
    training_args.distributed_state.wait_for_everyone()
    
    load_time = datetime.now() - start_time
    logger.info(f"Model loaded in {load_time.total_seconds():.2f} seconds")

    if script_args.spectrum_config_path:
        logger.info(f"Setting up model for Spectrum training with config: {script_args.spectrum_config_path}")
        model = setup_model_for_spectrum(model, script_args.spectrum_config_path)

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')

    logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***')
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Log metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()
    
    # Log final metrics to wandb
    if using_wandb:
        wandb.log(metrics)

    ##################################
    # Save model and create model card
    ##################################
    
    logger.info('*** Save model ***')
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
    
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f'Model saved to {training_args.output_dir}')
    training_args.distributed_state.wait_for_everyone()

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f'Tokenizer saved to {training_args.output_dir}')

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({'tags': ['sft', 'qwen', 'qlora', 'fine-tuning']})
    
    # Push to hub if needed
    if training_args.push_to_hub is True:
        logger.info('Pushing to hub...')
        trainer.push_to_hub()

    logger.info('*** Training complete! ***')
    
    # Finish wandb run
    if using_wandb:
        wandb.finish()


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Add wandb to reporting tools if specified
    if script_args.wandb_project and "wandb" not in training_args.report_to:
        training_args.report_to.append("wandb")

    # Run the main training loop
    train_function(model_args, script_args, training_args)


if __name__ == '__main__':
    main()