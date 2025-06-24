from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CustomizedArguments:
    max_seq_length: int = field(metadata={"help": "Maximum input sequence length"})
    train_file: str = field(metadata={"help": "Training data. If task_type=pretrain, specify a directory; all JSONL files inside will be used"})
    model_name_or_path: str = field(metadata={"help": "Path to pre-trained model or model identifier from huggingface.co/models"})
    template_name: str = field(default="", metadata={"help": "Template name used in SFT for formatting data"})
    eval_file: Optional[str] = field(default="", metadata={"help": "Evaluation dataset file path"})
    max_prompt_length: int = field(default=512, metadata={"help": "Maximum prompt length used in DPO"})
    beta: float = field(default=0.1, metadata={"help": "The beta factor in DPO loss"})
    tokenize_num_workers: int = field(default=10, metadata={"help": "Number of workers used for tokenizing during pretraining"})
    task_type: str = field(default="sft", metadata={"help": "Task type: choose from [pretrain, sft, dpo]"})
    train_mode: str = field(default="qlora", metadata={"help": "Training mode: choose from [full, lora, qlora, dora]"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})