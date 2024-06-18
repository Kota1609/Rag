import torch
from unsloth import FastLanguageModel
from peft import AutoPeftModelForCausalLM

load_in_4bit=False
max_seq_length = 3072
dtype = None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Meta-Llama-3-70B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
tokenizer.save_pretrained("llama3_70b_model")
 
# # Load PEFT model on CPU
#model = AutoPeftModelForCausalLM.from_pretrained(
#    "Kota123/ft-Gemma-7b-lora",
#    torch_dtype=torch.float16,
#    low_cpu_mem_usage=True,
#)
# Merge LoRA and base model and save
#merged_model = model.merge_and_unload()
#merged_model.save_pretrained("/home/kota/chat-with-website/ft_new",safe_serialization=True, max_shard_size="2GB")
