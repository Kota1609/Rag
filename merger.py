import torch
### COMMENT IN TO MERGE PEFT AND BASE MODEL ####
from peft import AutoPeftModelForCausalLM
 
# Load PEFT model on CPU
model = AutoPeftModelForCausalLM.from_pretrained(
    "/home/kota/chat-with-website/llama3_70b_model",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
# Merge LoRA and base model and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("/home/kota/chat-with-website/ft_new",safe_serialization=True, max_shard_size="2GB")
