from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = "/work/xinyu/models/RL-MemoryAgent-7B"
adapter_path = "/mnt/nfs/xinyu/models/memagent/MemAgent_enhance_question_answering_lora"
output_path = f"{adapter_path}_merged"

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="cpu"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, adapter_path)

print("Merging...")
merged = model.merge_and_unload()

print("Saving merged model...")
merged.save_pretrained(output_path, safe_serialization=True)

# save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print(f"Done! Merged model saved to {output_path}")