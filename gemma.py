from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=False, default='google/gemma-2b-it')
parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda", "mps"])
args = parser.parse_args()

rand_val = random.randint(1, 1111111111111111)
random.seed(rand_val)
torch.manual_seed(rand_val)

device = torch.device(args.device)

# Load model weights
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
model = model.eval()

# Chat templates
USER_CHAT_TEMPLATE = '<start_of_turn>user\n{prompt}<end_of_turn>\n'
MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n{prompt}<end_of_turn>\n'

# Sample formatted prompt
prompt = (
    USER_CHAT_TEMPLATE.format(
        prompt='What is a good place for travel in the US?'
    )
    + MODEL_CHAT_TEMPLATE.format(prompt='California.')
    + USER_CHAT_TEMPLATE.format(prompt='What can I do in California?')
    + '<start_of_turn>model\n'
)
print(USER_CHAT_TEMPLATE.format(prompt=prompt))

start_time = time.time()

input_ids = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**input_ids, max_length=1000)
output = tokenizer.decode(outputs[0])
output = output.split("<start_of_turn>model\n")[-1][:-5]
print(output)

print(f'Elapsed time: {time.time() - start_time:.2f} seconds')
