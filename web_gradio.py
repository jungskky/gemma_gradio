from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import random
import torch
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=str, required=False, default='google/gemma-2b-it')
parser.add_argument("--model", type=str, required=False, default='HuggingFaceH4/zephyr-7b-gemma-v0.1')
parser.add_argument("--device",
                    type=str,
                    default="cpu",
                    choices=["cpu", "cuda", "mps"])
parser.add_argument("--float16", action='store_true')
parser.add_argument("--quant", type=int, required=False, default=0, choices=[0, 4, 8])
args = parser.parse_args()

device = torch.device(args.device)

# Load model weights
tokenizer = AutoTokenizer.from_pretrained(args.model)
if args.float16:
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
elif args.quant == 4:
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", quantization_config=quantization_config)
elif args.quant == 8:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", quantization_config=quantization_config)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
model = model.eval()


def format_prompt(message, history):
    prompt = ""
    if history:
        for user_prompt, bot_response in history:
            prompt += f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
            prompt += f"<start_of_turn>model\n{bot_response}<end_of_turn>\n"
    prompt += f"<start_of_turn>user\n{message}<end_of_turn>\n<start_of_turn>model\n"
    return prompt


def chat_inf(system_prompt, prompt, history, seed, temp, tokens, top_p, rep_p):
    if not history:
        history = []
        hist_len = 0
    if history:
        hist_len = len(history)
        print(hist_len)

    random.seed(seed)
    torch.manual_seed(seed)

    generate_kwargs = dict(
        temperature=temp,
        max_new_tokens=tokens,
        top_p=top_p,
        repetition_penalty=rep_p,
        do_sample=True,
        # seed=seed,
    )
    formatted_prompt = format_prompt(f"{system_prompt}, {prompt}", history)
    print("formatted_prompt")
    print(formatted_prompt)
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**input_ids, **generate_kwargs)
    output = tokenizer.decode(outputs[0])
    output = output.split("<start_of_turn>model\n")[-1][:-5]
    print("output")
    print(output)
    print("")

    history.append((prompt, output))
    yield history


def clear_fn():
    return None


rand_val = random.randint(1, 1111111111111111)


def check_rand(inp, val):
    if inp is True:
        return gr.Slider(label="Seed", minimum=1, maximum=1111111111111111, value=random.randint(1, 1111111111111111))
    else:
        return gr.Slider(label="Seed", minimum=1, maximum=1111111111111111, value=int(val))


with gr.Blocks() as demo:
    gr.HTML(
        """<center><h1 style='font-size:xx-large;'>Google Gemma Models</h1></center>""")

    chat_b = gr.Chatbot(height=500)
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    rand = gr.Checkbox(label="Random Seed", value=True)
                    seed = gr.Slider(label="Seed", minimum=1, maximum=1111111111111111, step=1, value=rand_val)
                    tokens = gr.Slider(label="Max new tokens", value=6400, minimum=0, maximum=8000, step=64,
                                       interactive=True, visible=True, info="The maximum number of tokens")
            with gr.Column(scale=1):
                with gr.Group():
                    temp = gr.Slider(label="Temperature", step=0.01, minimum=0.01, maximum=1.0, value=0.9)
                    top_p = gr.Slider(label="Top-P", step=0.01, minimum=0.01, maximum=1.0, value=0.9)
                    rep_p = gr.Slider(label="Repetition Penalty", step=0.1, minimum=0.1, maximum=2.0, value=1.0)

    with gr.Group():
        with gr.Row():
            with gr.Column(scale=3):
                sys_inp = gr.Textbox(label="System Prompt (optional)")
                inp = gr.Textbox(label="Prompt")
                with gr.Row():
                    btn = gr.Button("Chat")
                    stop_btn = gr.Button("Stop")
                    clear_btn = gr.Button("Clear")

    chat_sub = inp.submit(check_rand, [rand, seed], seed).then(chat_inf,
                                                               [sys_inp, inp, chat_b, seed, temp, tokens,
                                                                top_p, rep_p], chat_b)
    go = btn.click(check_rand, [rand, seed], seed).then(chat_inf,
                                                        [sys_inp, inp, chat_b, seed, temp, tokens, top_p,
                                                         rep_p], chat_b)
    stop_btn.click(None, None, None, cancels=[go, chat_sub])
    clear_btn.click(clear_fn, None, [chat_b])

demo.queue(default_concurrency_limit=10).launch(share=True, inbrowser=False)
