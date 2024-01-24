from transformers import AutoTokenizer
import transformers 
import torch
model = "./output/503B_FT_lr1e-5_ep5_top1_2023-08-25"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = "Give me detailed info about Jeo Biden."
formatted_prompt = (
    f"### Human: {prompt} ### Assistant:"
)


sequences = pipeline(
    formatted_prompt,
    do_sample=True,
    top_k=50,
    top_p = 0.9,
    num_return_sequences=1,
    repetition_penalty=1.1,
    max_new_tokens=1024,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
