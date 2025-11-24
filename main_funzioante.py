from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gepa

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval()
seed_prompt = {
    "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
}
trainset, valset, _ = gepa.examples.aime.init_dataset()

print(seed_prompt)

def qwen_chat(messages):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inp = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=64)

    text = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    print(text)
    return text

gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm=qwen_chat,          # <-- adesso funziona
    reflection_lm=qwen_chat,    # <-- anche questo funziona
    max_metric_calls=25,
)
