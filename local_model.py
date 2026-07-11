"""Local model wrapper. Auto-detects GPU vs CPU."""
import torch
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class LocalResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

class LocalModel:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=self.device)

    def generate(self, prompt, max_tokens=256, temperature=0.0):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        gen_kwargs = dict(max_new_tokens=max_tokens, pad_token_id=self.tokenizer.eos_token_id)
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        out_ids = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = out_ids[0][prompt_tokens:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return LocalResponse(text=text, prompt_tokens=prompt_tokens, completion_tokens=len(gen_ids))

    def generate_twice_for_agreement(self, prompt, max_tokens=256):
        r1 = self.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        r2 = self.generate(prompt, max_tokens=max_tokens, temperature=0.7)
        return r1, r2
