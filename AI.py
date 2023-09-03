from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 模型路径
model_path = "D:/AI/pypro/BAAI_AquilaCode-multi"

# 加载模型和分词器，并设置trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# 使用模型进行代码补全或回答问题
text = "您想要补全的代码或问题"

tokens = tokenizer.encode_plus(text)['input_ids'][:-1]
tokens = torch.tensor(tokens)[None,]

with torch.no_grad():
    out = model.generate(tokens, do_sample=True, max_length=512, eos_token_id=100007)[0]
    out = tokenizer.decode(out.tolist())
    print(out)
