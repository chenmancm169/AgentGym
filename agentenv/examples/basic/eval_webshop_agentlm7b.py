import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from agentenv.controller import Agent, Evaluator
from agentenv.envs import TextCraftTask

MODEL_PATH = "THUDM/agentlm-7b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True).eval()

evaluator = Evaluator(
    Agent(model, tokenizer),
    [
        TextCraftTask(
            client_args={
                "env_server_base": "http://localhost:36001", # 如果你在前文修改了端口号，请在这里一并修改
                "data_len": 200, # data_len 参数目前没有实际用途，将会在后续开发中重构
                "timeout": 300,
            },
            # n_clients 参数保留用于后续批量生成的实现，现阶段留为 1 即可
            n_clients=1,
        )
    ],
)

exps = evaluator.eval(
    generation_config=GenerationConfig(
        max_length=4096,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    ),
    max_rounds=7,
    idxs=list(range(200)),
)

print("\n\n==== EVALUATION ====\n")
print(f"Score: {exps.score}")
print(f"Success: {exps.success}")

print("\n\n==== EXPERIENCES ====\n")
for idx, exp in enumerate(exps.experiences[:3]):
    print(f"\n\n==== EXP {idx} ====\n")
    for message in exp.conversation:
        if message["from"] == "gpt":
            print(f"\n### Agent\n{message['value']}")
        else:
            print(f"\n### Env\n{message['value']}")
