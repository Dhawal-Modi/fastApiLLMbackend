import torch
from transformers import Pipeline, pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

prompt = "How to set up a FastAPI project?"

system_prompt = """
You are a FastAPI bot and you are a helpful AI assistant responsible for teaching FastAPI to your users.
Always respond in markdown.
"""


def load_text_model():
    conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        quantization_config=conf,
        trust_remote_code=True,
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #     "microsoft/Phi-3-mini-4k-instruct",
    #     device_map="cuda",
    #     torch_dtype="auto",
    #     trust_remote_code=True,
    # )

    tokenizers = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizers,
    )

    return pipe


def generate_text(pipe: Pipeline, prompt: str, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    predictions = pipe(messages,
                       temperature=temperature,
                       max_new_tokens=256,
                       do_sample=True,
                       top_k=50,
                       top_p=0.95, )

    output = predictions[0]["generated_text"]

    return output[2]["content"]
