import json
import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from chat_sentiment_analysis.utils.prompter import Prompter
from chat_sentiment_analysis.common import common_path
from chat_sentiment_analysis.utils import file_utils

device = "cuda:0"


def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = os.path.join(common_path.project_dir, 'chat-sentiment-analysis'),
    prompt_template: str = "sentiment_analysis",  # The prompt template to use, will default to alpaca.
    data_path: str = os.path.join(common_path.data_dir, 'task_data', 'asote.test.json')
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        # https://huggingface.co/blog/how-to-generate
        # https://huggingface.co/docs/transformers/generation_strategies
        # https://medium.com/mlearning-ai/softmax-temperature-5492e4007f71
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    output_lines = []
    lines = file_utils.read_all_lines(data_path)
    for line in lines:
        line_obj = json.loads(line)
        instruction = line_obj['instruction']
        input = line_obj['input']
        pred = evaluate(instruction, input=input)
        line_obj['pred'] = pred
        output_line = json.dumps(line_obj, ensure_ascii=False)
        output_lines.append(output_line)

    output_filepath = data_path + '.with_pred'
    file_utils.write_lines(output_lines, output_filepath)


if __name__ == "__main__":
    fire.Fire(main)
