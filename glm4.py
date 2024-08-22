#注意: glm4  transformers==4.41.2 测试下来高了也不行和joy_caption冲突

import os
import torch
import folder_paths
import json

from transformers import AutoModelForCausalLM,AutoTokenizer, pipeline
from transformers import AutoProcessor

from PIL import Image
import numpy as np

# from .lib.ximg import *  

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class CXH_GLM_PIPE:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

class CXH_GLM4_load:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["nikravan/glm-4vq"],),
                "attention": ([ 'flash_attention_2', 'sdpa', 'eager'],{"default": 'eager'})
            },
        }

    RETURN_TYPES = ("CXH_GLM_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "gen"
    CATEGORY = "CXH/LLM"

    def gen(self,model,attention):
        # 下载本地
        model_id = model
        model_checkpoint = os.path.join(folder_paths.models_dir, 'LLM', os.path.basename(model_id))
        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
            
        causel_model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            # _attn_implementation= attention
        )
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

        glm4_model = CXH_GLM_PIPE()
        glm4_model.model = causel_model
        glm4_model.tokenizer = tokenizer
    
        return (glm4_model,)

class CXH_GLM4_Run:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("CXH_GLM_PIPE",),
                "image": ("IMAGE",),
                "prompt": ("STRING",{"default": '', "multiline": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 100, "max": 10000, "step": 500}),
                "device":(["cuda","cpu"],{"default":"cuda"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out",)
    FUNCTION = "gen"
    CATEGORY = "CXH/LLM"

    def gen(self,pipe,image,prompt,max_new_tokens,device):
        model = pipe.model
        tokenizer = pipe.tokenizer

        image = tensor2pil(image)

        inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": prompt}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)
        
        inputs = inputs.to(device)
        gen_kwargs = {"max_length": max_new_tokens, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            resp = tokenizer.decode(outputs[0])
            resp = resp.replace("<|endoftext|>","")

        return (resp,)

  