import sys
import unittest

# Absolute import: the installed package is `airllm` (this fork keeps the code under
# air_llm/airllm/), so a relative import beyond the top-level package fails under the
# current layout.
from airllm.auto_model import AutoModel



class TestAutoModel(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_auto_model_should_return_correct_model(self):
        # Since the Phase-2 rearchitecture (hook-based streaming) every STANDARD
        # *ForCausalLM architecture routes through the generic AirLLMBaseModel;
        # only the remote-code custom architectures keep dedicated subclasses.
        mapping_dict = {
            'garage-bAInd/Platypus2-7B': 'AirLLMBaseModel',
            'Qwen/Qwen-7B': 'AirLLMQWen',
            'internlm/internlm-chat-7b': 'AirLLMInternLM',
            'THUDM/chatglm3-6b-base': 'AirLLMChatGLM',
            'baichuan-inc/Baichuan2-7B-Base': 'AirLLMBaichuan',
            'mistralai/Mistral-7B-Instruct-v0.1': 'AirLLMBaseModel',
            'mistralai/Mixtral-8x7B-v0.1': 'AirLLMBaseModel'
        }


        for k,v in mapping_dict.items():
            module, cls = AutoModel.get_module_class(k)
            self.assertEqual(cls, v, f"expecting {v}")

