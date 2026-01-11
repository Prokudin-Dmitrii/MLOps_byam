import os
import json
import torch
from ts.torch_handler.base_handler import BaseHandler
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

class TextGenerationHandler(BaseHandler):
    def initialize(self, context):
        self.device = torch.device('cpu')

        model_dir = context.system_properties['model_dir']

        model_path = os.path.join(model_dir, 'model')
        tokenizer_path = os.path.join(model_dir, 'data')

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(tokenizer_path, 'tokenizer')
        )

        self.model = GPT2LMHeadModel.from_pretrained(
            os.path.join(model_path, 'byam_step_final')
        )

        self.model.to(self.device)
        self.model.eval()

        self.max_length = 50
        self.do_sample = True
        self.top_p = 0.9
        self.temperature = 0.9
        self.repetition_penalty = 1.2

    def preprocess(self, data):
        '''
        Expected input JSON:
        {
          'text': 'Počasí v Praze ',
          'max_length': 50,             
          'do_sample': true,            
          'top_p': 0.9,
          'temperature': 0.9,          
          'repetition_penalty': 1.2
        }
        '''

        body = data[0].get('body')

        if isinstance(body, (bytes, bytearray)):
            body = body.decode('utf-8')

        if isinstance(body, str):
            body = json.loads(body)

        text = body['text']

        self.max_length = body.get('max_length', self.max_length)
        self.do_sample = body.get('do_sample', self.do_sample)
        self.top_p = body.get('top_p', self.top_p)
        self.temperature = body.get('temperature', self.temperature)
        self.repetition_penalty = body.get(
            'repetition_penalty', self.repetition_penalty
        )

        inputs = self.tokenizer(
            text,
            return_tensors='pt'
        )

        return inputs['input_ids'].to(self.device)

    def inference(self, input_ids):
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_length,
                do_sample=self.do_sample,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty
            )
        return outputs

    def postprocess(self, outputs):
        texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        return [{'generated_text': t} for t in texts]
