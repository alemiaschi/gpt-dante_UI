import logging

import numpy as np
import torch

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop                                                                                                                                                                       \
                                                                                                                                                                                                                                              

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size                                                                                                                                                                 \
                                                                                                                                                                                                                                              
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop                                                                                                                                                                                           \
                                                                                                                                                                                                                                              
    return length

set_seed(21)

    # Initialize the model and tokenizer                                                                                                                                                                                                     \
                                                                                                                                                                                                                                              
try:
    # args.model_type = args.model_type.lower()                                                                                                                                                                                               
    model_class, tokenizer_class = MODEL_CLASSES['gpt2']
except KeyError:
    raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

tokenizer = tokenizer_class.from_pretrained('LorenzoDeMattei/GePpeTto')
tokenizer.add_tokens('\n')
model = model_class.from_pretrained('LorenzoDeMattei/GePpeTto')
model.resize_token_embeddings(len(tokenizer))
#model.load_state_dict(torch.load('/home/alessio/Documents/PhD/experiments/geppetto_dante/GePpeTto-web-master/GePpeTto_terza_rima-esteso.chkpt',
#                                 map_location=torch.device('cpu')))
model.eval()


length = adjust_length_to_model(36, max_sequence_length=model.config.max_position_embeddings)


def generate(prompt):  # noqa: E501
    """generate

    Generate text from prompt # noqa: E501

    :param prompt: 
    :type prompt: dict | bytes

    :rtype: GeneratedText
    """
    
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to("cpu")

    output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=100 + len(encoded_prompt[0]),
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=1,
    )

    # Remove the batch dimension when returning multiple sequences                                                                                                                                                                            
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))                                                                                                                                                            
        generated_sequence = generated_sequence.tolist()

        # Decode text                                                                                                                                                                                                                         
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token                                                                                                                                                                                                
        text = text[: text.find('<|endoftext|>')]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing                                                                                                                            
        total_sequence = (
            prompt + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        # generated_sequences.append(total_sequence)
        # print(total_sequence)                                                                                                                                                                                                               
        return total_sequence + '...'
