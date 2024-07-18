# Chatbot-with-LLM-using-T5
This project involves creating a chatbot using a fine-tuned T5 model. The model has been trained on the OASST1 dataset and is designed to to assist users with various tasks and queries.. The fine-tuned model is uploaded to Hugging Face under the name [KhantKyaw/T5-small_new_chatbot](https://huggingface.co/KhantKyaw/T5-small_new_chatbot). 

![T5](T5.png)

## Table of Contents

- Introduction
- Model 
- Dataset
- Installation
- Usage
- Contributing
- License

## Introduction
This project demonstrates how to fine-tune a T5 (Text-To-Text Transfer Transformer) which is a versatile language model capable of performing a wide range of natural language processing tasks. By fine-tuning T5 on the Open Assistant Conversations (OASST1) dataset, specifically on the training and validation splits (oasst1-train.csv and oasst1-val.csv), we can adapt it to serve as an effective assistant capable of handling user queries and providing useful information.

## Model 
The model used in this project can be found on Hugging Face under the name google-t5/t5-small. This model is a small variant of pre-trained T5 model.T5's approach allows it to seamlessly adapt to various NLP tasks—translation, summarization, question answering, sentiment analysis, and more.

## Dataset
The OASST1 dataset used for fine-tuning contains conversations that simulate interactions between a user and an assistant. The dataset is split into two parts:
- df_train.csv: Training dataset
- df_val.csv: Validation dataset
    
## Installation
To use the assistant chatbot, you'll need to install the required packages. You can do this using pip:

``` python
pip install transformers
pip install torch
pip install sentencepiece
```

## Usage
To use the fine-tuned chatbot model for to generating responses, you can load it using the transformers library:

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
model_name = "KhantKyaw/T5-small_new_chatbot"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids,
                             min_length=5,
                             max_length=300,
                             do_sample=True, num_beams=5, no_repeat_ngram_size=2)
    generated_text = tokenizer.decode(
        outputs[0], skip_special_tokens=True)
    return generated_text
generate_response("how to be healthy?")
```
## Contributing
Contributions are welcome! If you have any ideas, suggestions, or find a bug, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
