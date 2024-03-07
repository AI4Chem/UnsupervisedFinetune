import torch
import transformers
import accelerate
import copy

from transformers import GemmaForCausalLM
from transformers import AutoTokenizer




tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = GemmaForCausalLM.from_pretrained("google/gemma-2b-it")

inputs = tokenizer("我爱吃牛肉",return_tensors='pt')
print(tokenizer.batch_decode(model(**inputs).logits.argmax(dim=-1)))

def dpo_loss(y_l,y_w,x,model,model_ref,beta):
    A = torch.log(conditional_sampling(model,x,y_w)/conditional_sampling(model_ref,x,y_w))
    B = torch.log(conditional_sampling(model,x,y_l)/conditional_sampling(model_ref,x,y_l))
    final = torch.log(torch.sigma(beta*(A-B)))
    return final

def conditional_sampling(model,x,y):
    '''
    x, ids
    y,ids
    '''
    logits = model(input_ids=x)
    return _get_batch_logps(logits,y,True)

def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    tokenizer=None
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != tokenizer.label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == tokenizer.label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def templating(prompt,answer='',history=None,system=None):
    system = f"<bos>system\n{system}\n" if system is not None else '<bos>'
    history = "\n".join([f'<start_of_turn>user\n{qa[0]}<end_of_turn>\n<start_of_turn>model\n{qa[1]}<end_of_turn>' for qa in history] ) if history else ""
    prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>"
    # answer = f"\n{answer}<end_of_turn>"
    ret = system + history + prompt + '\n<start_of_turn>model'
    return ret,answer



def get_weak_answer(tokenizer,model,prompt):
    prompt = f"<question>\n{prompt}\n<answer this question>"
    input,_ = templating(prompt)
    inputs = tokenizer.encode(input, add_special_tokens=False, return_tensors="pt")
    outputs = tokenizer.decode(model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)[0])
    outputs = outputs[len(input):].replace('<eos>','')
    return outputs

def get_weak_hint(tokenizer,model,prompt,answer):
    prompt = f"<question>\n{prompt}\n<weak answer>\n{answer}\n<generate a hint to help answer this question better>"
    input,_ = templating(prompt)
    inputs = tokenizer.encode(input, add_special_tokens=False, return_tensors="pt")
    outputs = tokenizer.decode(model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)[0])
    outputs = outputs[len(input):].replace('<eos>','')
    return outputs

def get_better_answer_with_hint(tokenizer,model,prompt,hint):
    prompt = f"<question>\n{prompt}\n<hint>\n{hint}\n<answer this question according to hints>"
    input,_ = templating(prompt)
    inputs = tokenizer.encode(input, add_special_tokens=False, return_tensors="pt")
    outputs = tokenizer.decode(model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)[0])
    outputs = outputs[len(input):].replace('<eos>','')
    return outputs


def get_better_hint_with_ansers(tokenizer,model,prompt,answer1,answer2):
    prompt = f"<question>\n{prompt}\n<weak answer>\n{answer1}\n<better answer>\n{answer2}\n<generate a hint help to answer this question better according to answers>"
    input,_ = templating(prompt)
    inputs = tokenizer.encode(input, add_special_tokens=False, return_tensors="pt")
    outputs = tokenizer.decode(model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)[0])
    outputs = outputs[len(input):].replace('<eos>','')
    return outputs

def get_answers_and_hints(tokenizer,model,prompt):
    weak_answer = get_weak_answer(tokenizer,model,prompt)
    weak_hint = get_weak_hint(tokenizer,model,prompt,weak_answer)
    better_answer = get_better_answer_with_hint(tokenizer,model,prompt,weak_hint)
    better_hint = get_better_hint_with_ansers(tokenizer,model,prompt,weak_answer,better_answer)
    prompt_for_hint = f"<question>\n{prompt}\n<generate a hint to help answer this question>"
    answer_3let = {'prompt':prompt,'bad':weak_answer,'good':better_answer}
    hint_3let = {'prompt':prompt_for_hint,'bad':weak_hint,'good':better_hint}
    return answer_3let,hint_3let

print(templating("我爱你","你好"))
def chat(prompt,answer,history=None,system=None,loss=False):
    input,output = templating(prompt,answer,history,system)
    print(input)
    inputs = tokenizer(input,return_tensors='pt')
    outputs = tokenizer(output,return_tensors='pt')
    outputs = model(**inputs)
    # print(outputs[0])
    response = tokenizer.batch_decode(model(**inputs)[0].argmax(dim=-1))
    print(response)
    if history is not None:
        history.append([prompt,response])
    if loss:
        loss = outputs.loss
        return history,outputs,loss
    else:
        return history,outputs

# a = chat("How does the brain work?!","你好",loss=True)
# print(a[0])
a = get_answers_and_hints(tokenizer,model,'how does brains work?')
print(a)