import json
import os
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union
from contextlib import contextmanager, nullcontext
import torch.nn.functional as F
import warnings
from copy import deepcopy
from trl.models import PreTrainedModelWrapper, create_reference_model
import deepspeed

from transformers import PreTrainedModel
from trl.trainer.utils import pad_to_length

import numpy as np
import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger

from ..dpo.collator import DPODataCollatorWithPadding
from ...extras.constants import IGNORE_INDEX

from accelerate import Accelerator
accelerator = Accelerator()
if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput

from datasets import Dataset
from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
from transformers import BatchEncoding, Trainer
from trl import DPOTrainer
from trl.trainer.utils import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from transformers import PreTrainedModel

from trl.import_utils import is_peft_available
if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


logger = get_logger(__name__)

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss





class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """
    def set_template(self,template,data_args,model,beta,loss_type,ftx_gamma,ref_model=None,generation_config={}):   
        self.tokenizer.padding_side = 'left'
        self.template = template
        self.template_seperator_left = self.template.format_user.slots[0].split('{{content}}')[0]
        self.template_seperator_right = self.template.format_user.slots[0].split('{{content}}')[1]
        self.teacherforce_rate = -1
        self.second_collator = DPODataCollatorWithPadding(
        tokenizer=self.tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id)
        self.use_dpo_data_collator = True
        disable_dropout = True
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_adapter_name = None

        self.ref_model = ref_model
        self.beta = beta
        self.label_smoothing = 0
        self.loss_type = loss_type
        self.ftx_gamma = ftx_gamma
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)

        self.generation_config = generation_config

        # Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for label, pred in zip(decoded_labels, decoded_preds):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))

    @torch.no_grad()
    def extract_prompt(self,input_ids):
        inputs = self.tokenizer.batch_decode(input_ids,clean_up_tokenization_spaces=True,)
        prefixs = []
        prompts = []
        orig_ans = []
        for i in inputs:
            prompt = i.split(self.template_seperator_left)[-1].split(self.template_seperator_right)[0]
            prefix = i[:i.find(prompt)]
            ans = i[i.find(prompt)+len(prompt)+len(self.template_seperator_right):i.find(self.tokenizer.eos_token)]
            prompts.append(prompt)
            prefixs.append(prefix)
            orig_ans.append(ans)
        return prompts,prefixs,orig_ans

    @torch.no_grad()
    def get_weak_answer(self,model,prefixs,prompts,max_new_tokens):
        input = [f"{prefix}<question>\n{prompt}\n<answer this question>{self.template_seperator_right}" for prefix,prompt in zip(prefixs,prompts)]
        inputs = self.tokenizer(input, add_special_tokens=True, return_tensors="pt",padding=True)
        inputs = self._prepare_inputs(inputs)
        outputs = self.tokenizer.batch_decode(self.accelerator.unwrap_model(model).generate(**inputs,num_return_sequences=1,max_new_tokens=max_new_tokens,do_sample=True,repetition_penalty=2.0,num_beams=2,low_memory=True,bos_token_id=self.tokenizer.bos_token_id,eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id))
        outputs = [output[len(i):].replace('<pad>','') for output,i in zip(outputs,input)]
        return outputs

    @torch.no_grad()
    def get_weak_hint(self,model,prefixs,prompts,max_new_tokens,answers):
        input = [f"{prefix}<question>\n{prompt}\n<weak answer>\n{answer}\n<generate a hint to help answer this question better>{self.template_seperator_right}" for prefix,prompt,answer in zip(prefixs,prompts,answers)]
        inputs = self.tokenizer(input, add_special_tokens=True, return_tensors="pt",padding=True)
        inputs = self._prepare_inputs(inputs)
        outputs = self.tokenizer.batch_decode(self.accelerator.unwrap_model(model).generate(**inputs,num_return_sequences=1,max_new_tokens=max_new_tokens,do_sample=True,repetition_penalty=2.0,num_beams=2,low_memory=True,bos_token_id=self.tokenizer.bos_token_id,eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id))
        outputs = [output[len(i):].replace('<pad>','') for output,i in zip(outputs,input)]
        return outputs

    @torch.no_grad()
    def get_better_answer_with_hint(self,model,prefixs,prompts,max_new_tokens,hints):
        input = [f"{prefix}<question>\n{prompt}\n<hint>\n{hint}\n<answer this question according to hints>{self.template_seperator_right}" for prefix,prompt,hint in zip(prefixs,prompts,hints)]
        inputs = self.tokenizer(input, add_special_tokens=True, return_tensors="pt",padding=True)
        inputs = self._prepare_inputs(inputs)
        outputs = self.tokenizer.batch_decode(self.accelerator.unwrap_model(model).generate(**inputs,num_return_sequences=1,max_new_tokens=max_new_tokens,do_sample=True,repetition_penalty=2.0,num_beams=2,low_memory=True,bos_token_id=self.tokenizer.bos_token_id,eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id))
        outputs = [output[len(i):].replace('<pad>','') for output,i in zip(outputs,input)]
        return outputs

    @torch.no_grad()
    def get_better_hint_with_ansers(self,model,prefixs,prompts,max_new_tokens,answer1s,answer2s):
        input = [f"{prefix}<question>\n{prompt}\n<weak answer>\n{answer1}\n<better answer>\n{answer2}\n<generate a hint to help answer this question step-by-step better according to answers>{self.template_seperator_right}" for prefix,prompt,answer1,answer2 in zip(prefixs,prompts,answer1s,answer2s)]
        inputs = self.tokenizer(input, add_special_tokens=True, return_tensors="pt",padding=True)
        inputs = self._prepare_inputs(inputs)
        outputs = self.tokenizer.batch_decode(self.accelerator.unwrap_model(model).generate(**inputs,num_return_sequences=1,max_new_tokens=max_new_tokens,do_sample=True,repetition_penalty=2.0,num_beams=2,low_memory=True,bos_token_id=self.tokenizer.bos_token_id,eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id))
        outputs = [output[len(i):].replace('<pad>','') for output,i in zip(outputs,input)]
        return outputs

    def safegate(self,strs):
        count = 0
        p = 11
        for s in strs:
            for i in range(p - 1):
                a,b = random.choice(s[len(s)//p * i:len(s)//p * (i+1)]),random.choice(s[len(s)//p * (i+1):len(s)//p * (i+2)])
                if a == b:
                    count += 1
        return count / (len(strs) * p)

    @torch.no_grad()
    def get_answers_and_hints(self,model,prefixs,prompts,max_new_tokens,orig_ans=None,teacherforce=True):
        weak_answers = self.get_weak_answer(model,prefixs,prompts,max_new_tokens)
        weak_hints = self.get_weak_hint(model,prefixs,prompts,max_new_tokens,weak_answers)
        repetition_flag = False#self.safegate(weak_answers) > 0.1 or self.safegate(weak_hints) > 0.1 or any(['<question>' in i or 'answer>' in i for i in weak_answers]) or any(['<question>' in i or '<question>' in i for i in weak_hints])
        if (teacherforce or repetition_flag) and orig_ans is not None:
            better_answers = orig_ans
            # better_hints = self.get_better_hint_with_ansers(model,prefixs,prompts,max_new_tokens,weak_answers,better_answers)
        else:
            better_answers = self.get_better_answer_with_hint(model,prefixs,prompts,max_new_tokens,weak_hints)
        if repetition_flag and orig_ans is not None:
            better_hints = ['Sure,You can answer this question like ' + i for i in orig_ans]
        else:
            better_hints = self.get_better_hint_with_ansers(model,prefixs,prompts,max_new_tokens,weak_answers,better_answers)
        prompt_for_hints = [f"{prefix}<question>\n{prompt}\n<generate a hint to help answer this question>{self.tokenizer.eos_token}" for prefix,prompt in zip(prefixs,prompts)]
        prompts = prompts + prompt_for_hints
        rejected = weak_answers+weak_hints
        accepted = better_answers+better_hints
        if self.accelerator.is_main_process:
            print("\n#########\n".join((prompts[0],weak_answers[0],better_answers[0],weak_hints[0],better_hints[0])))
        return prompts,rejected,accepted

    def _pad_labels(self, batch: torch.Tensor, positions: List[Tuple[int, int]]) -> torch.Tensor:
        padded_labels = []
        for feature, (prompt_len, answer_len) in zip(batch, positions):
            if self.tokenizer.padding_side == "left":
                start, end = feature.size(0) - answer_len, feature.size(0)
            else:
                start, end = prompt_len, prompt_len + answer_len
            padded_tensor = self.label_pad_token_id * torch.ones_like(feature)
            padded_tensor[start:end] = feature[start:end]
            padded_labels.append(padded_tensor)
        return torch.stack(padded_labels, dim=0).contiguous()  # in contiguous memory

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.eval()
        ######genrate self-improving from hints examples########
        input_ids = inputs.input_ids
        # labels = inputs.labels
        # print(prompts,labels,)
        prompts,prefixs,orig_ans = self.extract_prompt(input_ids)
        # print(prompts,prefixs,orig_ans)
        max_new_tokens = min(max([len(i) for i in orig_ans])+1,300)
        teacherforce = random.uniform(0,1) < self.teacherforce_rate
        prompts,rejected,accepted = self.get_answers_and_hints(model,prefixs,prompts,max_new_tokens,orig_ans,teacherforce)
        promptandaccepted = [i+self.template_seperator_right+j for i,j in zip(prompts,accepted)]
        promptandrejected = [i+self.template_seperator_right+j for i,j in zip(prompts,rejected)]
        inputs = promptandaccepted + promptandrejected
        acceptedlabels = [j for i,j in zip(prompts,accepted)]
        rejectedlabels = [j for i,j in zip(prompts,rejected)]
        labels = acceptedlabels + rejectedlabels
        # prompt_ids = self.tokenizer(prompts, add_special_tokens=True, return_tensors="pt",padding=True)
        input_encoded = self.tokenizer(inputs, add_special_tokens=True, return_tensors="pt",padding=True)
        labels = self.tokenizer(labels, add_special_tokens=False, return_tensors="pt",padding=True).input_ids
        labels_length = [sum(i != self.tokenizer.pad_token_id ) for i in labels]
        labels = input_encoded.input_ids.detach().clone()
        for i in range(len(labels)):
            labels[i][:-labels_length[i]] = self.label_pad_token_id
        ########################################
        # inputs = [{'prompt_ids':prompt_id,'rejected_ids':rejected_id,'chosen_ids':chosen_id} for prompt_id,rejected_id,chosen_id in zip(prompt_ids,rejected_ids,chosen_ids)]
        # inputs = self.second_collator(inputs)
        length = len(input_encoded.input_ids)
        
        inputs = {'rejected_input_ids':input_encoded.input_ids[length//2:],'rejected_labels':labels[length//2:],
                  'rejected_attention_mask':input_encoded.attention_mask[length//2:],'chosen_input_ids':input_encoded.input_ids[:length//2],
                  'chosen_labels':labels[:length//2],'chosen_attention_mask':input_encoded.attention_mask[:length//2]}
        
        # print({k:v.shape for k,v in inputs.items()})
        inputs = self._prepare_inputs(inputs)
        # print(inputs)

        model.train()
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )


        return concatenated_batch

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]],loss=False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        self.is_encoder_decoder = self.accelerator.unwrap_model(model).config.is_encoder_decoder
        self.label_pad_token_id = self.data_collator.label_pad_token_id
        self.padding_value = self.tokenizer.pad_token_id
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        # print(concatenated_batch["concatenated_input_ids"].dtype)

        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )


        ret = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )

        all_logits = ret.logits
        if loss:
            all_loss = []
            all_labels = concatenated_batch["concatenated_labels"]
            # print(all_labels.shape,all_logits.shape,)
            for logits,labels in zip([all_logits[:all_logits.size(0)//2],all_logits[all_logits.size(0)//2:]],[all_labels[:all_labels.size(0)//2],all_labels[all_labels.size(0)//2:]]):
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.accelerator.unwrap_model(model).config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                all_loss.append(loss)
        

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if loss:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits,all_loss)
        else: 
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            all_loss
        ) = self.concatenated_forward(model, batch,loss=True)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean()+all_loss[0]/all_loss[1], metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        # compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with self.compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)