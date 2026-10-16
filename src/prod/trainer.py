"""ProD objective (paper Eq. 1 + SFT anchor):

    L = -log sigmoid( beta/|tau| log p(tau|x) - beta/|tau~| log p(tau~|x) - f(tau~, tau) )  -  lambda/|tau| log p(tau|x)
    f(tau~, tau) = c0 * ( alpha * max(0, dS(tau~, tau)) + (1 - alpha) * 1[Ans(tau~) != Ans(tau)] )

Ablations (Sec. 4.5): --fixed_margin gives "w/o AM"; adding --lambda_sft 0 gives "w/o SFT + AM".

tau is the teacher trajectory, tau~ the student's own rollout, dS the stagnation gap (Eq. 2).
Log-likelihoods are length-normalized and computed on assistant (thought + action) tokens only.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Trainer
from trl import SFTTrainer
from trl.trainer.utils import selective_log_softmax


class ProDTrainer(SFTTrainer):
    def __init__(self, *args, beta=2.5, c0=0.5, alpha=0.5, lambda_sft=1.0, fixed_margin=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_margin = fixed_margin
        self.beta = beta
        self.c0 = c0
        self.alpha = alpha
        self.lambda_sft = lambda_sft

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "teacher_input_ids", "teacher_attention_mask", "student_input_ids", "student_attention_mask",
                "stagnation_gap", "wrong_answer",
            ]

    @staticmethod
    def get_batch_logps(logits, labels):
        """Length-normalized log-likelihood of each row; even rows are teachers, odd rows students."""
        labels = nn.functional.pad(labels, (0, 1), value=-100)
        labels = labels[..., 1:].contiguous()
        loss_mask = labels != -100
        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        logps = per_token_logps.sum(-1) / loss_mask.sum(-1)
        return logps[::2], logps[1::2]

    def adaptive_margin(self, stagnation_gaps, wrong_answers):
        if self.fixed_margin:  # ablation "w/o AM": a constant margin shared by all pairs
            return torch.full_like(stagnation_gaps, self.c0)
        return self.c0 * (self.alpha * stagnation_gaps + (1 - self.alpha) * wrong_answers)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        stagnation_gaps = inputs.pop("stagnation_gaps")
        wrong_answers = inputs.pop("wrong_answers")

        # plain forward pass (the LM loss it also returns is not used)
        _, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        teacher_logps, student_logps = self.get_batch_logps(outputs.logits, inputs["labels"])

        device = self.accelerator.device
        teacher_rewards = self.beta * teacher_logps.to(device)
        student_rewards = self.beta * student_logps.to(device)
        margin = self.adaptive_margin(stagnation_gaps, wrong_answers)
        pro_losses = -F.logsigmoid(teacher_rewards - student_rewards - margin)
        sft_losses = -teacher_logps
        loss = (pro_losses + self.lambda_sft * sft_losses).mean()

        def gather(t):
            return self.accelerator.gather_for_metrics(t.detach()).float().mean().item()

        self._metrics[mode]["rewards/teacher"].append(gather(teacher_rewards))
        self._metrics[mode]["rewards/student"].append(gather(student_rewards))
        self._metrics[mode]["rewards/margins"].append(gather(teacher_rewards - student_rewards))
        self._metrics[mode]["rewards/accuracies"].append(gather((teacher_rewards > student_rewards).float()))
        self._metrics[mode]["adaptive_margin"].append(gather(margin))
        self._metrics[mode]["loss/pro"].append(gather(pro_losses))
        self._metrics[mode]["loss/sft"].append(gather(sft_losses))

        # Note: the loss is a per-pair mean that the Trainer does not divide by the number of
        # gradient-accumulation steps (the model accepts `num_items_in_batch`), so gradients are summed
        # over the accumulated micro-batches. This matches the paper's runs; the learning rates in
        # Table 7 were tuned under this behavior.
        loss = loss.to(self.args.device)
        return (loss, outputs) if return_outputs else loss
