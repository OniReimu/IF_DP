"""Qwen large language model wrapper."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .. import ModelBase, register_model


@register_model("qwen")
class QwenSequenceClassifier(ModelBase):
    def __init__(
        self,
        num_labels: int = 2,
        checkpoint: str = "Qwen/Qwen2-0.5B",
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__(
            name="qwen",
            task_type="text",
            num_labels=num_labels,
            description=f"Qwen sequence classifier ({checkpoint})",
        )
        config = AutoConfig.from_pretrained(
            checkpoint,
            num_labels=self.num_labels,
            problem_type="single_label_classification",
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
            if pad_token is None:
                pad_token = ""
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
        config.pad_token_id = self.tokenizer.pad_token_id
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            config=config,
            trust_remote_code=trust_remote_code,
        )
        self.backbone.resize_token_embeddings(len(self.tokenizer))

    def forward_features(self, features):  # type: ignore[override]
        if isinstance(features, dict):
            inputs = dict(features)
            inputs.pop("token_type_ids", None)
        else:
            inputs = features
        outputs = self.backbone(**inputs)
        return outputs.logits
