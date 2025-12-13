"""LLaMA-3.1-8B classifier."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForSequenceClassification

from .. import ModelBase, register_model


@register_model("llama3.1-8b")
@register_model("llama")
class Llama31SequenceClassifier(ModelBase):
    def __init__(
        self,
        num_labels: int = 2,
        checkpoint: str = "meta-llama/Meta-Llama-3.1-8B",
        torch_dtype=None,
        device_map=None,
    ) -> None:
        super().__init__(
            name="llama3.1-8b",
            task_type="text",
            num_labels=num_labels,
            description=f"LLaMA-3.1-8B sequence classifier ({checkpoint})",
        )
        config = AutoConfig.from_pretrained(
            checkpoint,
            num_labels=self.num_labels,
            problem_type="single_label_classification",
        )
        model_kwargs = {"config": config}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            **model_kwargs,
        )

    def forward_features(self, features):  # type: ignore[override]
        outputs = self.backbone(**features)
        return outputs.logits
