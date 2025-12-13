"""BERT-based classifier for text datasets."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForSequenceClassification

from .. import ModelBase, register_model


@register_model("bert")
class BertTextClassifier(ModelBase):
    def __init__(
        self,
        num_labels: int = 4,
        checkpoint: str = "bert-base-uncased",
    ) -> None:
        super().__init__(
            name="bert",
            task_type="text",
            num_labels=num_labels,
            description=f"BERT sequence classifier ({checkpoint})",
        )
        config = AutoConfig.from_pretrained(
            checkpoint,
            num_labels=self.num_labels,
            problem_type="single_label_classification",
        )
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            config=config,
        )

    def forward_features(self, features):  # type: ignore[override]
        outputs = self.backbone(**features)
        return outputs.logits
