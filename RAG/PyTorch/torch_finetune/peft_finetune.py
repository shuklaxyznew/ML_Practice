from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType
from torch_finetune.dataset import TRAIN_DATA, EVAL_DATA, SentimentDataset
from torch_finetune.utils import compute_metrics, get_device, print_trainable, run_inference

MODEL_NAME = "distilbert-base-uncased"
INFER_TEXTS = ["Absolutely wonderful!", "Terrible product.", "Pretty good overall."]


def full_finetune():
    print("\n=== Full Fine-Tuning (baseline) ===")
    device    = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    print_trainable(model)

    args = TrainingArguments(
        output_dir="./output/full_ft", num_train_epochs=3,
        per_device_train_batch_size=4, learning_rate=2e-5,
        evaluation_strategy="epoch", report_to="none",
    )
    Trainer(
        model=model, args=args,
        train_dataset=SentimentDataset(TRAIN_DATA, tokenizer),
        eval_dataset=SentimentDataset(EVAL_DATA, tokenizer),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    ).train()

    model.save_pretrained("./output/full_ft/final")
    run_inference(model, tokenizer, device, INFER_TEXTS)


def peft_prompt_tuning():
    print("\n=== PEFT Prompt Tuning ===")
    device    = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text="Classify sentiment",
        tokenizer_name_or_path=MODEL_NAME,
    )
    model = get_peft_model(base, config)
    print_trainable(model)

    args = TrainingArguments(
        output_dir="./output/peft_prompt", num_train_epochs=5,
        per_device_train_batch_size=4, learning_rate=3e-2,
        evaluation_strategy="epoch", report_to="none",
    )
    Trainer(
        model=model, args=args,
        train_dataset=SentimentDataset(TRAIN_DATA, tokenizer),
        eval_dataset=SentimentDataset(EVAL_DATA, tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    ).train()

    model.save_pretrained("./output/peft_prompt/final")
    run_inference(model, tokenizer, device, INFER_TEXTS)


if __name__ == "__main__":
    full_finetune()
    peft_prompt_tuning()
