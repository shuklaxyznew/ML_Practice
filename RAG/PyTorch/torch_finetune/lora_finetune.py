import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from torch_finetune.dataset import TRAIN_DATA, EVAL_DATA, INSTRUCTION_DATA, SentimentDataset, InstructionDataset
from torch_finetune.utils import compute_metrics, get_device, print_trainable, run_inference

MODEL_NAME  = "distilbert-base-uncased"
INFER_TEXTS = ["Absolutely wonderful!", "Terrible product.", "Not bad overall."]


def lora_classification():
    print("\n=== LoRA Classification ===")
    device    = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base      = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"], bias="none",
    )
    model = get_peft_model(base, config)
    print_trainable(model)

    args = TrainingArguments(
        output_dir="./output/lora_cls", num_train_epochs=5,
        per_device_train_batch_size=4, learning_rate=3e-4,
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

    model.save_pretrained("./output/lora_cls/adapter")
    tokenizer.save_pretrained("./output/lora_cls/adapter")
    run_inference(model, tokenizer, device, INFER_TEXTS)


def lora_merge():
    print("\n=== LoRA Merge Adapter into Base Model ===")
    tokenizer  = AutoTokenizer.from_pretrained("./output/lora_cls/adapter")
    base       = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    peft_model = PeftModel.from_pretrained(base, "./output/lora_cls/adapter")
    merged     = peft_model.merge_and_unload()
    merged.save_pretrained("./output/lora_merged")
    print("  Adapter merged → ./output/lora_merged (standard model, no PEFT overhead)")


def lora_causal_lm():
    print("\n=== LoRA Causal LM (GPT-style) ===")
    device    = get_device()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    base      = AutoModelForCausalLM.from_pretrained("gpt2")

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"], bias="none",
    )
    model = get_peft_model(base, config)
    print_trainable(model)

    args = TrainingArguments(
        output_dir="./output/lora_clm", num_train_epochs=3,
        per_device_train_batch_size=2, learning_rate=2e-4,
        logging_steps=5, report_to="none",
    )
    Trainer(
        model=model, args=args,
        train_dataset=InstructionDataset(INSTRUCTION_DATA, tokenizer),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    ).train()

    model.save_pretrained("./output/lora_clm/adapter")

    model.eval().to(device)
    prompt = "### Instruction:\nWhat is RAG?\n\n### Response:\n"
    inp    = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=60, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    print(f"\n  Generated: {tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]}")


if __name__ == "__main__":
    lora_classification()
    lora_merge()
    lora_causal_lm()
