import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig,
    DataCollatorWithPadding, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from torch_finetune.dataset import TRAIN_DATA, EVAL_DATA, INSTRUCTION_DATA, SentimentDataset, InstructionDataset
from torch_finetune.utils import compute_metrics, get_device, print_trainable, run_inference

MODEL_NAME  = "distilbert-base-uncased"
INFER_TEXTS = ["Absolutely wonderful!", "Terrible product."]


def bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def qlora_classification():
    print("\n=== QLoRA Classification ===")
    device    = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if device.type == "cuda":
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2, quantization_config=bnb_config(), device_map="auto")
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"], bias="none",
    )
    model = get_peft_model(model, config)
    print_trainable(model)

    args = TrainingArguments(
        output_dir="./output/qlora_cls", num_train_epochs=5,
        per_device_train_batch_size=4, learning_rate=3e-4,
        evaluation_strategy="epoch",
        fp16=device.type == "cuda",
        gradient_checkpointing=device.type == "cuda",
        report_to="none",
    )
    Trainer(
        model=model, args=args,
        train_dataset=SentimentDataset(TRAIN_DATA, tokenizer),
        eval_dataset=SentimentDataset(EVAL_DATA, tokenizer),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    ).train()

    model.save_pretrained("./output/qlora_cls/adapter")
    run_inference(model, tokenizer, device, INFER_TEXTS)


def qlora_causal_lm():
    print("\n=== QLoRA Causal LM ===")
    device    = get_device()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", quantization_config=bnb_config(), device_map="auto")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained("gpt2")

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"], bias="none",
    )
    model = get_peft_model(model, config)
    print_trainable(model)

    args = TrainingArguments(
        output_dir="./output/qlora_clm", num_train_epochs=3,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=2e-4,
        optim="paged_adamw_32bit" if device.type == "cuda" else "adamw_torch",
        fp16=device.type == "cuda",
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=5, report_to="none",
    )
    Trainer(
        model=model, args=args,
        train_dataset=InstructionDataset(INSTRUCTION_DATA, tokenizer),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    ).train()

    model.save_pretrained("./output/qlora_clm/adapter")

    model.eval()
    if device.type != "cuda":
        model.to(device)
    prompt = "### Instruction:\nWhat is QLoRA?\n\n### Response:\n"
    inp    = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=60, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    print(f"\n  Generated: {tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]}")


if __name__ == "__main__":
    qlora_classification()
    qlora_causal_lm()
