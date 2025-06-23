def train_model(train_file_path, val_file_path):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
    from datasets import load_dataset
    import torch

    model_name = "Salesforce/codet5p-220m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    data_files = {
        "train": train_file_path,
        "validation": val_file_path
    }
    dataset = load_dataset("json", data_files=data_files)

    max_length = 512

    def tokenize_fn(batch):
        return tokenizer(
            batch["input"],
            text_target=batch["output"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/checkpoints",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_dir="./models/logs",
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print("✅ CodeT5+ model training with validation complete.")
