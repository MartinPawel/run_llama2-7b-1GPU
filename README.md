### Train Llama2-7B on one GPU

This code example uses huggingface to finetune Llama2-7B using 1 GPU with 80GB of RAM. The following arguments to the trainer are important:

- ```fp16=False```; Changing this to true will make two copies of the model; one original model and in fp16. You don't want that as it takes up memory.
- ```optim="adafactor```: changing this to sth like adamw will use double the amounts of bytes per param.
- ```per_device_train_batch_size=1``` : Important as greater batch size requires more RAM.
- ```gradient_accumulation_steps=2``` or 4: This allows you to train with an effective batch size that is greater than 1; e.g., 2 or 4 when batch_size = 1 and acc steps are 2 or 4
- ```gradient_checkpointing=True```

In particular, you could change the train args like that:

```
 training_args = TrainingArguments(output_dir="finetuned_models",
                                      seed=0,
                                      fp16=False,
                                      gradient_accumulation_steps=4,
                                      gradient_checkpointing=True
                                      per_device_train_batch_size=args.batch_size,
                                      learning_rate=args.learning_rate,
                                      num_train_epochs=args.n_epochs,
                                      optim="adafactor"
                                    )
```


**1. Create conda environment and install requirements**

```
conda create -n p310 python=3.10 
conda activate p310
# Install the correct torch version depending on CUDA version from https://pytorch.org/
pip install -r requirements.txt
```

Next, finetune the llama model on the SST-2 dataset.

**2. Run models**

For example, to train the models on the SST-2 dataset using the 7B Llama2 model do the following:
```
sbatch run_sst2_ubs1_llama7b.sbatch
```
