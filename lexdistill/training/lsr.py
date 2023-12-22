from fire import Fire
import os
import ir_datasets as irds
from lexdistill.lsr.transformer import LSR
import torch
from lexdistill.lsr.losses.margin_mse_loss import MarginMSELoss
from lexdistill.lsr.models import TransformerMLMSparseEncoder, DualSparseEncoder
from lexdistill.lsr.tokenizer import HFTokenizer
from lexdistill.lsr.trainer import HFTrainer
from lexdistill.lsr.callback import SparseEarlyStoppingCallback
from lexdistill.lsr.datasets.triplet_distil_dataset import TripletIDDistilDataset
from lexdistill.lsr.datasets.data_collator import DistillDataCollator
from lexdistill.lsr.losses.regularizer import FLOPs
from transformers import AdamW, get_constant_schedule_with_warmup, TrainingArguments
import logging
from lexdistill.lsr.models.mlm import TransformerMLMConfig
import wandb

def main(
        triples_file : str, 
        teacher_file : str,
        dataset_name : str, 
        out_dir : str, 
        val_file : str = None,
        max_epochs : int = 1, 
        batch_size : int = 16, 
        val_batch_size : int = 128,
        num_negatives : int = 1,
        lr : float = 0.00001, 
        grad_accum : int = 1,
        warmup_steps : int = 0,
        min_train_steps : int = 50000,
        wandb_project=None,
        mode : str = 'std',
        early_patience : str = 30,
        early_check : str = 4000,
        rank : int = None):

    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, 'model')):
        logging.info(f'model already exists at {out_dir}, skipping training')
        return

    if wandb_project is not None:
        wandb.init(project=wandb_project, config={
                'variant': triples_file.split('/')[-1],
                'teacher': os.path.basename(teacher_file), 
                'dataset': dataset_name,
                'max_epochs': max_epochs,
                'warmup_steps': warmup_steps,
                'batch_size': batch_size * grad_accum,
                'num_negatives': num_negatives,
                'lr': lr,
                'mode': mode,
                'rank': rank,
                'early_patience': early_patience,
                'early_check': early_check,
                'min_train_steps': min_train_steps,
                'val_batch_size': val_batch_size,
            })

    logging.info('loading model...')
    q_reg = FLOPs(weight=0.1, T=50000)
    d_reg = FLOPs(weight=0.08, T=50000)
    loss_fn = MarginMSELoss(q_regularizer=q_reg, d_regularizer=d_reg, num_negatives=num_negatives)

    config = TransformerMLMConfig(tf_base_model_name_or_dir='google/electra-base-discriminator')
    model = DualSparseEncoder(query_encoder=TransformerMLMSparseEncoder(config), doc_encoder=TransformerMLMSparseEncoder(config))
    tokenizer = HFTokenizer.from_pretrained('google/electra-base-discriminator')

    callbacks = []
    if val_file:
        val_model = LSR(DualSparseEncoder(query_encoder=TransformerMLMSparseEncoder(config), doc_encoder=TransformerMLMSparseEncoder(config)), tokenizer)
        earlystop = SparseEarlyStoppingCallback(val_model, 
                                                val_file, 
                                                ir_dataset=dataset_name, 
                                                index='msmarco_passage', 
                                                metric='nDCG@10', 
                                                early_check=early_check, 
                                                min_train_steps=min_train_steps, 
                                                patience=early_patience)
        callbacks.append(earlystop)

    dataset = TripletIDDistilDataset(teacher_file, triples_file, irds.load(dataset_name), num_negatives=num_negatives, shuffle=False)
    dataloader = DistillDataCollator(tokenizer=tokenizer)

    opt = AdamW(model.parameters(), lr=lr)

    training_args = TrainingArguments(
        output_dir=out_dir,
        do_train=True,
        do_predict=False,
        do_eval=False,
        overwrite_output_dir=True,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        evaluation_strategy='no',
        warmup_steps=warmup_steps,
        logging_steps=1000,
        learning_rate=lr,
        save_steps=10000,
        save_total_limit=1,
        dataloader_num_workers=4,
        disable_tqdm=False,
        seed=42,
        report_to='wandb'
    )

    trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=dataloader,
        callbacks=callbacks,
        optimizers=(opt, get_constant_schedule_with_warmup(opt, warmup_steps)),
        loss_fn=loss_fn,
        device=rank
    )

    trainer.train()
    trainer.save_model(os.path.join(out_dir, 'model'))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)