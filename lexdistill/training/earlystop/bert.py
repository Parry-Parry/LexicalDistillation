from fire import Fire
import os
import ir_datasets as irds
from lexdistill import BERTLCETeacherLoader, MarginMultiLoss, MonoBERTModel
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import wandb

_logger = irds.log.easy()

def main(
        triples_file : str, 
        teacher_file : str,
        dataset_name : str, 
        out_dir : str, 
        max_epochs : int = 1, 
        batch_size : int = 16, 
        num_negatives : int = 1,
        lr : float = 0.00001, 
        grad_accum : int = 1,
        warmup_steps=0,
        shuffle=False,
        wandb_project=None,
        mode='std',
        rank : int = None):

    os.makedirs(out_dir, exist_ok=True)

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
            })

    corpus = irds.load(dataset_name)

    logging.info('loading model...')
    model = MonoBERTModel.init(rank=rank)

    loss_fn = MarginMultiLoss(batch_size, num_negatives)

    logging.info(f'loading loader with mode {mode}...')
    loader = BERTLCETeacherLoader(teacher_file, triples_file, corpus, model.tokenizer, mode=mode, batch_size=batch_size, num_negatives=num_negatives, shuffle=shuffle)

    opt = AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps//(batch_size*grad_accum), num_training_steps=total_steps//(batch_size*grad_accum))

    logging.info('init loader...')
    loader.setup()
    model.train()

    def _train_epoch(i):
        with _logger.pbar_raw(desc=f'training epoch {i}...', total=total_steps//batch_size) as pbar:
            total_loss = 0.
            for i in range(total_steps // batch_size):
                x, y = loader.get_batch(i)
                x = x.to(model.device)
                y = y.to(model.device)
                pred = model.forward(x)

                loss = loss_fn(pred, y) / grad_accum
                loss.backward()

                if (int(i + 1) % grad_accum == 0) or (int(i) == int(total_steps // batch_size - 1)): # Why is i a float?
                    opt.step()
                    opt.zero_grad()
                    sched.step()

                if wandb_project is not None:
                    wandb.log({'loss': loss.item()})
                    wandb.log({'lr': sched.get_last_lr()[0]})

                total_loss += loss.item()

                pbar.update(1)
                pbar.set_postfix({'loss': total_loss/(i+1)})

    epochs = 0
    while epochs < max_epochs:
        _train_epoch(epochs+1)
        epochs += 1
    

    model.save_pretrained(os.path.join(out_dir, 'model'))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)