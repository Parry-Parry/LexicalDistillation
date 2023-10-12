from fire import Fire
import os
import ir_datasets as irds
import pandas as pd
from lexdistill import BERTLCETeacherLoader, MarginMultiLoss, MonoBERTModel, EarlyStopping
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import wandb
from pyterrier_dr import ElectraScorer

_logger = irds.log.easy()

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
        shuffle : bool = False,
        wandb_project=None,
        mode : str = 'std',
        early_patience : str = 30,
        early_check : str = 4000,
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
                'rank': rank,
                'early_patience': early_patience,
                'early_check': early_check,
                'min_train_steps': min_train_steps,
                'val_batch_size': val_batch_size,
            })

    corpus = irds.load(dataset_name)

    logging.info('loading model...')
    model = MonoBERTModel.init(rank=rank)

    loss_fn = MarginMultiLoss(batch_size, num_negatives)

    logging.info(f'loading loader with mode {mode}...')
    loader = BERTLCETeacherLoader(teacher_file, triples_file, corpus, model.tokenizer, mode=mode, batch_size=batch_size, num_negatives=num_negatives, shuffle=shuffle)
    
    if val_file is not None:
        val_set = pd.read_csv(val_file, sep='\t', names=['qid', 'docno', 'score'], index_col=False)
        stopping = EarlyStopping(val_set, 'nDCG@10', corpus.qrels_iter(), mode='max', patience=early_patience)
        val_model = ElectraScorer(batch_size=val_batch_size, device=model.device)
        val_model.model = model.model
    
    logging.info('init loader...')
    loader.setup()
    total_steps = len(loader.triples) * max_epochs
    
    opt = AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps//(batch_size*grad_accum), num_training_steps=total_steps//(batch_size*grad_accum))
    
    model.train()

    logging.info('training for a maximum of {} epochs = {} steps'.format(max_epochs, total_steps))
    
    def _train_epoch(i):
        total_loss = 0.
        with _logger.pbar_raw(desc=f'training epoch {i}...', total=len(loader.triples)//batch_size) as pbar:
            for j in range(len(loader.triples)//batch_size):
                x, y = loader.get_batch(j)
                x = x.to(model.device)
                y = y.to(model.device)
                pred = model.forward(x)

                loss = loss_fn(pred, y) / grad_accum
                loss.backward()

                if (int(j + 1) % grad_accum == 0) or (int(j) == int(total_steps // batch_size - 1)): # Why is i a float?
                    opt.step()
                    opt.zero_grad()
                    sched.step()
                
                if (int(j) % early_check == 0) and (int(j) > min_train_steps and val_file is not None):
                    if stopping(model.transfer_state_dict(val_model)):
                        logging.info(f'early stopping at epoch {i}')
                        return True

                if wandb_project is not None:
                    wandb.log({'epoch': i, 'loss': loss.item(), 'lr': sched.get_last_lr()[0]})
                total_loss += loss.item()

                pbar.update(1)
                pbar.set_postfix({'Epoch Loss': total_loss/(i+1)})
                return False

    epochs = 0
    value = False
    while epochs < max_epochs and not value:
        value = _train_epoch(epochs+1)
        epochs += 1

    model.save_pretrained(os.path.join(out_dir, 'model'))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)