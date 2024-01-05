from fire import Fire
import os
import ir_datasets as irds
import pandas as pd
from lexdistill import BERTdotTeacherLoader, MarginMultiLoss, EarlyStopping, BERTDotModel, InBatchLoss
from transformers import AdamW, get_constant_schedule_with_warmup, ElectraModel
import logging
import wandb
from pyterrier_dr import HgfBiEncoder, BiScorer

_logger = irds.log.easy()

def main(
        triples_file : str, 
        teacher_file : str,
        dataset_name : str, 
        out_dir : str, 
        return_vecs : bool = False,
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

    corpus = irds.load(dataset_name)

    logging.info('loading model...')
    model = BERTDotModel.init(rank=rank, return_vecs=return_vecs)
    logging.info(f'using device {model.device}')

    loss_fn = MarginMultiLoss(batch_size, num_negatives)
    in_batch_loss = InBatchLoss(batch_size, num_negatives)

    logging.info(f'loading loader with mode {mode}...')
    loader = BERTdotTeacherLoader(teacher_file, triples_file, corpus, model.tokenizer, mode=mode, batch_size=batch_size, num_negatives=num_negatives, shuffle=shuffle)
    
    logging.info('init loader...')
    loader.setup()
    total_steps = len(loader.triples) * max_epochs

    if val_file is not None:
        val_set = pd.read_csv(val_file, sep='\t', index_col=False)
        val_set['query'] = val_set['qid'].apply(lambda x: loader.queries[str(x)])
        val_set['text'] = val_set['docno'].apply(lambda x: loader.docs[str(x)])
        stopping = EarlyStopping(val_set, 'nDCG@10', corpus.qrels_iter(), mode='max', patience=early_patience)
        val_backbone = ElectraModel.from_pretrained('google/electra-base-discriminator')
        val_backbone = HgfBiEncoder(val_backbone, model.tokenizer, {}, device=model.device)
        val_model = BiScorer(val_backbone, batch_size=val_batch_size, verbose=False)
    opt = AdamW(model.parameters(), lr=lr)
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=warmup_steps//(batch_size*grad_accum))
    
    model.train()

    logging.info('training for a maximum of {} epochs = {} steps'.format(max_epochs, total_steps))
    def _train_epoch(i, global_step=0):
        total_loss = 0.
        with _logger.pbar_raw(desc=f'training epoch {i}...', total=len(loader.triples)//batch_size) as pbar:
            for j in range(len(loader.triples)//batch_size):
                x, y = loader.get_batch(j)
                queries, docs = x[0].to(model.device), x[1].to(model.device) 
                y = y.to(model.device)

                pred, query_vec, doc_vec = model.forward((queries, docs, num_negatives))

                loss = loss_fn(pred, y) 
                if return_vecs:
                    loss += in_batch_loss(query_vec, doc_vec)
                loss = loss / grad_accum
                loss.backward()

                if (int(j + 1) % grad_accum == 0) or (global_step == int(total_steps // batch_size - 1)): # Why is i a float?
                    opt.step()
                    opt.zero_grad()
                    sched.step()
                
                if (global_step % early_check == 0) and (global_step > min_train_steps and val_file is not None):
                    model.transfer_state_dict(val_model)
                    if stopping(val_model):
                        logging.info(f'early stopping at epoch {i}')
                        return True, global_step

                if wandb_project is not None:
                    wandb.log({'epoch': i, 'loss': loss.item(), 'lr': sched.get_last_lr()[0]})
                total_loss += loss.item()
                global_step += 1

                pbar.update(1)
                pbar.set_postfix({'Epoch Loss': total_loss/(j+1)})
        return False, global_step

    epochs = 0
    global_step = 0
    value = False
    while epochs < max_epochs and not value:
        value, global_step = _train_epoch(epochs+1, global_step)
        epochs += 1

    model.save_pretrained(os.path.join(out_dir, 'model'))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)