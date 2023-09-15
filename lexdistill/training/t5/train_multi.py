from fire import Fire
import os
import ir_datasets as irds
from lexdistill import T5TeacherLoader, MarginMSELoss, MonoT5Model
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import wandb
import numpy as np

_logger = irds.log.easy()

def main(
        triples_file : str, 
        teacher_file : str,
        dataset_name : str, 
        out_dir : str, 
        mode : str = 'std',
        total_steps : int = 100000, 
        batch_size : int = 16, 
        lr : float = 0.001, 
        grad_accum : int = 1,
        warmup_steps=0,
        shuffle=False,
        wandb_project=None,
        aggr='mean',
        rank=None):

    os.makedirs(out_dir, exist_ok=True)

    if wandb_project is not None:
        wandb.init(project=wandb_project, config={
                'variant': triples_file.split('/')[-1],
                'teacher' : 'ensemble',
                'dataset': dataset_name,
                'total_steps': total_steps,
                'batch_size': batch_size * grad_accum,
                'lr': lr,
                'warmup_steps': warmup_steps,
                'mode': mode,
                'aggr' : aggr,
            })
        
    if aggr == 'mean':
        aggr = lambda x : [np.mean(x)]
    else:
        aggr = lambda x : x

    corpus = irds.load(dataset_name)

    logging.info('loading model...')
    model = MonoT5Model.init(rank=rank)

    logging.info(f'loading loader with mode {mode}...')
    loader = T5TeacherLoader(teacher_file, triples_file, corpus, model.tokenizer, mode=mode, batch_size=batch_size, shuffle=shuffle, aggr_func=aggr)

    opt = AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps//(batch_size*grad_accum), num_training_steps=total_steps//(batch_size*grad_accum))

    logging.info('init loader...')
    loader.setup()
    model.train()

    with _logger.pbar_raw(desc='training...', total=total_steps // batch_size) as pbar:
        total_loss = 0.
        for i in range(total_steps // batch_size):
            x, y = loader.get_batch(i)
            x = x.to(model.device)
            y = y.to(model.device)
            logging.info(y.shape)
            logging.info(y)
            pred = model.forward(x)
            logging.info(pred.shape)
            logging.info(pred)

            loss = MarginMSELoss(pred, y) / grad_accum
            loss.backward()

            if (int(i + 1) % grad_accum == 0) or (int(i) == int(total_steps // batch_size - 1)):
                opt.step()
                opt.zero_grad()
                sched.step()

            if wandb_project is not None:
                wandb.log({'loss': loss.item()})
                wandb.log({'lr': sched.get_last_lr()[0]})

            total_loss += loss.item()

            pbar.update(1)
            pbar.set_postfix({'loss': total_loss/(i+1)})

    model.save_pretrained(os.path.join(out_dir, 'model'))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)