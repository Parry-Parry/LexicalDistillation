from fire import Fire
import os
import ir_datasets as irds
from lexdistill import BERTPerfectLoader, MarginMSELoss, MonoBERTModel
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import wandb

_logger = irds.log.easy()

def main(
        triples_file : str, 
        dataset_name : str, 
        out_dir : str, 
        total_steps : int = 100000, 
        batch_size : int = 16, 
        lr : float = 0.001, 
        grad_accum : int = 1,
        warmup_steps=0,
        shuffle=False,
        wandb_project=None,):

    os.makedirs(out_dir, exist_ok=True)

    if wandb_project is not None:
        wandb.init(project=wandb_project, config={
                'variant': triples_file.split('/')[-1],
                'teacher' : 'perfect',
                'dataset': dataset_name,
                'total_steps': total_steps,
                'warmup_steps': warmup_steps,
                'batch_size': batch_size * grad_accum,
                'lr': lr,
                'mode': 'solo_perfect',
            })

    corpus = irds.load(dataset_name)

    logging.info(f'Total steps: {total_steps}, batch size: {batch_size}, grad accum: {grad_accum}, warmup steps: {warmup_steps}')

    logging.info('loading model...')
    model = MonoBERTModel.init()

    logging.info(f'loading loader...')
    loader = BERTPerfectLoader(triples_file, corpus, model.tokenizer, batch_size=batch_size, shuffle=shuffle)

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
            pred = model.forward(x)
            loss = MarginMSELoss(pred, y) / grad_accum
            loss.backward()

            if i + 1 % grad_accum == 0 or i == total_steps // batch_size - 1:
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