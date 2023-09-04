from fire import Fire
import os
import ir_datasets as irds
from lexdistill import SingleTeacherLoader, MarginMSELoss, MonoT5Model
from transformers import AdamW, get_constant_schedule_with_warmup
import logging
import wandb

_logger = irds.log.easy()

def main(
        triples_file : str, 
        teacher_file : str,
        dataset_name : str, 
        out_dir : str, 
        total_steps : int = 100000, 
        batch_size : int = 16, 
        lr : float = 0.001, 
        warmup_steps=0,
        shuffle=False,
        wandb_project=None,
        mode='std',):

    os.makedirs(out_dir, exist_ok=True)

    if wandb_project is not None:
        wandb.init(project=wandb_project, config={
                'variant': triples_file.split('/')[-1],
                'teacher': os.path.basename(teacher_file), 
                'dataset': dataset_name,
                'total_steps': total_steps,
                'warmup_steps': warmup_steps,
                'batch_size': batch_size,
                'lr': lr,
                'mode': mode,
            })

    corpus = irds.load(dataset_name)

    logging.info('loading model...')
    model = MonoT5Model.init()

    logging.info(f'loading loader with mode {mode}...')
    loader = SingleTeacherLoader(teacher_file, triples_file, corpus, model.tokenizer, mode=mode, batch_size=batch_size, shuffle=shuffle)

    opt = AdamW(model.parameters(), lr=lr)
    sched = get_constant_schedule_with_warmup(opt, num_warmup_steps=warmup_steps//batch_size, num_training_steps=total_steps//batch_size)

    logging.info('init loader...')
    loader.setup()
    model.train()

    with _logger.pbar_raw(desc='training...', total=total_steps // batch_size) as pbar:
        total_loss = 0.
        for i in range(total_steps // batch_size):
            x, y = loader.get_batch(i)
            x.to(model.device)
            y.to(model.device)
            pred = model.forward(x)

            opt.zero_grad()
            loss = MarginMSELoss(pred, y)
            loss.backward()
            opt.step()
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