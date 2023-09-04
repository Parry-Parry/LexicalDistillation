import subprocess as sp
from fire import Fire
import logging
from os.path import join

ARGS = {
    'bm25.one_sided.teacher' : {
        'script' : 'lexdistill.training.train_single',
        'model' : 'one_sided'
    },
    'bm25.perfect_one_sided.teacher' : {
        'script' : 'lexdistill.training.train_single',
        'mode' : 'perfect_one_sided'
    },
    'bm25.perfect.teacher' : {
        'script' : 'lexdistill.training.train_single',
        'mode' : 'perfect'
    },
    'bm25.teacher' : {
        'script' : 'lexdistill.training.train_single',
        'mode' : 'std'
    },
    'perfect' : {
        'script' : 'lexdistill.training.train_perfect',
        'mode' : None
    }
}

BATCH_SIZE = 16
WARMUP_STEPS = 2500

def main(model_dir : str, 
         triples_file : str, 
         teacher_file : str, 
         total_steps : int = 30000, 
         wandb_project : str = None):
    
    for name, args in ARGS.items():
        if name in model_dir:
            logging.info(f'Skipping {name}...')
            continue
        logging.info(f'Training {name}...')
        cmd = ['python', 
               '-m', 
               args['script'], 
               '--triples_file' 
               triples_file, 
               '--teacher_file',
               teacher_file, 
               '--dataset_name',
               'msmarco-passage/train/triples-small', 
               '--out_dir',
               join(model_dir, name),
                '--total_steps', 
               str(total_steps), 
                '--batch_size',
               str(BATCH_SIZE), 
                '--warm8up_steps',
               str(WARMUP_STEPS)
                ]
        if args['mode'] is not None:
            cmd.extend(['--mode', args['mode']])
        if wandb_project is not None:
            cmd.extend(['--wandb_project', wandb_project])
        sp.run(cmd)

    return "Done!"
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    Fire(main)
        


