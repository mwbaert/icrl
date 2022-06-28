import wandb
import os
import shutil
import argparse

files_to_download = [
    "best_model.zip",
    "config.json",
    "config.yaml",
    "monitor.csv",
    "output.log",
    "requirements.txt",
    "train_env_stats.pkl",
    "wandb-metadata.json",
    "wandb-summary.json"
]


def main(args):
    api = wandb.Api()

    if os.path.exists('files'):
        print('the directory "/files" already exists')
        return
    
    os.mkdir('files')
    run = api.run(f"/mwbaert/CPG-2/runs/{args['hash']}")
    
    for file in files_to_download:
        run.file(file).download()
        shutil.move(file, 'files/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hash", type=str)
    args = vars(parser.parse_args())

    main(args)
