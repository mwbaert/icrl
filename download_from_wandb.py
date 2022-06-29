import wandb
import os
import shutil
import argparse

files_to_download = {
    'CPG-2': [
        "best_model.zip",
        "config.json",
        "config.yaml",
        "monitor.csv",
        "output.log",
        "requirements.txt",
        "train_env_stats.pkl",
        "wandb-metadata.json",
        "wandb-summary.json"],
    'icrl': [
        "best_cn_model.pt",
        "best_nominal_model.zip",
        "config.json",
        "config.yaml",
        "monitor.csv",
        "output.log",
        "requirements.txt",
        "train_env_stats.pkl",
        "wandb-metadata.json",
        "wandb-summary.json"],
}


def main(args):
    api = wandb.Api()

    if os.path.exists('files'):
        print('the directory "/files" already exists')
        return

    os.mkdir('files')
    run = api.run(f"/mwbaert/{args['project']}/runs/{args['hash']}")

    for file in files_to_download[args['project']]:
        run.file(file).download()
        shutil.move(file, 'files/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--hash", type=str)
    args = vars(parser.parse_args())

    main(args)

# python download_from_wandb --project CPG-2 --hash xxxxxxx
