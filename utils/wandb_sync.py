import os
import time
import subprocess


def sync(path):
    print(f"syncing start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        subprocess.run(args=["wandb", "sync", "--include-offline"], cwd=dir_path)
    print(f"syncing finish: {time.strftime('%Y-%m-%d %H:%M:%S')}")


def main(path):
    while True:
        sync(path)
        time.sleep(60 * 5)



if __name__ == "__main__":
    path = "/home/vmeshchaninov/DiffusionTextGeneration/latent_diffusion/wandb"
    main(path)
