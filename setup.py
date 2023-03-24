# Create directory structure
import os
DIRECTORIES = [
    "figures",
    "figures/data_PDFs",
    "figures/data_EXPLORE",
    "DATA",
    "DATA/SHIPS",
    "DATA/model_output",
]

def setup_directory(target):
    if not os.path.isdir(target):
        os.mkdir(target)
        print(f"directory {target} has been created.")
    else:
        print(f"directory {target} already exists.")


def setup_directories():
    for target in DIRECTORIES:
        setup_directory(target)


if __name__ == "__main__":
    setup_directories()

