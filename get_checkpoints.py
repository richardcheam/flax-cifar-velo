import sys
sys.path.append("../")
from glob import glob
import pickle
from flax import serialization

def load_pickle(filename="times.pkl"):
    """
    Loads a Python object from a pickle file.

    Args:
        filename (str): The file path to the pickle file (default: "times.pkl").

    Returns:
        Any: The deserialized Python object.
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        pickle.UnpicklingError: If the file is not a valid pickle format.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
        
def save_checkpoint(params, batch_stats, path):
    to_save = {"params": params, "batch_stats": batch_stats}
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(to_save))
    print(f"Checkpoint is written to {path}")

optimizer_set = ['VeLO', 'SGD', 'SGDM', 'Adam', 'AdamW']
    
for OPT in optimizer_set:
    files = glob(f"metrics/wrn28_8_cifar10_{OPT.lower()}_seed*.pkl")
    print(files)
    for f in sorted(files):
        res = load_pickle(filename=f)
        SEED_NUM = int(f.split("_seed")[-1].split(".pkl")[0])
        filename = f"{OPT.lower()}_wrn28_8_cifar10_seed{SEED_NUM}_pretrained.msgpack"
        save_checkpoint(res["params"], res["batch_stats"], path=f'checkpoints/cifar10/{filename}') 
    print("\n")

    


