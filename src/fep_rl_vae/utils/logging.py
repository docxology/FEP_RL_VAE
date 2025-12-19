"""Logging and data tracking utilities for FEP-RL-VAE."""


def add_to_epoch_history(epoch_history, epoch_dict):
    """Add epoch data to the complete training history."""
    for key, value in epoch_dict.items():

        if isinstance(value, float):
            if key not in epoch_history:
                epoch_history[key] = []
            epoch_history[key].append(value)

        elif isinstance(value, list):
            if key not in epoch_history:
                epoch_history[key] = []
            # Store the entire list as one entry per epoch
            epoch_history[key].append(value)

        elif isinstance(value, dict):
            if key not in epoch_history:
                epoch_history[key] = {}
            for k, v in value.items():
                if k not in epoch_history[key]:
                    epoch_history[key][k] = []
                epoch_history[key][k].append(v)


def print_epoch_summary(epoch_history):
    """Print summary of training history."""
    for key, value in epoch_history.items():
        print(f"{key}: {type(value)}")

        if isinstance(value, list):
            if isinstance(value[0], list):
                for v in value:
                    print(f"\t{len(v)}")
            else:
                print(f"\t{len(value)}")

        elif isinstance(value, dict):
            for k, v in value.items():
                print(f"\t{k}")
                print(f"\t\t{len(epoch_history[key][k])}")


def print_epoch_dict(epoch_dict):
    """Print current epoch data."""
    print("\nEpoch data:")
    for key, value in epoch_dict.items():
        print(f"{key}: {type(value)}")
        if isinstance(value, float):
            print(f"\t{value}")
        elif isinstance(value, list):
            print(f"\t{value}")
        elif isinstance(value, dict):
            for k, v in value.items():
                print(f"\t{k}:")
                print(f"\t\t{v}")
