import os

repo_path = os.getcwd()
print(f"Setting MISO_ROOT to {repo_path} in all configuration files.")

# List of YAML files with MISO_ROOT to replace
yaml_files = ["envs/cartpole/config.yaml",
              "envs/nuplan/config.yaml",
              "envs/reacher/config.yaml",
              "training/configs/dataset.yaml",
              "training/configs/training.yaml"]


def update_miso_root(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify only the MISO_ROOT line
        with open(file_path, 'w') as file:
            for line in lines:
                if line.startswith("MISO_ROOT:"):
                    file.write(f"MISO_ROOT: {repo_path}\n")
                    print(f"Updated MISO_ROOT in {file_path}")
                else:
                    file.write(line)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found.")


for file_path in yaml_files:
    update_miso_root(file_path)

print("MISO_ROOT path updated in all specified YAML files.")
