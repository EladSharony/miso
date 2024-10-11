import warnings

# Ignore the specific UserWarning related to Hydra's _self_ key
warnings.filterwarnings("ignore", message="In 'default_simulation': Defaults list is missing `_self_`")

