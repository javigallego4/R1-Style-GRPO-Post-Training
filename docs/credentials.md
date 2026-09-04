# Credentials Setup

This project must not store private credentials in the repository.

## Weights & Biases

For local runs:

```bash
wandb login
```

For Kaggle runs, store the W&B API key as a Kaggle secret or inject it as an environment variable before training.
The standard secret name is `WANDB_API_KEY`. Additional compatible secret names can be configured through `tracking.secret_names` if needed.
After pushing a kernel from the Kaggle CLI, re-check the notebook's Add-ons > Secrets panel in the Kaggle web UI. CLI pushes can leave the new run without access to previously selected secrets, so the `WANDB_API_KEY` secret may need to be enabled again before rerunning the kernel.

Expected settings:

```yaml
tracking:
  enabled: true
  project_name: r1-grpo-kaggle
  mode: online
  secret_names:
    - WANDB_API_KEY
```

The default config keeps adapter artifact uploads disabled:

```yaml
tracking:
  enabled: true
  log_code: false
  log_adapter_artifacts: false
  log_model_artifacts: false

export:
  publish_adapter: false
```

## Kaggle API

To launch Kaggle kernels from local, install and authenticate the Kaggle CLI outside the repository:

```bash
pip install kaggle
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Then generate kernel metadata with your username:

```bash
python scripts/create_kaggle_metadata.py --username YOUR_KAGGLE_USERNAME --private --internet
```

Finally push the script kernel:

```bash
kaggle kernels push -p . --accelerator NvidiaTeslaT4
```

## GitHub

Authenticate GitHub CLI locally:

```bash
gh auth login -h github.com
```

After authentication, create the remote repository and push when ready.
