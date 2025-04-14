hello world, this is **wandb** tutorial
# Members
| Name                | MSSV      | Roles  |
|---------------------|-----------|--------|
| Nguyễn Hữu Nam      | 22520917  | Leader |
| Nguyễn Khánh        | asdas  | Member |
| Nguyễn Minh Sơn        | asdsd  | Member |

# Structure
```python
├── data
├── config
├── src
│    ├── loader
│    ├── parsers
│    ├── utils
│    └── train.py
└── .env  #contain wandb api key
```

# Setup for the project 
## Lab requirement: 
- [x] Training pipeline (data preprocessing, training, validation, evaluation)
- Applying [wandb](https://wandb.ai/) for:
- [x] Logging hyperparameters
- [x] Training dataset info (dataset link, source, dataset version...)
- [x] Logging metrics
- [x] Checkpoints directory (uploaded on wandb Artifacts)

## How to use
1. Login to wandb
```python
import wandb
api_key  = "your_api_key_here"
wandb.login(key=api_key)
wandb.require("core")
```
2. Training phase (including logging and checkpoints, you can also look at `src/TRAINING.ipynb`)
```bash
python train.py --config ../config/exp.yaml --wandb 1
```
then you need to go to `https://wandb.ai/{ENTITY}/{PROJECT}/{RUN_ID}` to see the result (eventually the wandb will show the link for you)

3. Hyperparameter tuning. 

You can see in `src/SWEEP.ipynb` to get the `sweep_id`, then:
```bash
wandb agent --count 20 {sweep_id}
```
