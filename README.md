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
│    ├── TRAINING.ipynb
│    ├── SWEEP.ipynb
│    └── train.py
└── .env  #contain wandb api key
```

# Giới thiệu chung về pipeline
- Data preprocessing: resize và áp dụng 1 số augmentation (có thể thay đổi trong file config `config/exp.yaml`)
- Xây dựng mô hình: Có thể chọn mô hình mình muốn ở trong file config `config/exp.yaml`, thêm mô hình chỉ cần add {model}.py vào trong `src/models`
- Training phase: khởi tạo optimizer, scheduler, loss function (có thể thay đổi trong file config `config/exp.yaml`, tương tự với các thông số train) 
- Evaluation: Mô hình load lại checkpoints đã lưu, đánh giá trên tập test.
- Storing: Log các thông tin cần thiết lên wandb run và tải mô hình tốt nhất lên wandb Artifacts thuộc run đó
- Hyperparameter tuning: Sử dụng W&B Sweep để tối ưu tham số.
- Điểm sáng tạo: Modular design - các hàm cần thiết được tách biệt để dễ training, optimizing cũng như bảo trì, mở rộng + sử dụng wandb

# Giới thiệu về wandb
- Ứng dụng trong logging và tracking experiments (W&B Models). Ngoài ra còn có Artifacts để lưu trữ dataset (data version) dễ dàng và lưu model tốt nhất dựa trên checkpoints.

# Setup for the project 
## Lab requirement: 
- [x] Training pipeline (data preprocessing, training, validation, evaluation)
- Applying [wandb](https://wandb.ai/) for:
- [x] Logging hyperparameters
- [x] Training dataset info (dataset link, source, dataset version...)
- [x] Logging metrics
- [x] Checkpoints directory (uploaded on wandb Artifacts)

## How to use
0. Install dependencies
```bash
pip install -r requirements.txt
```
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
