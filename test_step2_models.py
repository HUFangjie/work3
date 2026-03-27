import torch

from models.model_zoo import get_model, adapt_model_config_for_dataset
from config.base_config import get_base_config

def main():
    base_cfg = get_base_config()
    data_cfg = base_cfg["data_config"]
    model_cfg = base_cfg["model_config"]

    # 根据 dataset 自动适配 model_config
    model_cfg = adapt_model_config_for_dataset(data_cfg["dataset"], model_cfg)

    print("Final model_config:", model_cfg)

    model = get_model(**model_cfg)
    print(model)

    # 随机打一发前向，确认尺寸对得上
    if data_cfg["dataset"].lower() in ["fmnist", "fashion_mnist"]:
        x = torch.randn(4, 1, 28, 28)
    elif data_cfg["dataset"].lower() == "cifar10":
        x = torch.randn(4, 3, 32, 32)
    else:
        raise ValueError("Unknown dataset for test_step2_models")

    logits = model(x)
    print("logits shape:", logits.shape)

if __name__ == "__main__":
    main()
