import os

from utils.config_parser import load_config
from utils.seed_utils import set_global_seed
from core.data_manager import DataManager


def main():
    # 1. 读取配置（此时用的是 base_config，也可以后面加 --dataset 参数）
    config = load_config()

    # 2. 设随机种子，保证划分可复现
    set_global_seed(config["seed"])

    # 3. 初始化 DataManager（会自动下载数据、划分 public/private、做 Non-IID 分配）
    dm = DataManager(config)

    # 4. 打印一下划分结果
    print(dm.summary())

    # 5. 随便拿几个 batch 看看形状是否正常
    public_loader = dm.get_public_loader()
    client0_loader = dm.get_client_private_loader(0)

    print("Iterating one batch from public_loader...")
    x_pub, y_pub = next(iter(public_loader))
    print(f"  public batch shape: x={x_pub.shape}, y={y_pub.shape}")

    print("Iterating one batch from client 0 private loader...")
    x_p0, y_p0 = next(iter(client0_loader))
    print(f"  client0 batch shape: x={x_p0.shape}, y={y_p0.shape}")

    # 6. 也可以检查 val/test loader 是否可用
    val_loader = dm.get_val_loader()
    test_loader = dm.get_test_loader()
    print(f"  val size:  {len(val_loader.dataset)}")
    print(f"  test size: {len(test_loader.dataset)}")


if __name__ == "__main__":
    # 确保当前工作目录是工程根目录（包含 config/, core/, data/, utils/）
    print("CWD:", os.getcwd())
    main()
