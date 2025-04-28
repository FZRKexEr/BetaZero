import os
import torch
from config import Config
from data_handler import DataHandler
from trainer import Trainer 
from utils import find_latest_model, set_random_seed, save_torchscript_model
from gomoku_neural_net import GameModel
import datetime

def main():
    # 直接使用DEFAULT_CONFIG，不需要命令行参数
    config = Config()
    
    # 设置随机种子
    set_random_seed(42)
    
    # 初始化模型
    model = GameModel(
        board_size=config["board_size"],
        input_channels=config["input_channels"],
        device=config["device"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"]
    )


    # 初始化数据处理器
    data_handler = DataHandler(config)
    
    # 初始化训练器
    trainer = Trainer(model, data_handler, config)
 
    
    # 加载最新模型（如果存在）
    latest_model = find_latest_model(config["model_dir"])
    if latest_model:
        print(f"加载最新模型: {latest_model}")
        model.load(latest_model)
    else:
        print("创建新模型")
        # 创建并保存初始模型
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存PyTorch模型
        pytorch_path = os.path.join(config["model_dir"], f"pytorch_{timestamp}.pt")
        model.save(pytorch_path)
        print(f"初始PyTorch模型已保存: {pytorch_path}")

        torchscript_path = os.path.join(config["model_dir"], f"torchscript_{timestamp}.pt")
        
        # 保存TorchScript模型
        save_torchscript_model(
            model=model,
            model_dir=config["model_dir"],
            timestamp=timestamp,
            input_channels=config["input_channels"],
            board_size=config["board_size"],
            device=config["device"]
        )

        print("开始之前先自我对弈两次，积累数据，减少训练不稳定性")
        trainer.run_selfplay(torchscript_path)
        trainer.run_selfplay(torchscript_path)
   
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()