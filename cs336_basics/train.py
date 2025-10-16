import torch
import torch.nn as nn
import os
import argparse
from Transformer import TransformerLM
from Learning_rate import lr_cosine_schedule,clip_grad_l2_
import wandb
from loguru import logger
import json,yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
from Loss import  AdamW,cross_entropy_loss
from DataLoader import get_batch,load_checkpoint,save_checkpoint,evaluate_model
def build_argparse() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train TransformerLM with CLI controllable hyperparams")

    # logging
    p.add_argument("--log_file", type=str, default="./data/log/train_v0.log")
    p.add_argument("--log_rotation", type=str, default="1 day")
    p.add_argument("--log_retention", type=str, default="7 days")
    p.add_argument("--log_level", type=str, default="INFO")

    # model
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=int, default=10000)

    # optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--max_norm", type=float, default=1.0)

    # training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--total_epochs", type=float, default=0.5, help="支持小数（按 token 比例换算步数）")
    p.add_argument("--checkpoint_freq", type=int, default=2000)
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--val_freq", type=int, default=400)
    p.add_argument("--val_batch_size", type=int, default=16)
    p.add_argument("--val_batches", type=int, default=20)

    # data
    p.add_argument("--training_dataset_path", type=str, default="./data/token/TinyStories_train_10000_token_ids.npy")
    p.add_argument("--validation_dataset_path", type=str, default="./data/token/TinyStories_valid_10000_token_ids.npy")

    # checkpoint io
    p.add_argument("--checkpoint_load_path", type=str, default=None)
    p.add_argument("--checkpoint_save_format", type=str, default="./data/model/checkpoint_v0_{}.pt")
    p.add_argument("--final_model_path", type=str, default="./data/model/final_model_v0.pt")

    # wandb
    p.add_argument("--wandb_project", type=str, default="cs336-assignment-1")
    p.add_argument("--wandb_run_name", type=str, default="train_v1")
    p.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda:0 ...")

    # optional config file
    p.add_argument("--config", type=str, default=None, help="JSON 或 YAML 配置文件路径（命令行可覆盖其中字段）")

    return p


def args_to_configs(args:argparse.Namespace):
    model_config = dict(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    optim_config = dict(
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        max_norm=args.max_norm
    )
    train_config = dict(
        batch_size=args.batch_size,
        total_epochs=args.total_epochs,
        checkpoint_freq=args.checkpoint_freq,
        log_freq=args.log_freq,
        val_freq=args.val_freq,
        val_batch_size=args.val_batch_size,
        val_batches=args.val_batches,
    )
    data_paths = dict(
        training_dataset_path=args.training_dataset_path,
        validation_dataset_path=args.validation_dataset_path,
        checkpoint_load_path=args.checkpoint_load_path,
        checkpoint_save_format=args.checkpoint_save_format,
        final_model_path=args.final_model_path,
    )
    return model_config, optim_config, train_config, data_paths

#可能会有 config 的yaml文件
def load_config_file(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        if path.endswith((".yml", ".yaml")) and yaml is not None:
            return yaml.safe_load(f)
        raise ValueError("仅支持 .json / .yml / .yaml（需要安装 pyyaml）")
def main():
    # 代码中定义的默认值（最低） < 配置文件（yaml）中的默认值 < 命令行参数（最高）
    parser=build_argparse()
    partial_args,_=parser.parse_known_args() #第一个值 一个包含已成功解析的参数的命名空间对象（Namespace） 第二个值包含未被解析的
    file_cfg=load_config_file(partial_args.config)
    if file_cfg :
        parser.set_defaults(**file_cfg)
    args=parser.parse_args()

    Path(os.path.dirname(args.log_file) or ".").mkdir(parents=True, exist_ok=True)
    logger.add(args.log_file,rotation=args.log_rotation,rentention=args.log_rentention,level=args.log_level)

    #随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #设置
    if args.device == "auto":
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else :
        device =torch.device(args.device)

    # 组装配置（为了 wandb 与日志打印）
    model_config, optim_config, train_config, data_paths = args_to_configs(args)

    # wandb 模式
    if args.wandb_mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
    elif args.wandb_mode == "offline":
        os.environ["WANDB_MODE"] = "offline"

    run=wandb.init(project=args.wandb_project,name=args.wandb_run_name,config={
        "model": model_config,
        "optimizer": optim_config,
        "training": train_config,
        "data": data_paths,
    }) # name 参数设置本次实验的运行名称，用于区分不同的实验运行

# 初始化模型
    logger.info("开始初始化模型...")
    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_model=model_config["d_model"],
        d_ff=model_config["d_ff"],
        rope_theta=model_config["rope_theta"],
        device=device,
    )
    logger.info("模型初始化完成。")

    optimizer = AdamW(model.parameters(),
                      lr=optim_config["lr"],
                      betas=optim_config["betas"],
                      weight_decay=optim_config["weight_decay"],
                      eps=optim_config["eps"],
                      )
    logger.info("优化器初始化完成。")

    # 如果有checkpoint，则加载checkpoint
    start_iter =1
    if data_paths["checkpoint_load_path"] :
        logger.info(f"开始加载模型检查点:{data_paths['checkpoint_load_path']}")
        start_iter=load_checkpoint(data_paths[""],model=model,optimizer=optimizer)
        start_iter+=1
        logger.info(f"模型检查点加载成功，当前迭代次数: {start_iter}")
    else:
        logger.info("没有提供模型检查点，开始从头训练。")

    #加载数据集
    logger.info(f"开始加载数据集，训练集:{data_paths['training_dataset_path']},验证集:{data_paths['validation_dataset_path']}")
    training_dataset=np.load(data_paths["training_dataset_path"],mmap_mode="r+")
    #该语句主要用于加载以 .npy 或 .npz 格式存储的数据
    #"r+"它不会将整个数据一次性加载到内存中，而是在需要访问数据的某个部分时，才将对应的数据从磁盘映射到内存。
    if data_paths["validation_dataset_path"] :
        validation_dataset=np.load(data_paths["validation_dataset_path"],mmap_mode="r+")
    logger.info("数据集加载完成")

    # 训练步数
    total_tokens = training_dataset.shape[0]
    total_steps = int(train_config["total_epochs"] * total_tokens) // (
                train_config["batch_size"] * model_config["context_length"])
    logger.info(
        f"总token数: {total_tokens}, 训练轮数: {train_config['total_epochs']},"
        f" batch大小: {train_config['batch_size']}, 上下文长度: {model_config['context_length']}")
    logger.info(f"总训练步数: {total_steps}")
    # step循环开始
    logger.info("开始训练模型...")
    for  step in tqdm(range(start_iter,total_steps+1),desc="训练进度", unit="step"):
        optimizer.zero_grad()
        lr_now=lr_cosine_schedule(t=step,a_max=optim_config["lr"],a_min=0.1*optim_config["lr"],
                                  T_w=int(0.05*total_steps),T_c=int(total_steps))
        for pg in optimizer.param_groups:
            pg["lr"]=lr_now
        #bacth
        inputs,outputs=get_batch(training_dataset,train_config["batch_size"],
                                 train_config["context_length"],device=device)
        #训练
        logits=model(inputs)
        #损失
        loss= cross_entropy_loss(logits,outputs)
        #反向传播和优化参数
        loss.backward()
        #处理异常的梯度
        clip_grad_l2_(model.parameters(),max_norm=optim_config["max_norm"])

        optimizer.step()
        # logging
        if step % train_config["log_freq"] == 0:
            logger.info(f"Step {step}, Loss: {loss.item():.6f}")
            wandb.log({"train_loss": loss.item(), "lr": lr_now,"step": step})

        # validation
        if validation_dataset is not None and step % train_config["val_freq"] == 0:
            logger.info("在验证集上评估模型...")
            val_loss = evaluate_model(
                model=model,
                dataset=validation_dataset,
                device=device,
                batch_size=train_config["val_batch_size"],
                context_length=model_config["context_length"],
                num_batches=train_config["val_batches"]
            )
            logger.info(f"验证集损失: {val_loss:.6f}")
            wandb.log({"val_loss": val_loss, "step": step})

        # checkpoint
        if step % train_config["checkpoint_freq"] == 0:
            ckpt_path = data_paths["checkpoint_save_format"].format(step)
            Path(os.path.dirname(ckpt_path) or ".").mkdir(parents=True, exist_ok=True)
            logger.info(f"正在保存模型检查点到: {ckpt_path}")
            save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=ckpt_path)
            logger.info("模型检查点保存成功。")

    logger.info("模型训练完成。")

    # 保存最终模型
    Path(os.path.dirname(data_paths["final_model_path"]) or ".").mkdir(parents=True, exist_ok=True)
    logger.info(f"正在保存最终模型到: {data_paths['final_model_path']}")
    save_checkpoint(model=model, optimizer=optimizer, iteration=total_steps, out=data_paths["final_model_path"])
    logger.info("最终模型保存成功。")

    wandb.finish()


if __name__ == "__main__":
    main()









