import os
import torch
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ModelSaver:
    """
    模型保存类：根据配置保存模型权重
    
    参数:
        save_dir: 模型保存目录
        save_best_only: 是否只保存最佳模型
        verbose: 是否打印保存信息
    """
    def __init__(self, save_dir='model_weights', save_best_only=True):
        self.save_dir = save_dir
        self.save_best_only = save_best_only #是否只保存最佳模型
        self.best_score = None #最佳验证准确率
        
        # 确保保存目录存在

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir) #创建保存目录
            
    def __call__(self, model, epoch, val_acc):
        """
        保存模型
        
        参数:
            model: 需要保存的模型
            epoch: 当前训练轮数
            val_acc: 当前验证准确率
        """
        # 生成文件名
        filename = f'model_epoch_{epoch}_acc_{val_acc:.4f}.pth'
        save_path = os.path.join(self.save_dir, filename)
        
        # 是否仅保存最佳模型
        if self.save_best_only:
            # 首次调用或验证准确率提高时保存
            if self.best_score is None or val_acc > self.best_score:
                self.best_score = val_acc
                torch.save(model.state_dict(), save_path)
                
                # 删除之前的最佳模型
                for old_file in os.listdir(self.save_dir):
                    if old_file != filename and old_file.endswith('.pth'):
                        os.remove(os.path.join(self.save_dir, old_file))
        else:
            # 每个epoch都保存
            torch.save(model.state_dict(), save_path)


class EarlyStopping:
    """
    早停类：当验证集准确率在一定轮数内不再提高时，停止训练
    
    参数:
        patience: 容忍验证集准确率不提升的轮数
        delta: 判定准确率是否提升的阈值
        verbose: 是否打印早停信息
    """
    def __init__(self, patience:int=5, delta:float=0.0001, verbose:bool=True):
        self.patience = patience #容忍验证集准确率不提升的轮数
        self.delta = delta #判定准确率是否提升的阈值
        self.counter = 0 #记录验证准确率没有提升的轮数
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_acc):
        score = val_acc
        
        # 首次调用时初始化第一次得到的验证准确率
        if self.best_score is None:
            self.best_score = score
            return
        
        # 如果验证准确率没有提升
        if score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience: #如果验证准确率没有提升的轮数大于等于容忍验证集准确率不提升的轮数，则早停
                self.early_stop = True
                print(f"早停触发!")
                print(f"{np.abs(self.best_score):.4f}")
        # 如果验证准确率提升了
        else:
            self.best_score = score #更新最佳验证准确率
            self.counter = 0




# 回归评估模型
def evaluate_regression_model(model, dataloader,  device,criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():#禁止 autograd 记录计算图，节省显存与算力。
        for inputs, targets in dataloader:  # dataloader是数据加载器，包含输入数据和目标数据
            if isinstance(inputs, list):  # 如果datas是tuple,list，则将每个元素都放到device上
                inputs = [x.to(device) for x in inputs]
            else:
                inputs = inputs.to(device)

            targets = targets.to(device)

            # 模型前向计算
            if isinstance(inputs, list):
                outputs = model(*inputs)
            else:
                outputs = model(inputs)

            loss = criterion(outputs, targets) #计算损失
            
            running_loss += loss.item()  #累加损失，inputs.size(0)返回每个批次的样本数量
    
    return running_loss / len(dataloader) #返回平均损失


# 分类评估模型
def evaluate_classification_model(model, data_loader, device, criterion):
    """
    评估模型在给定分类数据集上的准确率和损失
    
    参数:
        model: 需要评估的模型
        data_loader: 数据加载器
        device: 计算设备(CPU/GPU)
        criterion: 损失函数（可选）
    
    返回:
        accuracy: 模型准确率
        avg_loss: 平均损失（如果提供了损失函数）
    """
    model.eval()  # 设置为评估模式
    correct = 0  # 预测正确样本数
    total = 0  # 总样本数
    running_loss = 0.0  # 总损失

    with torch.no_grad():  # 不计算梯度
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # torch.max(outputs.data, 1)返回两个值，第一个是最大值，第二个是最大值的索引
            total += labels.size(0)  # labels.size(0)返回标签的维度，这里返回的是batch_size，因为每个批次有batch_size个标签
            correct += (predicted == labels).sum().item()  # (predicted == labels).sum().item()返回预测正确的标签的个数

            # 如果提供了损失函数，计算损失
            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)  # loss.item()返回损失值，images.size(0)返回每个批次的样本数量

    accuracy = 100 * correct / total  # 计算准确率

    # 如果计算了损失，返回平均损失
    if criterion is not None:
        avg_loss = running_loss / total
        return accuracy, avg_loss

    return accuracy,None


#回归训练模型
def train_regression_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    num_epochs=10, 
    tensorboard_logger=None,
    model_saver=None,
    early_stopping=None,
    eval_step=500
):
    """
    基于tqdm的训练函数，与training函数类似
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        num_epochs: 训练轮次
        tensorboard_callback: Tensorboard回调函数
        model_saver: 保存检查点回调函数
        early_stopping: 早停回调函数
        eval_step: 每多少步评估一次
    
    返回:
        record_dict: 包含训练和验证记录的字典
    """
    record_dict = {
        "train": [],
        "val": []
    }
    
    global_step = 0
    model.train()
    epoch_val_loss=0    #验证集损失

    with tqdm(total=num_epochs * len(train_loader)) as pbar:
        for epoch_id in range(num_epochs): 
            for datas, labels in train_loader:
                labels = labels.to(device)  # 标签放到device上

                if isinstance(datas,list):#如果datas是tuple，则将每个元素都放到device上
                    inputs = [x.to(device) for x in datas]
                else:
                    inputs = datas.to(device)
                
                
                # 梯度清空
                optimizer.zero_grad()
                
                # 模型前向计算
                if isinstance(inputs,list):
                    outputs = model(*inputs)
                else:
                    outputs = model(inputs)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 梯度回传，计算梯度
                loss.backward()
                
                # 更新模型参数
                optimizer.step()
                
                # 更新步骤
                global_step += 1
                
                # 在每个批次后记录训练损失
                epoch_train_loss = loss.item()
                record_dict["train"].append({
                    "loss": epoch_train_loss,
                    "step": global_step
                })
                # 评估
                if global_step % eval_step == 0:
                    epoch_val_loss = evaluate_regression_model(model, val_loader, device, criterion)
                    #  val_acc, val_loss = evaluate(model, val_loader, device,criterion) 分类且上方只需训练一次
                    record_dict["val"].append({
                        "loss": epoch_val_loss, "step": global_step
                    })
                    # 训练
                    model.train()
                    
                    # 保存模型权重
                    # 如果有模型保存器，保存模型
                    if model_saver is not None:
                        model_saver(model, -epoch_val_loss,epoch_id)#保存模型，-epoch_val_loss是因为val_loss越小越好，所以取负数
                    
                    # 如果有早停器，检查是否应该早停
                    if early_stopping is not None:
                        early_stopping(-epoch_val_loss)
                        if early_stopping.early_stop:
                            print(f'早停: 已有{early_stopping.patience}轮验证损失没有改善！')
                            return model,record_dict

                pbar.update(1)
                pbar.set_postfix({"epoch": epoch_id, "loss": f"{loss.item():.4f},{epoch_train_loss:.4f}", "val_loss": f"{epoch_val_loss:.4f}", "step": global_step})
    
    return model,record_dict


#分类训练模型

def train_classification_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=100,
        tensorboard_logger=None,
        model_saver=None,
        early_stopping=None,
        eval_step=500
):
    """
    基于tqdm的训练函数，与training函数类似
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        num_epochs: 训练轮次
        tensorboard_callback: Tensorboard回调函数
        model_saver: 保存检查点回调函数
        early_stopping: 早停回调函数
        eval_step: 每多少步评估一次
    
    返回:
        record_dict: 包含训练和验证记录的字典
    """
    record_dict = {
        "train": [],
        "val": []
    }

    global_step = 0
    model.train()
    print(f"训练开始，共{num_epochs*len(train_loader)}步")
    with tqdm(total=num_epochs * len(train_loader)) as pbar:
        for epoch_id in range(num_epochs):
            # 训练
            for datas, labels in train_loader:
                datas = datas.to(device)  # 数据放到device上
                labels = labels.to(device)  # 标签放到device上

                # 梯度清空
                optimizer.zero_grad()

                # 模型前向计算
                logits = model(datas)

                # 计算损失
                loss = criterion(logits, labels)

                # 梯度回传，计算梯度
                loss.backward()

                # 更新模型参数
                optimizer.step()

                # 计算准确率
                preds = logits.argmax(axis=-1)
                acc = (preds == labels).float().mean().item() * 100
                loss_value = loss.cpu().item()

                # 记录训练数据
                record_dict["train"].append({
                    "loss": loss_value, "acc": acc, "step": global_step
                })

                # 评估
                if global_step % eval_step == 0:
                    val_acc, val_loss = evaluate_classification_model(model, val_loader, device, criterion)
                    record_dict["val"].append({
                        "loss": val_loss, "acc": val_acc, "step": global_step
                    })
                    model.train()  # 切换回训练集模式

                    # 如果有Tensorboard记录器，记录训练和验证指标
                    if tensorboard_logger is not None:
                        tensorboard_logger.log_training(global_step, loss_value, acc)
                        tensorboard_logger.log_validation(global_step, val_loss, val_acc)

                    # 保存模型权重
                    # 如果有模型保存器，保存模型
                    if model_saver is not None:
                        model_saver(model, val_acc, epoch_id)

                    # 如果有早停器，检查是否应该早停
                    if early_stopping is not None:
                        early_stopping(val_acc)
                        if early_stopping.early_stop:
                            print(f'早停: 在{global_step} 步')
                            return model, record_dict

                # 更新步骤
                global_step += 1
                pbar.update(1)
                pbar.set_postfix({"epoch": epoch_id, "loss": f"{loss_value:.4f}", "acc": f"{acc:.2f}%"})

    return model, record_dict


# 画线要注意的是损失是不一定在零到1之间的
def plot_learning_curves(record_dict, sample_step=500,eval_step=1000):
    """
    画学习曲线，横坐标是steps，纵坐标是loss和acc,回归问题只有loss
    
    参数:
        record_dict: 包含训练和验证记录的字典
        sample_step: 每多少步画一个点，默认500步
    """
    train_df = pd.DataFrame(record_dict["train"]).set_index("step").iloc[::sample_step]
    val_df = pd.DataFrame(record_dict["val"]).set_index("step")

    fig_num = len(train_df.columns)  # 因为有loss和acc两个指标，所以画个子图
    fig, axs = plt.subplots(1, fig_num, figsize=(5 * fig_num, 5))  # fig_num个子图，figsize是子图大小
    for idx, item in enumerate(train_df.columns):
        # index是步数，item是指标名字
        axs[idx].plot(train_df.index, train_df[item], label=f"train_{item}")  # 绘制训练曲线
        axs[idx].plot(val_df.index, val_df[item], label=f"val_{item}")        # 绘制验证曲线
        axs[idx].grid()  # 显示网格
        axs[idx].legend()  # 显示图例

        # 根据eval_step动态设置x轴刻度和标签
        if eval_step < 1000:
            # eval_step小于1000时，每eval_step步显示一次
            x_data = range(0, train_df.index[-1] + 1, eval_step)
            axs[idx].set_xticks(x_data)
            axs[idx].set_xticklabels([str(x) for x in x_data])  # 直接显示步数
        else:
            # eval_step大于等于1000时，仍然以k为单位显示
            x_data = range(0, train_df.index[-1] + 1, 5000)
            axs[idx].set_xticks(x_data)
            axs[idx].set_xticklabels([f"{int(x / 1000)}k" for x in x_data])  # 以k为单位显示

        axs[idx].set_xlabel("step")  # 设置x轴标签为英文

    plt.show()


def plot_learning_loss_curves(record_dict, sample_step=500):
    """
    画学习曲线，横坐标是steps，纵坐标是loss和acc,回归问题只有loss

    参数:
        record_dict: 包含训练和验证记录的字典
        sample_step: 每多少步画一个点，默认500步
    """
    train_df = pd.DataFrame(record_dict["train"]).set_index("step").iloc[::sample_step]
    val_df = pd.DataFrame(record_dict["val"]).set_index("step")

    # 只绘制一个loss图，不需要循环
    fig, ax = plt.subplots(figsize=(10, 5))  # 创建单个图表

    # 绘制训练和验证的loss曲线
    ax.plot(train_df.index, train_df['loss'], label="train_loss")
    ax.plot(val_df.index, val_df['loss'], label="val_loss")

    # 设置图表属性
    ax.grid()
    ax.legend()
    x_data = range(0, train_df.index[-1], 5000)  # 每隔5000步标出一个点
    ax.set_xticks(x_data)
    ax.set_xticklabels(map(lambda x: f"{int(x / 1000)}k", x_data))  # map生成label
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("loss curves")

    plt.show()





def evaluate_milti_output_model(model, data_loader, device, criterion):
    """
    评估模型在给定数据集上的准确率和损失
    
    参数:
        model: 需要评估的模型
        data_loader: 数据加载器
        device: 计算设备(CPU/GPU)
        criterion: 损失函数（可选）
    
    返回:
        accuracy: 模型准确率
        avg_loss: 平均损失（如果提供了损失函数）
    """
    model.eval()  # 设置为评估模式
    running_loss = 0.0 #总损失
    
    with torch.no_grad():  # 不计算梯度
        for inputs , labels in data_loader:
            # 处理多输入情况
            if isinstance(inputs, tuple):
                # 将所有输入数据移至设备
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                
                # 模型前向传播
                outputs = model(*inputs)  # 使用解包操作符传递多个输入
            else:
                # 单输入情况
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
            
            output,deep=outputs
            # 处理deep：求平均，reshape为output尺寸，并和output相加
            deep_mean = torch.mean(deep, dim=1)  # 沿着第1维求平均
            deep_reshaped = deep_mean.view_as(output)  # 重塑为与output相同的尺寸
            loss_output = criterion(output, labels)
            loss_deep = criterion(deep_reshaped, labels)
            loss = loss_output + loss_deep  # 总损失为两部分之和
            
            running_loss += loss.item()
    
    avg_loss = running_loss / len(data_loader)  # 计算平均损失
    
    return avg_loss


def train_milti_output_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    num_epochs=10, 
    tensorboard_logger=None,
    model_saver=None,
    early_stopping=None,
    eval_step=500
):
    """
    基于tqdm的训练函数，与training函数类似
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 训练设备
        num_epochs: 训练轮次
        tensorboard_callback: Tensorboard回调函数
        model_saver: 保存检查点回调函数
        early_stopping: 早停回调函数
        eval_step: 每多少步评估一次
    
    返回:
        record_dict: 包含训练和验证记录的字典
    """
    record_dict = {
        "train": [],
        "val": []
    }
    
    global_step = 0
    model.train()
    epoch_val_loss=0
    with tqdm(total=num_epochs * len(train_loader)) as pbar:
        for epoch_id in range(num_epochs): 
            for datas, labels in train_loader:
                # 处理多输入情况，假设inputs是一个包含多个tensor的元组，targets是最后一个元素
                
                if isinstance(datas, list):
                    # 将所有输入数据移至设备
                    inputs = [x.to(device) for x in datas]
                    targets = labels.to(device)
                else:
                    inputs = datas.to(device)  # 数据放到device上
                    targets = labels.to(device)  # 标签放到device上
                
                # 梯度清空
                optimizer.zero_grad()

                # 模型前向计算
                if isinstance(inputs, list):
                    outputs = model(*inputs)
                else:
                    outputs = model(inputs)
                
                # 计算损失
                output, deep = outputs
                
                # 处理deep：求平均，reshape为output尺寸，并和output相加
                deep_mean = torch.mean(deep, dim=1)  # 沿着第1维求平均
                deep_reshaped = deep_mean.view_as(output)  # 重塑为与output相同的尺寸
                
                # 分别计算output和deep_reshaped的损失，然后求和
                loss_output = criterion(output, targets)  # 计算output的损失
                loss_deep = criterion(deep_reshaped, targets)  # 计算deep_reshaped的损失
                loss = loss_output + loss_deep  # 损失求和
                
                # 梯度回传，计算梯度
                loss.backward()
                
                # 更新模型参数
                optimizer.step()

                #更新步骤
                global_step += 1

                # 在每个批次后记录训练损失
                epoch_train_loss = loss.item()
                record_dict["train"].append({
                    "loss": epoch_train_loss,
                    "step": global_step
                })

                # 验证评估
                if global_step % eval_step == 0:
                    epoch_val_loss = evaluate_milti_output_model(model, val_loader, device,criterion)
                    record_dict["val"].append({
                        "loss": epoch_val_loss, "step": global_step
                    })
                    # 训练
                    model.train()
                    # 保存模型权重
                    # 如果有模型保存器，保存模型
                    if model_saver is not None:
                        model_saver(model, -epoch_val_loss,epoch_id)#保存模型，-val_loss是因为val_loss越小越好，所以取负数
                    
                    # 如果有早停器，检查是否应该早停
                    if early_stopping is not None:
                        early_stopping(-epoch_val_loss) #早停器需要取负数
                        if early_stopping.early_stop:
                            print(f'早停: 已有5轮验证损失没有改善！')
                            return model,record_dict
                            
                pbar.update(1)
                pbar.set_postfix({"epoch": epoch_id, "loss": f"{epoch_train_loss:.4f}", "val_loss": f"{epoch_val_loss:.4f}", "step": global_step})
    
    return model,record_dict