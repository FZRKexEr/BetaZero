import os
import sys
import numpy as np
import torch
import random  # 导入 random 模块
import re # 导入正则表达式模块
from datetime import datetime # 导入 datetime 用于比较时间戳

# 设置matplotlib使用非交互式后端，避免线程问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path # 用于处理文件路径和创建目录
import concurrent.futures
from tqdm import tqdm
import threading  # 添加线程模块
import time  # 用于性能测量

# 创建matplotlib的线程锁
plt_lock = threading.Lock()

# === 修改：注释掉中文设置，恢复默认或指定英文字体（可选） ===
try:
    # plt.rcParams['font.sans-serif'] = ['SimHei'] # 注释掉中文设置
    # plt.rcParams['axes.unicode_minus'] = False # 可以保留或注释掉
    # print("尝试设置字体为 SimHei") # 修改提示
    pass # 保持安静
except Exception as e:
    # print(f"设置 SimHei 字体失败: {e}") # 修改提示
    # print("请检查您的系统是否安装了支持中文的字体，并在代码中指定正确的字体名称。")
    pass # 字体设置失败时静默处理或打印不同信息
# === 结束修改 ===

from data_handler import DataHandler

def print_board_state(state, policy_np, board_size, prob_threshold=0.01):
    """将状态张量和策略概率可视化为棋盘"""
    black_stones = state[0].numpy()
    white_stones = state[1].numpy()
    # 假设 state[2] 的所有值都相同，取第一个元素判断当前玩家
    current_player_indicator = state[2][0][0].item()
    # === 修改：使用第 4 个通道 (索引 3) 作为最近一步棋的位置平面 ===
    last_move_plane = state[3].numpy() # 最近一步棋的位置平面

    is_black_turn = current_player_indicator == 1
    current_player = "黑方 (X)" if is_black_turn else "白方 (O)"
    # 最近一步棋是对手下的
    last_move_color_symbol = 'O' if is_black_turn else 'X'

    # 查找最近一步棋的位置
    last_move_pos = None
    # === 检查修改后的 last_move_plane ===
    if np.sum(last_move_plane) > 0:
        # 找到值为1的索引
        last_move_indices = np.argwhere(last_move_plane == 1)
        if len(last_move_indices) > 0:
            # === 添加检查：确保只有一个最近步 ===
            if len(last_move_indices) > 1:
                print(f"  警告：在最近一步平面 (state[3]) 中找到多个标记位置！将使用第一个：{last_move_indices}")
            last_move_pos = tuple(last_move_indices[0]) # (row, col)
        # === (可选) 添加 else 分支处理平面有和但找不到索引的情况 ===
        # else:
        #     print(f"  警告：最近一步平面 (state[3]) 的和大于0，但 argwhere 未找到值为1的索引。Sum: {np.sum(last_move_plane)}")

    # === 设置单元格宽度 ===
    cell_width = 6
    board = [['+' for _ in range(board_size)] for _ in range(board_size)]
    last_move_display = "无" # 默认值

    for r in range(board_size):
        for c in range(board_size):
            is_last_move = last_move_pos is not None and (r, c) == last_move_pos
            symbol = '+'
            display_content = "" # 用于存储最终在单元格显示的内容

            if black_stones[r, c] == 1:
                symbol = 'X'
            elif white_stones[r, c] == 1:
                symbol = 'O'

            if is_last_move:
                # === 格式化最近一步 ===
                display_content = f"({last_move_color_symbol})"
                col_label = chr(ord('A') + c)
                row_label = board_size - r
                last_move_display = f"{last_move_color_symbol} at {col_label}{row_label} (坐标: {r}, {c})"
            elif symbol != '+':
                # === 格式化普通棋子 ===
                display_content = symbol
            else:
                # === 处理空位：检查策略概率 ===
                idx = r * board_size + c
                prob = policy_np[idx]
                if prob >= prob_threshold:
                    # === 格式化概率 ===
                    display_content = "{:.2f}".format(prob) # 保留两位小数
                else:
                    display_content = '+'

            # === 填充并居中显示 ===
            board[r][c] = display_content.center(cell_width)

    # 打印棋盘
    print(f"\n当前轮到: {current_player}")
    # === 使用计算出的 last_move_display ===
    print(f"最近一步: {last_move_display}")
    # === 调整分隔线长度 ===
    separator = "-" * (board_size * cell_width + 4)
    print(separator)

    # === 调整列标格式 ===
    header = "   " + "".join([f"{chr(ord('A') + c)}".center(cell_width) for c in range(board_size)])
    print(header)
    # === 添加顶部空行 ===
    print() # 在列标和第一行之间加一个空行

    # 打印棋盘行和行标 (15, 14, ..., 1)
    for r in range(board_size):
        row_str = f"{board_size - r:2d} " # 行标，右对齐
        row_str += "".join(board[r]) # 连接所有单元格内容
        row_str += f" {board_size - r:2d}" # 右侧行标
        print(row_str)
        # === 在每行数据后添加空行 ===
        # === 但最后一行后面不需要再加空行，避免底部间距过大 ===
        if r < board_size - 1:
            print() # 打印一个空行来增加垂直间距

    # === 添加底部空行 ===
    print() # 在最后一行和列标之间加一个空行
    print(header)
    print(separator)

def save_board_image(state, policy_np, board_size, move_number, value, filename, prob_threshold=0.005):
    """生成高质量五子棋棋盘分析图"""
    fig = None # 初始化 fig 变量
    try:
        # 只在创建figure和图表绘制时使用线程锁，分离数据准备部分
        black_stones_np = state[0].numpy()
        white_stones_np = state[1].numpy()
        current_player_indicator = state[2][0][0].item()
        last_move_plane_np = state[3].numpy()

        is_black_turn = current_player_indicator == 1
        current_player = "Black (X)" if is_black_turn else "White (O)"
        last_move_color = 'X' if not is_black_turn else 'O'

        last_move_board_coord = None
        last_move_plot_coord = None
        last_move_display = "None"

        last_move_indices = np.argwhere(last_move_plane_np == 1)
        if len(last_move_indices) > 0:
            last_move_board_coord = tuple(last_move_indices[0])
            r, c = last_move_board_coord
            last_move_plot_coord = (c, board_size - 1 - r)
            col_label = chr(ord('A') + c)
            row_label = board_size - r
            last_move_display = f"{last_move_color} at {col_label}{row_label} ({r}, {c})"

        # 最小化锁定区域 - 只锁定matplotlib图形创建部分
        with plt_lock:
            # 创建高质量棋盘图形
            plt.rcParams['figure.dpi'] = 300
            fig = plt.figure(figsize=(10, 10), constrained_layout=True)
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            
            # 设置精美的木质纹理背景
            wood_color = '#E2BD83' # 精选木质色调
            border_color = '#8B5A2B' # 深棕色边框
            
            # 创建整个棋盘的底色和边框
            board_rect = patches.Rectangle((-1, -1), board_size + 1, board_size + 1, 
                                         facecolor=wood_color, edgecolor=border_color,
                                         linewidth=2, zorder=-2)
            ax.add_patch(board_rect)
            
            # 创建高品质网格纹理效果
            for i in range(board_size):
                # 垂直线
                ax.plot([i, i], [0, board_size-1], color='#442200', 
                       linewidth=0.8, alpha=0.7, zorder=-1)
                # 水平线
                ax.plot([0, board_size-1], [i, i], color='#442200', 
                       linewidth=0.8, alpha=0.7, zorder=-1)
            
            # 棋盘外层增加装饰边框
            outer_border = patches.Rectangle((-1.2, -1.2), board_size + 2.4, board_size + 2.4,
                                           fill=False, edgecolor=border_color,
                                           linewidth=3, zorder=-3)
            ax.add_patch(outer_border)
            
            # 创建细致的星位点
            hoshi_positions = [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), 
                              (7, 11), (11, 3), (11, 7), (11, 11)]
            for r, c in hoshi_positions:
                x, y = c, board_size - 1 - r
                # 使用小圆点而非方块，更符合围棋风格
                hoshi = patches.Circle((x, y), radius=0.13, facecolor='#442200', 
                                     edgecolor=None, alpha=0.9, zorder=0)
                ax.add_patch(hoshi)
                
            # 获取最高概率值用于颜色归一化
            max_prob = np.max(policy_np)
            min_prob = max(0.01, np.min(policy_np[policy_np > 0]))
            
            # 创建自定义颜色映射
            def get_prob_color(prob, max_val=max_prob):
                # 创建从冷色到暖色的渐变
                if prob < 0.02:
                    return 'lightgray', 0.5  # 非常低的概率用浅灰色
                elif prob < 0.1:
                    # 蓝色调
                    ratio = (prob - 0.02) / 0.08
                    return plt.cm.Blues(0.5 + ratio * 0.5), 0.7
                elif prob < 0.2:
                    # 绿色调
                    ratio = (prob - 0.1) / 0.1
                    return plt.cm.Greens(0.6 + ratio * 0.4), 0.8
                else:
                    # 红色调，概率越高颜色越强
                    ratio = min(1.0, (prob - 0.2) / (max_val - 0.2) * 1.3)
                    return plt.cm.Reds(0.5 + ratio * 0.5), 1.0
                
            # 绘制棋子和概率分布
            stone_radius = 0.43  # 棋子半径
            for r in range(board_size):
                for c in range(board_size):
                    x, y = c, board_size - 1 - r
                    idx = r * board_size + c
                    
                    is_black = black_stones_np[r, c] == 1
                    is_white = white_stones_np[r, c] == 1
                    
                    if is_black:
                        # 黑棋 - 添加逼真的3D效果
                        # 阴影效果
                        shadow = patches.Ellipse((x+0.08, y-0.08), stone_radius*2.05, 
                                              stone_radius*1.95, color='black', alpha=0.3, zorder=1)
                        ax.add_patch(shadow)
                        
                        # 主体
                        stone = patches.Circle((x, y), radius=stone_radius, 
                                             facecolor='black', edgecolor='#333333', 
                                             linewidth=0.7, zorder=2)
                        ax.add_patch(stone)
                        
                        # 光泽
                        highlight1 = patches.Ellipse((x-0.15, y+0.15), stone_radius*0.8, 
                                                   stone_radius*0.5, angle=-45, 
                                                   facecolor='white', alpha=0.3, zorder=3)
                        highlight2 = patches.Ellipse((x-0.1, y+0.1), stone_radius*0.3, 
                                                   stone_radius*0.2, angle=-45, 
                                                   facecolor='white', alpha=0.5, zorder=3)
                        ax.add_patch(highlight1)
                        ax.add_patch(highlight2)
                        
                    elif is_white:
                        # 白棋 - 增强3D立体感
                        # 阴影效果
                        shadow = patches.Ellipse((x+0.08, y-0.08), stone_radius*2.05, 
                                              stone_radius*1.95, color='black', alpha=0.2, zorder=1)
                        ax.add_patch(shadow)
                        
                        # 外层光晕
                        outer_glow = patches.Circle((x, y), radius=stone_radius*1.02, 
                                                  facecolor='none', edgecolor='#888888', 
                                                  linewidth=0.7, alpha=0.5, zorder=2)
                        ax.add_patch(outer_glow)
                        
                        # 主体
                        stone = patches.Circle((x, y), radius=stone_radius, 
                                             facecolor='white', edgecolor='#555555', 
                                             linewidth=0.5, zorder=4)
                        ax.add_patch(stone)
                        
                        # 光泽和反光
                        highlight = patches.Ellipse((x-0.1, y+0.1), stone_radius*0.7, 
                                                  stone_radius*0.4, angle=-45, 
                                                  facecolor='white', alpha=0.7, zorder=5)
                        ax.add_patch(highlight)
                        
                    else:
                        # 空位，显示策略概率
                        prob = policy_np[idx]
                        if prob >= prob_threshold:
                            color, alpha = get_prob_color(prob)
                            
                            # 使用圆形热力图表示概率
                            heat_radius = min(0.4, 0.2 + prob * 0.8)  # 根据概率调整大小
                            heat = patches.Circle((x, y), radius=heat_radius,
                                               facecolor=color, alpha=alpha*0.4, zorder=1)
                            ax.add_patch(heat)
                            
                            # 概率文本，大小和颜色基于概率值
                            font_size = 7 + prob * 15  # 动态字体大小
                            weight = 'bold' if prob > 0.1 else 'normal'
                            
                            # 为高概率值选择更清晰的对比色
                            if prob > 0.2:
                                text_color = 'black' if isinstance(color, tuple) else 'black'
                            else:
                                text_color = 'black'
                                
                            ax.text(x, y, f"{prob:.2f}", ha='center', va='center',
                                  fontsize=font_size, color=text_color, 
                                  weight=weight, zorder=2)
            
            # 标记最后一步落子位置 - 增强视觉效果
            if last_move_plot_coord:
                x, y = last_move_plot_coord
                
                # 创建多层光晕效果
                for i in range(3):
                    size = 0.45 - i * 0.08
                    glow = patches.Rectangle((x - size/2, y - size/2), 
                                          size, size, 
                                          edgecolor='red', fill=False,
                                          linewidth=3-i, alpha=0.6-i*0.15, zorder=10)
                    ax.add_patch(glow)
                    
                # 添加醒目的红色方框
                marker = patches.Rectangle((x - 0.25, y - 0.25), 
                                         0.5, 0.5, 
                                         edgecolor='red', fill=False,
                                         linewidth=2.5, zorder=11)
                ax.add_patch(marker)
                
                # 可选：添加动态效果线
                for angle in [45, 135, 225, 315]:
                    dx = 0.4 * np.cos(np.radians(angle))
                    dy = 0.4 * np.sin(np.radians(angle))
                    ax.plot([x, x+dx], [y, y+dy], color='red', 
                           linestyle='-', linewidth=1, alpha=0.5, zorder=9)
            
            # 设置坐标轴和显示范围
            ax.set_xlim(-1.5, board_size + 0.5)
            ax.set_ylim(-1.5, board_size + 0.5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # 添加更精美的坐标标签
            label_color = '#442200'
            for i in range(board_size):
                # 更大、更清晰的标签
                ax.text(i, -0.7, chr(ord('A') + i), ha='center', va='center',
                      fontsize=11, color=label_color, fontweight='bold')
                ax.text(i, board_size-0.3, chr(ord('A') + i), ha='center', va='center',
                      fontsize=11, color=label_color, fontweight='bold')
                
                ax.text(-0.7, i, str(board_size - i), ha='center', va='center',
                      fontsize=11, color=label_color, fontweight='bold')
                ax.text(board_size-0.3, i, str(board_size - i), ha='center', va='center',
                      fontsize=11, color=label_color, fontweight='bold')
                
            # 添加更专业的标题信息
            title_color = '#442200'
            
            # 主标题
            fig.text(0.5, 0.975, f"Move: {move_number}   Player to Play: {current_player}", 
                   fontsize=14, ha='center', color=title_color, fontweight='bold')
            
            # 副标题
            fig.text(0.5, 0.955, f"Value: {value:.4f}   Last Move: {last_move_display}", 
                   fontsize=12, ha='center', color=title_color)
            
            # 保存高质量图像
            plt.savefig(str(filename), dpi=300, bbox_inches='tight', 
                       facecolor=fig.get_facecolor())
            plt.close(fig)
        
        # 移动到锁之外，减少锁定时间
        print(f"  生成高质量棋盘图片: {filename}")
        
    except Exception as e:
        print(f"Error during enhanced board image generation ({filename}): {e}")
        if 'fig' in locals() and fig is not None:
            with plt_lock:
                if plt.fignum_exists(fig.number):
                    plt.close(fig)

def save_board_image_thread(args):
    """线程函数：保存单个棋盘图像"""
    state, policy_np, board_size, move_number, value_item, image_filename = args
    try:
        # 直接调用save_board_image并传递参数
        save_board_image(state, policy_np, board_size, move_number, value_item, image_filename)
        return True, move_number, image_filename
    except Exception as e:
        # 捕获并返回任何异常
        error_msg = f"错误(move_{move_number:03d}): {str(e)}"
        return False, move_number, error_msg

def save_game_images(game_data, game_description_prefix, board_size, output_dir, max_workers=20):
    """使用多线程为一个完整对局的每一步保存棋盘图片"""
    print(f"\n===== 正在为 '{game_description_prefix}' 对局生成图片 (共 {len(game_data)} 步) =====")
    
    # 为每个特定对局创建独立的子目录
    game_output_dir = output_dir / game_description_prefix
    game_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  图片将保存至: {game_output_dir}")
    
    # 准备任务列表 - 每个任务完全独立，无需共享状态
    tasks = []
    for move_index, (state, policy, value) in enumerate(game_data):
        move_number = move_index + 1
        policy_np = policy.numpy()  # 预先转换为numpy，减少线程中的工作
        value_item = value.item()
        image_filename = game_output_dir / f"move_{move_number:03d}.png"
        
        # 打包任务参数
        task = (state, policy_np, board_size, move_number, value_item, image_filename)
        tasks.append(task)
    
    # 使用内存信号量限制并发任务数量，避免内存溢出
    import threading
    # 限制同时在内存中的棋盘数据量，避免内存过度使用
    memory_semaphore = threading.Semaphore(max(1, min(max_workers, 8)))
    
    # 封装带有信号量的线程函数
    def memory_controlled_task(args):
        with memory_semaphore:
            start_time = time.time()
            result = save_board_image_thread(args)
            end_time = time.time()
            # 返回结果和耗时
            return result + (end_time - start_time,)

    # 使用线程池执行任务
    successful = 0
    failed = 0
    
    # 使用合适数量的线程
    actual_workers = min(max_workers, len(tasks))
    print(f"  使用 {actual_workers} 个工作线程进行并行处理")
    
    # 记录性能数据
    start_total = time.time()
    task_times = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(memory_controlled_task, task) for task in tasks]
        
        # 使用带有柔和刷新频率的tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(tasks), 
                          desc=f"生成 {game_description_prefix} 图片",
                          mininterval=0.5,  # 减少刷新频率
                          smoothing=0.1):   # 平滑进度条
            success, move_number, result, task_time = future.result()
            task_times.append(task_time)
            if success:
                successful += 1
            else:
                failed += 1
                print(f"  步骤 {move_number} 生成失败: {result}")
    
    # 计算性能统计
    total_time = time.time() - start_total
    serial_time = sum(task_times)
    speedup = serial_time / total_time if total_time > 0 else 0
    avg_task_time = sum(task_times) / len(task_times) if task_times else 0
    
    print(f"===== '{game_description_prefix}' 对局图片生成完成: 成功 {successful} 张, 失败 {failed} 张 =====")
    print(f"  性能统计:")
    print(f"    总耗时: {total_time:.2f}秒")
    print(f"    串行等效耗时: {serial_time:.2f}秒")
    print(f"    平均每张图片: {avg_task_time:.2f}秒")
    print(f"    加速比: {speedup:.2f}x")
    print(f"    处理率: {successful/total_time:.2f}张/秒")
    
    return total_time, serial_time, speedup, successful

def plot_game_length_distribution(game_lengths, filename):
    """绘制对局步数分布直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(game_lengths, bins='auto', align='left', rwidth=0.8)
    plt.xlabel("Number of Moves")
    plt.ylabel("Number of Games")
    plt.title(f"Game Length Distribution (Total {len(game_lengths)} Games)")
    
    # 添加统计信息
    mean = np.mean(game_lengths)
    std = np.std(game_lengths)
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1)
    plt.text(0.98, 0.95, 
             f'Mean: {mean:.1f}\nStd: {std:.1f}', 
             transform=plt.gca().transAxes,
             horizontalalignment='right',
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.savefig(str(filename), dpi=150, bbox_inches='tight')
    plt.close()

def test_data_reading(max_threads=20):
    """分析对局步数分布并绘制特征对局"""
    # 配置参数
    config = {
        "data_dir": "data",
        "board_size": 15,
        "input_channels": 8,
        "batch_size": 1024,
        "num_workers": 2,
        "memory_size": 10
    }

    # 初始化数据处理器
    data_handler = DataHandler(config)

    # 查找最新的数据文件 (代码与之前相同)
    latest_file = None
    latest_timestamp = None
    pattern = re.compile(r"selfplay_(\d{8})_(\d{4})\.data")
    print(f"在目录 '{config['data_dir']}' 中搜索最新的数据文件...")
    for root, _, files in os.walk(config["data_dir"]):
        for file in files:
            match = pattern.match(file)
            if match:
                date_str, time_str = match.groups()
                try:
                    current_timestamp = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
                    file_path = os.path.join(root, file)
                    if latest_timestamp is None or current_timestamp > latest_timestamp:
                        latest_timestamp = current_timestamp
                        latest_file = file_path
                except ValueError:
                    print(f"  警告：文件名 '{file}' 中的日期时间格式无效，已跳过。")

    if latest_file is None:
        print("未找到任何符合格式 'selfplay_YYYYMMDD_HHMM.data' 的数据文件。")
        return

    test_file = latest_file
    print(f"\n选择最新的数据文件进行测试: {test_file}")
    print(f"文件时间戳: {latest_timestamp.strftime('%Y-%m-%d %H:%M')}")

    # === 创建本次运行的输出目录 (带时间戳) ===
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 将所有图片保存在 ./images/{timestamp_str}/ 目录下
    output_image_dir = Path("./images") / timestamp_str
    output_image_dir.mkdir(parents=True, exist_ok=True)
    print(f"所有图片将保存至: {output_image_dir}")

    # === 确定 .data 文件信息 ===
    base_filename = Path(test_file).stem
    print(f"\n分析数据文件: {test_file}")

    # 读取数据
    examples = data_handler._read_data_file(test_file)

    if not examples:
        print("读取数据失败或数据为空")
        return

    print(f"共读取 {len(examples)} 个样本")
    if len(examples) % 16 != 0:
         print("警告：样本总数不是 16 的倍数，数据可能不完整或与预期结构不符。")

    # === 提取所有对局 ===
    all_games = []
    all_game_indices = []  # 保存每个对局第一步在examples中的索引
    current_game = []
    current_game_indices = []
    game_lengths = []
    tengen_game = None  # 用于存储第一步落在天元的对局
    tengen_game_indices = None  # 其对应的样本索引

    # 遍历所有原始样本
    for i in range(0, len(examples), 16):
        state, policy, value = examples[i]
        is_empty_board = (torch.sum(state[0]) == 0 and torch.sum(state[1]) == 0)

        if is_empty_board:
            if current_game:
                game_lengths.append(len(current_game))
                all_games.append(current_game)
                all_game_indices.append(current_game_indices)
                
                # 检查这个对局的第二步（第一手）是否落在天元
                # 第一步是空棋盘，第二步才是第一手落子
                if len(current_game) > 1 and tengen_game is None:
                    first_move_state = current_game[1][0]  # 获取第一手的状态
                    # 检查黑子是否落在天元
                    if first_move_state[0][7][7] == 1:  # 假设黑子在channel 0
                        tengen_game = current_game
                        tengen_game_indices = current_game_indices
            
            current_game = [(state, policy, value)]
            current_game_indices = [i]
        else:
            if current_game:
                current_game.append((state, policy, value))
                current_game_indices.append(i)
                assert len(current_game) <= 225, f"错误：对局步数超过225步！"

    # 处理最后一个对局
    if current_game:
        game_lengths.append(len(current_game))
        all_games.append(current_game)
        all_game_indices.append(current_game_indices)
        # 同样检查最后一个对局
        if len(current_game) > 1 and tengen_game is None:
            first_move_state = current_game[1][0]
            if first_move_state[0][7][7] == 1:
                tengen_game = current_game
                tengen_game_indices = current_game_indices

    # === 分析步数分布 ===
    game_lengths = np.array(game_lengths)
    min_len = np.min(game_lengths)
    max_len = np.max(game_lengths)
    avg_len = np.mean(game_lengths)
    
    print(f"对局数量: {len(all_games)}")
    print(f"步数统计: 最短 {min_len}, 最长 {max_len}, 平均 {avg_len:.1f}")

    # === 绘制步数分布图 ===
    dist_plot_filename = output_image_dir / "game_length_distribution.png"
    plot_game_length_distribution(game_lengths, dist_plot_filename)

    # === 找出特征对局 ===
    min_idx = np.argmin(game_lengths)
    max_idx = np.argmax(game_lengths)
    avg_idx = np.abs(game_lengths - avg_len).argmin()  # 找到最接近平均值的对局

    # === 保存特征对局 ===
    board_size = config["board_size"]
    
    # 性能统计收集
    perf_stats = []
    
    # 使用多线程版本保存对局图片
    print("\n===== 保存原始棋局图像 =====")
    stats1 = save_game_images(all_games[min_idx], 
                    f"shortest_game_len_{min_len}", 
                    board_size, output_image_dir, max_workers=max_threads)
    perf_stats.append(("最短对局", stats1))

    stats2 = save_game_images(all_games[max_idx], 
                    f"longest_game_len_{max_len}", 
                    board_size, output_image_dir, max_workers=max_threads)
    perf_stats.append(("最长对局", stats2))

    stats3 = save_game_images(all_games[avg_idx], 
                    f"average_game_len_{avg_len}", 
                    board_size, output_image_dir, max_workers=max_threads)
    perf_stats.append(("平均长度对局", stats3))

    if tengen_game is not None:
        tengen_game_len = len(tengen_game)
        stats4 = save_game_images(tengen_game, 
                        f"tengen_game_len_{tengen_game_len}", 
                        board_size, output_image_dir, max_workers=max_threads)
        perf_stats.append(("天元开局", stats4))
        print("找到并保存了一局天元开局的对局")
    else:
        print("未找到天元开局的对局")
        
    # === 提取并保存最短对局的所有16种数据增强变体 ===
    print("\n===== 开始提取最短对局的所有数据增强变体 =====")
    shortest_game_indices = all_game_indices[min_idx]
    print(f"最短对局共 {len(shortest_game_indices)} 步，将提取所有16种数据增强")
    
    # 提取并保存16种数据增强变体
    augmented_games = extract_augmented_games(
        examples,
        shortest_game_indices,
        board_size,
        output_image_dir,
        max_threads
    )

    # === 显示总体性能统计 ===
    print("\n====== 性能统计总结 ======")
    total_imgs = 0
    total_parallel_time = 0
    total_serial_time = 0
    
    for name, (parallel_time, serial_time, speedup, n_imgs) in perf_stats:
        total_imgs += n_imgs
        total_parallel_time += parallel_time
        total_serial_time += serial_time
        
    overall_speedup = total_serial_time / total_parallel_time if total_parallel_time > 0 else 0
    
    print(f"总图片数量: {total_imgs} 张")
    print(f"总并行处理时间: {total_parallel_time:.2f} 秒")
    print(f"总串行等效时间: {total_serial_time:.2f} 秒")
    print(f"总体加速比: {overall_speedup:.2f}x")
    print(f"总体处理率: {total_imgs/total_parallel_time:.2f} 张/秒")
    
    # 显示每个对局的性能
    print("\n各对局处理性能:")
    for name, (parallel_time, serial_time, speedup, n_imgs) in perf_stats:
        print(f"  {name}: {n_imgs}张图片, 加速比 {speedup:.2f}x, 处理率 {n_imgs/parallel_time:.2f}张/秒")
    
    print("\n====== 图片生成流程结束 ======")
    
    return augmented_games  # 返回所有增强变体供后续分析

def extract_augmented_games(examples, original_game_indices, board_size, output_dir, max_threads=20):
    """提取和保存一个对局的所有16种数据增强变体"""
    print(f"\n===== 正在提取并保存对局的所有16种数据增强变体 =====")
    
    # 创建输出目录
    augmented_dir = output_dir / "augmented_shortest_game"
    augmented_dir.mkdir(parents=True, exist_ok=True)
    print(f"  数据增强变体将保存至: {augmented_dir}")
    
    # 重组16种变体的对局
    augmented_games = [[] for _ in range(16)]  # 每个元素对应一种数据增强
    
    # 遍历原始对局的每一步，收集所有对应的数据增强
    for step_idx, orig_idx in enumerate(original_game_indices):
        # 获取此步骤的所有16个数据增强样本
        for aug_idx in range(16):
            example_idx = orig_idx + aug_idx
            if example_idx < len(examples):
                state, policy, value = examples[example_idx]
                augmented_games[aug_idx].append((state, policy, value))
    
    # 确认所有变体的长度一致
    game_lengths = [len(game) for game in augmented_games]
    print(f"  已提取16种变体，步数: {game_lengths[0]}")
    if len(set(game_lengths)) > 1:
        print(f"  警告：不同变体的步数不一致: {game_lengths}")
    
    # 保存所有16种变体
    perf_stats = []
    
    # 保存原始变体
    stats0 = save_game_images(augmented_games[0], 
                   f"aug_0_original", 
                   board_size, augmented_dir, max_workers=max_threads)
    perf_stats.append(("原始对局", stats0))
    
    # 保存其余15种数据增强变体
    for aug_idx in range(1, 16):
        aug_name = f"aug_{aug_idx}"
        stats = save_game_images(augmented_games[aug_idx], 
                       aug_name, 
                       board_size, augmented_dir, max_workers=max_threads)
        perf_stats.append((f"数据增强 #{aug_idx}", stats))
    
    # 显示性能统计
    total_imgs = 0
    total_parallel_time = 0
    total_serial_time = 0
    
    for name, (parallel_time, serial_time, speedup, n_imgs) in perf_stats:
        total_imgs += n_imgs
        total_parallel_time += parallel_time
        total_serial_time += serial_time
        
    overall_speedup = total_serial_time / total_parallel_time if total_parallel_time > 0 else 0
    
    print(f"\n===== 数据增强变体保存完成 =====")
    print(f"  总图片数量: {total_imgs} 张")
    print(f"  总体加速比: {overall_speedup:.2f}x")
    print(f"  总体处理率: {total_imgs/total_parallel_time:.2f} 张/秒")
    
    return augmented_games

if __name__ == "__main__":
    # 自动检测CPU核心数，设置合适的线程数
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    max_recommended_threads = 20
    print(f"系统有 {n_cores} 个CPU核心，推荐最大线程数: {max_recommended_threads}")
    
    # 运行测试，使用推荐的线程数
    test_data_reading(max_threads=max_recommended_threads)