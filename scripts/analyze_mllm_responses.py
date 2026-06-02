# MLLM 回复类型统计分析
# 在 Jupyter Notebook 中运行

import os
import json
import glob
from collections import defaultdict
import pandas as pd

# ============================================
# 配置: 修改为你的实际路径
# ============================================
OUTPUT_ROOT = "/path/to/pv_benchmark/outputs/sectors6_50"  # 修改为实际路径

# ============================================
# 数据加载和统计函数
# ============================================

def load_episode_data(episode_json_path):
    """加载单个 episode.json"""
    with open(episode_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_mllm_responses(episode_data):
    """
    从 episode 数据中提取所有 MLLM 回复
    
    Returns:
        list of dict: 每个回复的信息
    """
    responses = []
    
    # 获取样本类型 (positive/hard_negative/easy_negative)
    pair_type = episode_data.get("pair_type", "unknown")
    
    # 遍历所有 step 的 action_log
    action_log = episode_data.get("action_log", [])
    
    for step_idx, step_data in enumerate(action_log):
        # 获取每个 step 的对话记录
        step_dialogues = step_data.get("_step_dialogues", [])
        
        for dialog in step_dialogues:
            # 属性模式: 有 "attr" 和 "parsed_state"
            if "attr" in dialog:
                responses.append({
                    "step": step_idx + 1,
                    "attr_name": dialog.get("attr", ""),
                    "expected": dialog.get("expected", ""),
                    "parsed_answer": dialog.get("parsed_answer", ""),
                    "parsed_state": dialog.get("parsed_state", "Missing"),
                    "reason": dialog.get("reason", ""),
                    "pair_type": pair_type
                })
            # 直接模式: 有 "desc_idx"
            elif "desc_idx" in dialog:
                responses.append({
                    "step": step_idx + 1,
                    "attr_name": f"desc_{dialog.get('desc_idx', '')}",
                    "expected": dialog.get("desc_text", "")[:50] + "...",
                    "parsed_answer": dialog.get("parsed_answer", ""),
                    "parsed_state": dialog.get("parsed_state", "Missing"),
                    "reason": dialog.get("reason", ""),
                    "pair_type": pair_type
                })
    
    return responses

def analyze_agent_folder(agent_folder):
    """
    分析单个 agent 的所有 episode
    
    目录结构: agent_folder/{positive,neg_diff,neg_same}/{correct,wrong}/scene_id/obj_id/episode.json
    
    Returns:
        dict: 统计结果
    """
    stats = {
        "total": 0,
        "by_state": defaultdict(int),  # Matched, Contradictory, Missing
        "by_pair_type": {
            "positive": defaultdict(int),
            "neg_diff": defaultdict(int),
            "neg_same": defaultdict(int)
        }
    }
    
    # 修正: 更深的目录结构
    # agent_folder/{positive,neg_diff,neg_same}/{correct,wrong}/scene_id/obj_id/episode.json
    pattern = os.path.join(agent_folder, "*", "*", "*", "*", "episode.json")
    episode_files = glob.glob(pattern)
    
    print(f"  Found {len(episode_files)} episode files")
    
    for ep_file in episode_files:
        try:
            ep_data = load_episode_data(ep_file)
            
            # 从路径提取 pair_type (positive/neg_diff/neg_same)
            # 结构: .../agent/{pair_type}/{correct_wrong}/scene/obj/episode.json
            path_parts = ep_file.replace("\\", "/").split("/")
            pair_type = None
            for part in path_parts:
                if part in ["positive", "neg_diff", "neg_same"]:
                    pair_type = part
                    break
            
            if not pair_type:
                pair_type = "unknown"
            
            # 遍历 transcript 获取 _step_dialogues
            transcript = ep_data.get("transcript", [])
            for step_data in transcript:
                action = step_data.get("action", {})
                step_dialogues = action.get("_step_dialogues", [])
                
                for dialog in step_dialogues:
                    # 字段名是 "state" 不是 "parsed_state"
                    state = dialog.get("state", dialog.get("parsed_state", "Missing"))
                    
                    stats["total"] += 1
                    stats["by_state"][state] += 1
                    
                    if pair_type in stats["by_pair_type"]:
                        stats["by_pair_type"][pair_type][state] += 1
                    
        except Exception as e:
            print(f"  Error loading {ep_file}: {e}")
    
    return stats


# ============================================
# 从 summary.json 读取准确率
# ============================================

def load_accuracy_summary(output_root=None):
    """
    从各 agent 的 summary.json 读取准确率数据
    
    Returns:
        pd.DataFrame: 包含各 agent 准确率的表格
    """
    if output_root is None:
        output_root = OUTPUT_ROOT
    
    rows = []
    
    for agent_path in sorted(glob.glob(os.path.join(output_root, "*"))):
        if not os.path.isdir(agent_path):
            continue
        
        agent_name = os.path.basename(agent_path)
        summary_file = os.path.join(agent_path, "summary.json")
        
        if not os.path.exists(summary_file):
            print(f"  {agent_name}: summary.json not found")
            continue
        
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            # 总体准确率
            total_acc = summary.get("accuracy", 0)
            total_correct = summary.get("correct_count", 0)
            total_episodes = summary.get("total_episodes", 0)
            
            # 各子集准确率
            per_pair = summary.get("per_pair_type", {})
            
            pos_data = per_pair.get("positive", {})
            neg_diff_data = per_pair.get("neg_diff", {})
            neg_same_data = per_pair.get("neg_same", {})
            
            rows.append({
                "Agent": agent_name,
                "Total_Acc": round(total_acc * 100, 1),
                "Total_Correct": total_correct,
                "Total_Episodes": total_episodes,
                "Pos_Acc": round(pos_data.get("accuracy", 0) * 100, 1),
                "Pos_Correct": pos_data.get("correct", 0),
                "Pos_Total": pos_data.get("total", 0),
                "NegDiff_Acc": round(neg_diff_data.get("accuracy", 0) * 100, 1),
                "NegDiff_Correct": neg_diff_data.get("correct", 0),
                "NegDiff_Total": neg_diff_data.get("total", 0),
                "NegSame_Acc": round(neg_same_data.get("accuracy", 0) * 100, 1),
                "NegSame_Correct": neg_same_data.get("correct", 0),
                "NegSame_Total": neg_same_data.get("total", 0),
            })
            
        except Exception as e:
            print(f"  {agent_name}: Error reading summary.json - {e}")
    
    df = pd.DataFrame(rows)
    
    # 按总准确率排序
    if not df.empty:
        df = df.sort_values("Total_Acc", ascending=False).reset_index(drop=True)
    
    return df


def plot_accuracy_comparison(acc_df, agents_per_page=4):
    """
    绘制准确率对比图 - 分批展示
    
    Args:
        acc_df: load_accuracy_summary 返回的 DataFrame
        agents_per_page: 每页显示的 agent 数量
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    if acc_df.empty:
        print("No accuracy data to plot")
        return []
    
    all_agents = acc_df["Agent"].values
    num_agents = len(all_agents)
    num_pages = math.ceil(num_agents / agents_per_page)
    
    print(f"Total {num_agents} agents, will generate {num_pages} accuracy figures")
    
    figs = []
    
    for page in range(num_pages):
        start_idx = page * agents_per_page
        end_idx = min(start_idx + agents_per_page, num_agents)
        page_df = acc_df.iloc[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"Accuracy Comparison (Page {page+1}/{num_pages})", fontsize=14, fontweight='bold')
        
        x = np.arange(len(page_df))
        width = 0.2
        
        # 简化 agent 名称
        labels = [a.replace("_dino_50", "").replace("_", "\n") for a in page_df["Agent"]]
        
        # 绘制四组柱状图
        bars1 = ax.bar(x - 1.5*width, page_df["Total_Acc"], width, label="Total", color="#3498db")
        bars2 = ax.bar(x - 0.5*width, page_df["Pos_Acc"], width, label="Positive", color="#2ecc71")
        bars3 = ax.bar(x + 0.5*width, page_df["NegDiff_Acc"], width, label="Neg_Diff", color="#e74c3c")
        bars4 = ax.bar(x + 1.5*width, page_df["NegSame_Acc"], width, label="Neg_Same", color="#9b59b6")
        
        # 在柱子上添加数值标签
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 110)
        ax.legend(loc="upper right")
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        save_path = f"accuracy_comparison_page{page+1}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        plt.show()
        figs.append(fig)
    
    return figs

# ============================================
# 主统计逻辑
# ============================================

def run_analysis():
    """运行完整分析"""
    
    # 获取所有 agent 文件夹
    agent_folders = [d for d in glob.glob(os.path.join(OUTPUT_ROOT, "*")) 
                     if os.path.isdir(d)]
    
    print(f"Found {len(agent_folders)} agent folders")
    print("=" * 60)
    
    all_results = {}
    
    for agent_folder in sorted(agent_folders):
        agent_name = os.path.basename(agent_folder)
        print(f"\nAnalyzing: {agent_name}")
        
        stats = analyze_agent_folder(agent_folder)
        all_results[agent_name] = stats
        
        if stats["total"] == 0:
            print("  No data found")
            continue
        
        # 打印总体统计
        print(f"  Total responses: {stats['total']}")
        for state, count in stats["by_state"].items():
            pct = count / stats["total"] * 100
            print(f"    {state}: {count} ({pct:.1f}%)")
        
    return all_results

# ============================================
# 生成汇总表格
# ============================================

def generate_summary_table(all_results):
    """生成汇总 DataFrame"""
    
    rows = []
    
    for agent_name, stats in all_results.items():
        total = stats["total"]
        if total == 0:
            continue
        
        matched = stats["by_state"].get("Matched", 0)
        contra = stats["by_state"].get("Contradictory", 0)
        missing = stats["by_state"].get("Missing", 0)
        
        # 按 pair_type 细分
        for pair_type in ["positive", "neg_diff", "neg_same"]:
            pt_stats = stats["by_pair_type"].get(pair_type, {})
            pt_total = sum(pt_stats.values())
            
            if pt_total > 0:
                rows.append({
                    "Agent": agent_name,
                    "PairType": pair_type,
                    "Total": pt_total,
                    "Matched": pt_stats.get("Matched", 0),
                    "Matched%": pt_stats.get("Matched", 0) / pt_total * 100,
                    "Contradictory": pt_stats.get("Contradictory", 0),
                    "Contra%": pt_stats.get("Contradictory", 0) / pt_total * 100,
                    "Missing": pt_stats.get("Missing", 0),
                    "Missing%": pt_stats.get("Missing", 0) / pt_total * 100,
                })
    
    df = pd.DataFrame(rows)
    return df

# ============================================
# 可视化
# ============================================

def plot_response_distribution(df, agents_per_page=4):
    """绘制回复分布图 - 分批展示"""
    import matplotlib.pyplot as plt
    import math
    
    # 获取所有唯一的 agent
    all_agents = df["Agent"].unique()
    num_agents = len(all_agents)
    num_pages = math.ceil(num_agents / agents_per_page)
    
    print(f"Total {num_agents} agents, will generate {num_pages} figures ({agents_per_page} agents each)")
    
    figs = []
    
    for page in range(num_pages):
        start_idx = page * agents_per_page
        end_idx = min(start_idx + agents_per_page, num_agents)
        page_agents = all_agents[start_idx:end_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"MLLM Response Distribution (Page {page+1}/{num_pages})", fontsize=14, fontweight='bold')
        
        for col_idx, pair_type in enumerate(["positive", "neg_diff", "neg_same"]):
            ax = axes[col_idx]
            
            # 筛选当前页的 agent 和 pair_type
            subset = df[(df["PairType"] == pair_type) & (df["Agent"].isin(page_agents))]
            
            if subset.empty:
                ax.set_title(f"{pair_type}: No Data")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 100)
                continue
            
            # 按 agent 顺序排列
            subset = subset.set_index("Agent").loc[[a for a in page_agents if a in subset["Agent"].values]].reset_index()
            
            agents = subset["Agent"].values
            x = range(len(agents))
            
            matched = subset["Matched%"].values
            contra = subset["Contra%"].values
            missing = subset["Missing%"].values
            
            # 堆叠柱状图
            bars1 = ax.bar(x, matched, label="Matched (Yes)", color="#2ecc71", width=0.6)
            bars2 = ax.bar(x, contra, bottom=matched, label="Contradictory (No)", color="#e74c3c", width=0.6)
            bars3 = ax.bar(x, missing, bottom=matched+contra, label="Missing (Unsure)", color="#f39c12", width=0.6)
            
            ax.set_title(f"{pair_type}", fontsize=12, fontweight='bold')
            ax.set_ylabel("Percentage (%)")
            ax.set_xticks(x)
            
            # 简化 agent 名称显示
            short_names = [a.replace("_dino_50", "").replace("_", "\n") for a in agents]
            ax.set_xticklabels(short_names, rotation=0, ha="center", fontsize=9)
            
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)
            
            # 只在第一个子图显示图例
            if col_idx == 0:
                ax.legend(loc="upper right", fontsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # 给 suptitle 留空间
        
        # 保存图片
        save_path = f"mllm_response_distribution_page{page+1}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        plt.show()
        figs.append(fig)
    
    return figs

# ============================================
# 运行入口
# ============================================

if __name__ == "__main__":
    # 运行分析
    results = run_analysis()
    
    # 生成汇总表
    df = generate_summary_table(results)
    print("\n" + "=" * 60)
    print("Summary Table:")
    print(df.to_string(index=False))
    
    # 保存为 CSV
    df.to_csv("mllm_response_stats.csv", index=False)
    print("\nSaved to mllm_response_stats.csv")
    
    # 可视化 (可选)
    # plot_response_distribution(df)

# ============================================
# Jupyter Notebook 版本: 直接复制下面的代码块运行
# ============================================
"""
# Cell 1: 配置路径
OUTPUT_ROOT = "/path/to/pv_benchmark/outputs/sectors6_50"

# Cell 2: 运行分析
results = run_analysis()

# Cell 3: 查看表格
df = generate_summary_table(results)
df

# Cell 4: 可视化
plot_response_distribution(df)
"""
