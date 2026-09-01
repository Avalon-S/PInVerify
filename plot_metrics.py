import matplotlib.pyplot as plt
import numpy as np

# Data Extraction
# Format: {Split: {Agent: {Metric: [Old, New]}}}
data = {
    "Cat": {
        "DINOv2": {
            "SR": [14.8365, 15.5071],
            "SPL": [7.9426, 8.0906]
        },
        "OWL": {
            "SR": [9.5557, 10.1425],
            "SPL": [4.4057, 4.576]
        }
    },
    "No Cat": {
        "DINOv2": {
            "SR": [23.135, 24.3085],
            "SPL": [11.6077, 12.3371]
        },
        "OWL": {
            "SR": [10.9807, 11.3998],
            "SPL": [5.0263, 5.1207]
        }
    }
}

def plot_metrics(data_dict, output_file="metric_comparison.png"):
    splits = list(data_dict.keys())
    metrics = ["SR", "SPL"]
    agents = ["DINOv2", "OWL"]
    
    # Setup plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plt.subplots_adjust(wspace=0.2)
    
    # Colors
    color_old = '#95a5a6' # Concrete Gray
    color_new = '#2ecc71' # Emerald Green
    
    for idx, split in enumerate(splits):
        ax = axes[idx]
        split_data = data_dict[split]
        
        # Positions
        x = np.arange(len(metrics))  # [0, 1] for SR, SPL
        width = 0.15 # Bar width
        inner_gap = 0.05 # Gap between agents (smaller than metric gap)
        
        # Offsets
        # DINOv2: Shift left by half gap
        offsets = {
            "DINOv2": [-(1.5 * width + inner_gap/2), -(0.5 * width + inner_gap/2)],
            "OWL": [(0.5 * width + inner_gap/2), (1.5 * width + inner_gap/2)]
        }
        
        for agent in agents:
            stats = split_data[agent]
            # Plot SR and SPL
            for m_idx, metric in enumerate(metrics):
                vals = stats[metric] # [Old, New]
                old_val, new_val = vals[0], vals[1]
                diff = new_val - old_val
                
                # Positions
                pos_old = x[m_idx] + offsets[agent][0]
                pos_new = x[m_idx] + offsets[agent][1]
                
                # Draw bars
                rects1 = ax.bar(pos_old, old_val, width, label=f'{agent} (Old)' if m_idx==0 else "", color=color_old, alpha=0.7, edgecolor='white')
                rects2 = ax.bar(pos_new, new_val, width, label=f'{agent} (New)' if m_idx==0 else "", color=color_new, edgecolor='white')
                
                # Add text annotations
                # For improvement
                ax.annotate(f'+{diff:.2f}',
                            xy=(pos_new, new_val),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, fontweight='bold', color='black')
                
                # Text for Agent Name below the bars
                center_pos = x[m_idx] + (offsets[agent][0] + offsets[agent][1])/2
                ax.text(center_pos, -2, agent, ha='center', fontsize=10, fontweight='bold')

        # Formatting
        ax.set_title(f"Performance Comparison ({split})", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, max([max(d['DINOv2']['SR'] + d['DINOv2']['SPL'] + d['OWL']['SR'] + d['OWL']['SPL']) for d in [split_data]]) * 1.5) # Dynamic ylim? Just ample space.
        
        # Grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Global Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=color_old, lw=4),
                    Line2D([0], [0], color=color_new, lw=4)]
    fig.legend(custom_lines, ['Original (v1)', 'Repaired (v2)'], loc='upper center', ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300)
    print(f"Comparison plot saved to {output_file}")

if __name__ == "__main__":
    plot_metrics(data)
