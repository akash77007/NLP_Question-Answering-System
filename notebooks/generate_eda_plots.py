"""
notebooks/generate_eda_plots.py
Automated script to generate EDA charts and save as image for presentations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data_path = os.path.join("data", "squad_2.0_dev_complete_11873.xlsx")
    if not os.path.exists(data_path):
        data_path = os.path.join("..", "data", "squad_2.0_dev_complete_11873.xlsx")
    
    print(f"Loading {data_path}...")
    df = pd.read_excel(data_path)
    
    # Clean context
    def clean_ctx(x):
        s = str(x)
        if s.startswith("['") and s.endswith("']"):
            return s[2:-2]
        return s

    df["clean_context"] = df["context"].apply(clean_ctx)
    df["ctx_len"] = df["clean_context"].apply(lambda x: len(str(x).split()))
    df["q_len"] = df["question"].apply(lambda x: len(str(x).split()))
    
    class_counts = df["is_impossible"].value_counts()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Class Distribution
    colors = ['#10B981', '#EF4444']
    axes[0, 0].bar(['Answerable (False)', 'Unanswerable (True)'], 
                   [class_counts[False], class_counts[True]], 
                   color=colors, edgecolor='black', width=0.55)
    axes[0, 0].set_title("SQuAD 2.0 Class Distribution", fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel("Number of Samples")
    for i, v in enumerate([class_counts[False], class_counts[True]]):
        axes[0, 0].text(i, v + 120, f"{v}\n({v/len(df)*100:.1f}%)", ha='center', fontweight='bold', fontsize=10)
    axes[0, 0].set_ylim(0, max(class_counts) * 1.15)

    # 2. Context Length Distribution
    axes[0, 1].hist(df["ctx_len"], bins=40, color='#3B82F6', edgecolor='black', alpha=0.85)
    q95_ctx = df["ctx_len"].quantile(0.95)
    axes[0, 1].axvline(q95_ctx, color='red', linestyle='--', linewidth=2, label=f"95th %ile ({int(q95_ctx)} words)")
    axes[0, 1].set_title("Context Length Distribution (Word Count)", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel("Words per Passage")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    # 3. Question Length Distribution
    axes[1, 0].hist(df["q_len"], bins=30, color='#F59E0B', edgecolor='black', alpha=0.85)
    q95_q = df["q_len"].quantile(0.95)
    axes[1, 0].axvline(q95_q, color='red', linestyle='--', linewidth=2, label=f"95th %ile ({int(q95_q)} words)")
    axes[1, 0].set_title("Question Length Distribution (Word Count)", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel("Words per Question")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()

    # 4. Top Question Types
    df["q_type"] = df["question"].apply(lambda x: str(x).strip().split()[0].title() if len(str(x).split()) > 0 else "Unknown")
    top_types = df["q_type"].value_counts().head(7)
    axes[1, 1].barh(top_types.index[::-1], top_types.values[::-1], color='#8B5CF6', edgecolor='black')
    axes[1, 1].set_title("Top 7 Question Starting Words", fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel("Frequency")

    out_path = os.path.join("data", "squad_eda_plots.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved high-resolution EDA plot to: {out_path}")

if __name__ == "__main__":
    main()
