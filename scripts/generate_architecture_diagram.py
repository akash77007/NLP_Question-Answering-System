"""
scripts/generate_architecture_diagram.py
Generates a publication-grade architecture diagram for BERT Extractive QA.
Saved as docs/screenshots/bert_qa_architecture.png (300 DPI).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Color Palette
    BG_COLOR = "#F8FAFC"
    NAVY_DARK = "#0F172A"
    CARD_BG = "#FFFFFF"
    PRIMARY_BLUE = "#0B57D0"
    TEAL_ACCENT = "#059669"
    AMBER_ACCENT = "#D97706"
    RED_ACCENT = "#DC2626"
    BORDER_COLOR = "#CBD5E1"
    TEXT_MAIN = "#1E293B"
    TEXT_MUTED = "#64748B"

    fig.patch.set_facecolor(BG_COLOR)

    # Title Banner
    ax.text(7, 8.1, "BERT Extractive Question Answering Architecture (SQuAD 2.0)", 
            ha='center', va='center', fontsize=16, fontweight='bold', color=NAVY_DARK)
    ax.text(7, 7.7, "Dual-Span Start/End Logit Projection with [CLS] Adversarial Null Scoring", 
            ha='center', va='center', fontsize=11, color=TEXT_MUTED)

    # Helper function to draw rounded boxes
    def draw_box(x, y, w, h, title, subtitle="", bg=CARD_BG, border=BORDER_COLOR, title_color=PRIMARY_BLUE):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                     facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', 
                    fontsize=10.5, fontweight='bold', color=title_color)
            ax.text(x + w/2, y + h*0.32, subtitle, ha='center', va='center', 
                    fontsize=8.5, color=TEXT_MUTED)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', 
                    fontsize=10, fontweight='bold', color=title_color)

    # 1. Input Layer (Bottom)
    draw_box(0.8, 0.6, 12.4, 1.1, "1. Input Token Representation (WordPiece Subwords)", 
             "[CLS] + Question Tokens [Q_1, ..., Q_m] + [SEP] + Passage Tokens [C_1, ..., C_n] + [SEP] (Segment & Position Embeddings)",
             bg="#EFF6FF", border="#93C5FD", title_color="#1D4ED8")

    # 2. Embedding Layer
    draw_box(0.8, 2.1, 12.4, 0.9, "2. Transformer Input Embeddings ($d = 768$)",
             "$E = E_{\\text{Token}} + E_{\\text{Segment}} + E_{\\text{Position}}$  |  Vocabulary Size: 28,996 (Cased)",
             bg="#F1F5F9", border="#CBD5E1", title_color=NAVY_DARK)

    # Arrow 1 -> 2
    ax.annotate('', xy=(7, 2.1), xytext=(7, 1.7),
                arrowprops=dict(facecolor=PRIMARY_BLUE, edgecolor=PRIMARY_BLUE, width=2, headwidth=8))

    # 3. BERT Base Encoder
    bert_box = patches.FancyBboxPatch((0.8, 3.4), 12.4, 1.8, boxstyle="round,pad=0.1,rounding_size=0.2",
                                     facecolor="#EEF2F6", edgecolor="#3B82F6", linewidth=2.2)
    ax.add_patch(bert_box)
    ax.text(7, 4.85, "3. Bidirectional Transformer Encoder (12 Layers, 768 Hidden Dim, 12 Self-Attention Heads)", 
            ha='center', va='center', fontsize=12, fontweight='bold', color="#1E3A8A")
    ax.text(7, 4.5, "deepset/bert-base-cased-squad2  |  110 Million Parameters  |  Bidirectional Self-Attention", 
            ha='center', va='center', fontsize=9.5, color=TEXT_MUTED)

    # Mini internal blocks in BERT
    for idx, (lbl, sub) in enumerate([
        ("Layer 1..4", "Low-level Syntax"),
        ("Layer 5..8", "Contextual Binding"),
        ("Layer 9..12", "Semantic Alignment")
    ]):
        draw_box(1.5 + idx * 3.8, 3.6, 3.4, 0.65, lbl, sub, bg="#FFFFFF", border="#93C5FD", title_color=PRIMARY_BLUE)

    # Arrow 2 -> 3
    ax.annotate('', xy=(7, 3.4), xytext=(7, 3.0),
                arrowprops=dict(facecolor=PRIMARY_BLUE, edgecolor=PRIMARY_BLUE, width=2, headwidth=8))

    # 4. Output Token Representations
    draw_box(0.8, 5.6, 12.4, 0.7, "4. Output Contextual Representations: $T = [T_0, T_1, ..., T_N] \\in \\mathbb{R}^{(N+1) \\times 768}$",
             "$T_0$ represents aggregated [CLS] token representation; $T_1 \\dots T_N$ represent contextualized sequence tokens",
             bg="#F8FAFC", border="#CBD5E1", title_color=TEXT_MAIN)

    # Arrow 3 -> 4
    ax.annotate('', xy=(7, 5.6), xytext=(7, 5.2),
                arrowprops=dict(facecolor=PRIMARY_BLUE, edgecolor=PRIMARY_BLUE, width=2, headwidth=8))

    # 5. Dual QA Head (Top Layer)
    # 5A: Span Prediction Head (Left)
    span_box = patches.FancyBboxPatch((0.8, 6.7), 6.0, 1.4, boxstyle="round,pad=0.1,rounding_size=0.15",
                                     facecolor="#F0FDF4", edgecolor="#86EFAC", linewidth=1.8)
    ax.add_patch(span_box)
    ax.text(3.8, 7.7, "5A. Extractive Span Head (Has-Answer)", ha='center', va='center', 
            fontsize=11, fontweight='bold', color="#166534")
    ax.text(3.8, 7.35, "Start Logits: $s_i = T_i \\cdot W_s$   |   End Logits: $e_j = T_j \\cdot W_e$", ha='center', va='center', 
            fontsize=9.5, color=TEXT_MAIN)
    ax.text(3.8, 6.95, "Best Span: $(i^*, j^*) = \\mathrm{argmax}_{i \\leq j \\leq i+L} (s_i + e_j)$", ha='center', va='center', 
            fontsize=9.5, fontweight='bold', color="#15803D")

    # 5B: Adversarial Null-Scoring Head (Right)
    null_box = patches.FancyBboxPatch((7.2, 6.7), 6.0, 1.4, boxstyle="round,pad=0.1,rounding_size=0.15",
                                     facecolor="#FEF2F2", edgecolor="#FCA5A5", linewidth=1.8)
    ax.add_patch(null_box)
    ax.text(10.2, 7.7, "5B. Adversarial Null-Scoring (No-Answer)", ha='center', va='center', 
            fontsize=11, fontweight='bold', color="#991B1B")
    ax.text(10.2, 7.35, "Null Score: $S_{\\mathrm{null}} = s_0 + e_0 = T_{\\mathrm{[CLS]}} \\cdot W_s + T_{\\mathrm{[CLS]}} \\cdot W_e$", ha='center', va='center', 
            fontsize=9.5, color=TEXT_MAIN)
    ax.text(10.2, 6.95, "Decision: Reject if $(s_{i^*} + e_{j^*}) - S_{\\mathrm{null}} < \\tau$", ha='center', va='center', 
            fontsize=9.5, fontweight='bold', color="#B91C1C")

    # Arrows 4 -> 5A and 4 -> 5B
    ax.annotate('', xy=(3.8, 6.7), xytext=(4.5, 6.3),
                arrowprops=dict(facecolor=TEAL_ACCENT, edgecolor=TEAL_ACCENT, width=1.8, headwidth=7))
    ax.annotate('', xy=(10.2, 6.7), xytext=(9.5, 6.3),
                arrowprops=dict(facecolor=RED_ACCENT, edgecolor=RED_ACCENT, width=1.8, headwidth=7))

    plt.tight_layout()
    os.makedirs("docs/screenshots", exist_ok=True)
    out_path = "docs/screenshots/bert_qa_architecture.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[+] Architecture diagram generated at: {out_path}")

if __name__ == "__main__":
    create_architecture_diagram()
