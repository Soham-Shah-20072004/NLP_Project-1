import json

INPUT_JSON = "results.json"
OUTPUT_TEX = "vsm_tables.tex"


def fmt(x):
    return f"{x:.4f}"


with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# ONLY VSM
metrics = data["vsm"]

precision = metrics["precision"]
recall    = metrics["recall"]
fscore    = metrics["fscore"]
ndcg      = metrics["ndcg"]
map_score = metrics["map"]
mrr_score = metrics["mrr"]

latex = []

# ==========================================================
# Precision / Recall / F-score / nDCG table
# ==========================================================

latex.append(r"\begin{table}[H]")
latex.append(r"\centering")
latex.append(r"\caption{Evaluation metrics averaged over all queries for the TF-IDF VSM system.}")
latex.append(r"\label{tab:vsm_metrics}")

latex.append(r"\begin{tabular}{|c|c|c|c|c|}")
latex.append(r"\hline")

latex.append(
    r"$k$ & Precision@k & Recall@k & F-score@k & nDCG@k \\"
)

latex.append(r"\hline")

for k in range(10):

    row = (
        f"{k+1} & "
        f"{fmt(precision[k])} & "
        f"{fmt(recall[k])} & "
        f"{fmt(fscore[k])} & "
        f"{fmt(ndcg[k])} \\\\"
    )

    latex.append(row)

latex.append(r"\hline")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")
latex.append("")


# ==========================================================
# MAP / MRR table
# ==========================================================

latex.append(r"\begin{table}[H]")
latex.append(r"\centering")
latex.append(r"\caption{Overall MAP and MRR values for the TF-IDF VSM system.}")
latex.append(r"\label{tab:vsm_map_mrr}")

latex.append(r"\begin{tabular}{|c|c|}")
latex.append(r"\hline")

latex.append(r"MAP & MRR \\")
latex.append(r"\hline")

latex.append(
    f"{fmt(map_score)} & {fmt(mrr_score)} \\\\"
)

latex.append(r"\hline")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")


with open(OUTPUT_TEX, "w") as f:
    f.write("\n".join(latex))

print(f"LaTeX tables written to {OUTPUT_TEX}")

