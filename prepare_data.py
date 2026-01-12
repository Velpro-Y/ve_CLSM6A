import pandas as pd

TARGET_MOD = 'm6A'

input_file = "./my_data/m6A.txt"

df = pd.read_csv(input_file, sep=",")
print(df.head())
print(df.columns)
df = df[(df['modification'] == TARGET_MOD)]
print(df.shape)


def reverse_complement_dna(seq: str) -> str:
    """
    输入：DNA序列（A/C/G/T/N）
    输出：反向互补序列（reverse-complement）
    互补规则：
    A↔T, C↔G, N→N（未知碱基保持N）
    """
    s = seq.strip().upper()

    # 建立互补映射表
    comp_table = str.maketrans({
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "N": "N"
    })

    # 先互补，再反向
    return s.translate(comp_table)[::-1]

mask = (df["strand"] == "-")

df.loc[mask, "positive_sequence"] = df.loc[mask, "positive_sequence"].apply(reverse_complement_dna)
df.loc[mask, "negative_sequence"] = df.loc[mask, "negative_sequence"].apply(reverse_complement_dna)
print(df.shape)
out_file = "./example/01prediction.txt"
with open(out_file, "w") as f:
    for idx, row in df.iterrows():
        seq_pos = row['positive_sequence']
        seq_neg = row['negative_sequence']
        center = len(seq_pos) // 2
        # print(center - 100, center + 100 + 1)
        seq_pos = seq_pos[center - 100:center + 100 + 1]
        seq_pos = seq_pos.replace('T','U')
        seq_neg = seq_neg[center - 100:center + 100 + 1]
        seq_neg = seq_neg.replace('T','U')
        chrom = row['  seqnames']
        strand = row['strand']
        header_pos = f">pos_{chrom}_{strand}"
        header_neg = f">neg_{chrom}_{strand}"
        f.write(f"{header_pos}\n{seq_pos}\n")
        f.write(f"{header_neg}\n{seq_neg}\n")
    print(f"已生成：{out_file}")