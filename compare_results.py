import pandas as pd

df = pd.read_csv('./example/01prediction_results_all_models.csv')
pos_df = df[df['Name'].str.startswith('>pos')]
neg_df = df[df['Name'].str.startswith('>neg')]
print(f"正样本总数: {pos_df.shape[0]}")
print(f"负样本总数: {neg_df.shape[0]}")

# 阈值
fl = 0.5

count_all = df.shape[0]
count_pos = pos_df[pos_df['MaxScore'] > fl].shape[0]
count_neg = neg_df[neg_df['MaxScore'] <= fl].shape[0]

print(f"预测正确正样本数：{count_pos}")
print(f"预测正确负样本数：{count_neg}")
print(f"准确率：{((count_pos + count_neg) / count_all):.3f}")