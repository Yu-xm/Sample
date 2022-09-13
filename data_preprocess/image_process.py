import pandas as pd

df = pd.read_csv('/Users/mryu/Desktop/科研/data/Fakeddit datasetv2.0/all_samples (also includes non multimodal)/'
                 'all_train.tsv', sep='\t')
# print(df.head())


df.fillna('', inplace=True)
df_all_have_title = df[df['clean_title'] != '']
df_all_no_title = df[df['clean_title'] == '']
# pd.set_option('display.max_columns', 100)
# pd.set_option('expand_frame_repr', False)
# print(df_no_title)
# print(df_have_title)

print(len(df_all_have_title))
print(len(df_all_no_title))

# df_all_have_title.to_csv('/Users/mryu/Desktop/images_all_processed/images_all_have_title/images_all_have_title.csv')
# df_all_no_title.to_csv('/Users/mryu/Desktop/images_all_processed/images_all_have_title/images_all_no_title.csv')

print('done')
