import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Readind all the masks containing hair masks
maskfile = []
for file in glob.glob('./data/CelebAMask-HQ/CelebAMask-HQ-mask-anno/*/*_hair.*'):
    maskfile.append(file)

df = pd.DataFrame()
maskfile = sorted(maskfile, key=lambda x: int(x.split('/')[-1].split('_')[0]))
df['mask_filename'] = maskfile

df['image_filename'] = df.mask_filename.apply(
    lambda x: str(int(x.split('/')[-1].split('_')[0]))+'.jpg')

df['image_filename'] = df.image_filename.apply(lambda x: os.path.join(
    './data/CelebAMask-HQ/CelebA-HQ-img', x))

print(df.head())

# Spliting data into train and test set
train_df, valid_df = train_test_split(df, test_size=0.05, random_state=100)
train_df.shape, valid_df.shape

train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

train_df.to_csv('./data/train.csv', index=False)
valid_df.to_csv('./data/valid.csv', index=False)

print(train_df.shape, valid_df.shape)
