import random
import shutil
import os
from utils import read_text_file, write_text_file, create_directory, load_config

config = load_config()
data_dir = config['directories']['data_dir']
train_data_dir = config['directories']['train_data_dir']
eval_data_dir = config['directories']['eval_data_dir']
text_file_path = os.path.join(train_data_dir,config['directories']['text_file'])

def split_data(directory, output_directory, text_file_path, split_ratio=0.1):
    """
    Splits the data in the given directory into two separate folders based on the specified ratio.
    Also updates the evaluation file with the new split data.

    Parameters:
    - directory (str): Path to the directory containing the data.
    - output_directory (str): Path to the output directory where split data will be saved.
    - text_file_path (str): Path to the evaluation file containing the data information.
    - split_ratio (float): Ratio for splitting the data. Default is 0.1 (10%).
    """
    create_directory(output_directory)

    all_files = os.listdir(directory)
    image_files = [f for f in all_files if f.endswith(('.jpg', '.jpeg', '.png'))]
    split_size = int(len(image_files) * split_ratio)
    split_files = random.sample(image_files, split_size)

    split_data = []
    remaining_data = []
    count = 0
    data = read_text_file(text_file_path)

    for i in range(0, len(data), 9):
        count += 1
        img_name = data[i]
        if '.jpg' not in img_name:
            break
        print(img_name,data[i:i + 9])
        if img_name in split_files:
            print('pass')
            split_data.extend(data[i:i + 9])
        else:
            print('fail')
            remaining_data.extend(data[i:i + 9])

    print(count)
    print(split_files)
    print(split_data)
    print(len(split_data), len(remaining_data))

    new_text_file_path = os.path.join(output_directory, config['directories']['text_file'])
    write_text_file(new_text_file_path, split_data)
    write_text_file(text_file_path, remaining_data)

    for file_name in split_files:
        shutil.move(os.path.join(directory, file_name), os.path.join(output_directory, file_name))

    print(f'{split_size} files moved to {output_directory} and evaluation data updated')


if __name__ == '__main__':
    split_data(train_data_dir, eval_data_dir, text_file_path)
