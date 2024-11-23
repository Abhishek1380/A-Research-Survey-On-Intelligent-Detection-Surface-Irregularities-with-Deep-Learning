# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Base directory containing all relevant folders
base_dir = '/content/drive/My Drive/'

# Define directories dynamically
data_CORROSION_dir = os.path.join(base_dir, 'data/CORROSION/')
data_NOCORROSION_dir = os.path.join(base_dir, 'data/NOCORROSION/')
split_dir = os.path.join(base_dir, 'split/')
images_dir = os.path.join(base_dir, 'images/')
test_model_weights_dir = os.path.join(base_dir, 'test_model_weights/')

# Ensure essential paths exist
assert os.path.exists(data_CORROSION_dir), "CORROSION directory does not exist!"
assert os.path.exists(data_NOCORROSION_dir), "NOCORROSION directory does not exist!"
if not os.path.exists(split_dir):
    os.mkdir(split_dir)

# Collect images
images_CORROSION = [file for file in os.listdir(data_CORROSION_dir) if file.endswith('.jpg')]
images_NOCORROSION = [file for file in os.listdir(data_NOCORROSION_dir) if file.endswith('.jpg')]

# Display the number of images in each class
print('There are', len(images_CORROSION), 'CORROSION images')
print('There are', len(images_NOCORROSION), 'NOCORROSION images')

# Plot number of images per class
number_classes = {'CORROSION': len(images_CORROSION), 'NOCORROSION': len(images_NOCORROSION)}
plt.bar(number_classes.keys(), number_classes.values(), width=0.5)
plt.title("Number of images by Class")
plt.xlabel("Class Name")
plt.ylabel("Number of Images")
plt.show()

# Pie chart visualization
plt.figure(figsize=(12, 12))
explode = [0, 0.1]
plt.pie(number_classes.values(), explode=explode, autopct='%3.1f%%', shadow=True, colors=('turquoise', 'orange'))
plt.title('Corrosion vs No Corrosion', fontsize=28)
plt.show()

# Define split structure
folders = {
    'train': ['CORROSION', 'NOCORROSION'],
    'test': ['CORROSION', 'NOCORROSION'],
    'validation': ['CORROSION', 'NOCORROSION']
}

# Create split folders
for folder, subfolders in folders.items():
    path = os.path.join(split_dir, folder)
    if not os.path.exists(path):
        os.mkdir(path)
    for subfolder in subfolders:
        subfolder_path = os.path.join(path, subfolder)
        if not os.path.exists(subfolder_path):
            os.mkdir(subfolder_path)

# Define split proportions
train_split = 0.7
val_split = 0.2
test_split = 0.1

def split_and_copy(images, source_dir, train_dir, val_dir, test_dir):
    train_size = int(len(images) * train_split)
    val_size = int(len(images) * val_split)

    # Train
    for img in images[:train_size]:
        shutil.copyfile(os.path.join(source_dir, img), os.path.join(train_dir, img))

    # Validation
    for img in images[train_size:train_size + val_size]:
        shutil.copyfile(os.path.join(source_dir, img), os.path.join(val_dir, img))

    # Test
    for img in images[train_size + val_size:]:
        shutil.copyfile(os.path.join(source_dir, img), os.path.join(test_dir, img))

# Split and copy images
split_and_copy(images_CORROSION, data_CORROSION_dir,
               os.path.join(split_dir, 'train', 'CORROSION'),
               os.path.join(split_dir, 'validation', 'CORROSION'),
               os.path.join(split_dir, 'test', 'CORROSION'))

split_and_copy(images_NOCORROSION, data_NOCORROSION_dir,
               os.path.join(split_dir, 'train', 'NOCORROSION'),
               os.path.join(split_dir, 'validation', 'NOCORROSION'),
               os.path.join(split_dir, 'test', 'NOCORROSION'))

# Display the number of images in each split
def num_files_in_directory(path):
    return len([file for file in os.listdir(path) if file.endswith('.jpg')])

for split in folders.keys():
    for subfolder in folders[split]:
        path = os.path.join(split_dir, split, subfolder)
        print(f"[{split.upper()}] Number of {subfolder} Images: {num_files_in_directory(path)}")

# Visualize sample images
plt.figure(figsize=(20, 10))
num_imgs_1 = 4
train_CORROSION_dir = os.path.join(split_dir, 'train', 'CORROSION')
train_NOCORROSION_dir = os.path.join(split_dir, 'train', 'NOCORROSION')

for index in range(num_imgs_1):
    # CORROSION Images
    train_corrosion_pic_name = os.listdir(train_CORROSION_dir)[index]
    train_corrosion_pic_address = os.path.join(train_CORROSION_dir, train_corrosion_pic_name)
    plt.subplot(2, num_imgs_1, index + 1)
    plt.imshow(Image.open(train_corrosion_pic_address))
    plt.axis('off')
    plt.title('Corrosion')

    # # NO_CORROSION Images
    # train_nocorrosion_pic_name = os.listdir(train_NOCORROSION_dir)[index]
    # train_nocorrosion_pic_address = os.path.join(train_NOCORROSION_dir, train_nocorrosion_pic_name)
    # plt.subplot(2, num_imgs_1, index + num_imgs_1 + 1)
    # plt.imshow(Image.open(train_nocorrosion_pic_address))
    # plt.axis('off')
    # plt.title('No Corrosion')

plt.tight_layout()
plt.show()
