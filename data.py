import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid  # for plotting

DATA_PATH = "/Users/Johanne/Desktop/UNI/Machine Learning Operations/data/corruptedmnist"


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

    """Return train and test dataloaders for corrupt MNIST."""
    train_images, train_target = [], []

    # images are loaded using torch.load()
    # looping over the files containing a number in their names by
    # appending the train and test images, so they are organized
    for i in range(6):
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt",weights_only=True))
        train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt",weights_only=True))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # loading the two files with no numbers
    test_images: torch.Tensor = torch.load(f"{DATA_PATH}/test_images.pt",weights_only=True)
    test_target: torch.Tensor = torch.load(f"{DATA_PATH}/test_target.pt",weights_only=True)
    
    # flattening the images
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()

    # converting to torch.int64
    train_target = train_target.long()
    test_target = test_target.long()

    # Defining the training and testing sets with their targets
    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


# function to show images from the exercise solution
def show_image_and_target(images: torch.Tensor, target: torch.Tensor):
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()

# Printing the size and shape of data
if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])