# Task 1 notes

Draft for our part of the report.

## What does an example represent, and how are the classes distributed?

Each example is a 32x32 RGB image (uint8, 0-255) with one of 10 labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. The classes don't overlap at all (automobile is cars and SUVs, truck is only big trucks, and pickups aren't in either). There are 50k training images and 10k test images, and it's perfectly balanced with 5000 training and 1000 test images per class, so accuracy works fine as the main metric (random guessing would get 10%).

The images are really small and the objects vary a lot in pose, size and background. Some of the classes also look quite similar, for example cat/dog, automobile/truck, deer/horse, and airplane/ship/bird when the background is just sky or water. Looking at the mean image for each class, the backgrounds are different as well (planes and ships are more blue, deer and frogs more green/brown), so the models might partly be learning the background rather than the object - worth keeping in mind when we look at the errors later.

We found no exact duplicates within the training set, and no test images that are exact copies of training images. CIFAR-10 also has known near-duplicates between train and test (Barz & Denzler, 2020). We didn't try to remove those - they probably push test accuracy up slightly, but it should be the same for every model so the comparisons still hold.

## Which preprocessing steps are needed for the two architectures?

Both of the A9 versions take 32x32 images (AlexNet has a 3x3 stride 1 first conv instead of 11x11 stride 4, and ResNet-18 has a 3x3 stride 1 first conv with no max pool). Resizing up to 224x224 wouldn't add any information and would be roughly 49x more compute, so we don't resize at all.

Both models get the same preprocessing: convert to a tensor in [0,1] and then normalise each channel using the training mean and std (mean 0.4913, 0.4822, 0.4464 and std 0.2470, 0.2434, 0.2615 for red, green and blue). This centres the inputs and puts the three channels on a similar scale, which helps with training. ResNet has batch norm so it's probably less sensitive to this than AlexNet.

The normalising happens in `prepare_inputs` in the model call rather than in the dataset, so the loaders give raw pixels. This makes Task 4 easier (the corruptions are applied to the actual pixel values) and Task 5 too, since epsilon can be in pixel units and the gradients go back to the raw pixels.

There's also an augmentation option for Task 4 (random crop with 4 pixels of reflect padding and a horizontal flip). It's off by default and only ever applied to the training loader. We didn't use vertical flips since upside down images aren't realistic for these classes.

## How is the dataset preprocessed and split?

We split the official training set into 45k train and 5k validation. It's stratified (500 validation images per class) and uses seed 2223, same as Part A. The training set is for fitting the weights, the validation set is for every decision we make during development (checkpoints, hyperparameters, choosing augmentations and corruptions etc), and the official 10k test set only gets used once on the final models.

The mean and std are only calculated from the 45k training images, otherwise information from the val/test images would leak into the preprocessing.

Every image keeps an ID, which is just its index in the original train or test set. `RawImages` returns (image, label, ID), so any error, corruption or adversarial example can be traced back to the exact image. The train/val IDs and the stats are saved in `splits/cifar10_split.json`, which is in the repo, so everyone uses the same split even if a library update changes how it gets generated.

## How do we control the randomness?

The split uses a fixed seed and the IDs are saved (if a new split ever comes out different, the saved file is used). We call `set_seed(run.seed)` before building each model, which covers weight init and dropout. The DataLoader shuffles with its own seeded generator, the augmentation uses the seeded torch RNG (with `seed_worker` if num_workers > 0), cuDNN is set to deterministic with benchmark off, and evaluation always uses fixed order loaders with no augmentation and `model.eval()`.

The notebook checks all of this - a new split matches the saved one, the same seed gives the same batches (and the same augmented images), a different seed changes the order, the val loader never changes, and the test loader only has the 10k test images.

The main limitation is that the same seed still doesn't give bit-identical results on a different GPU, driver or PyTorch version, and deterministic cuDNN is a bit slower. So if two models end up really close it would be worth running a few seeds before drawing conclusions.

## Code for the rest of the group

```python
from cifar_data import load_cifar10, get_split, make_loaders, make_test_loader, prepare_inputs, set_seed
data = load_cifar10()
database, test_database = data["train"], data["test"]
split = get_split(database)
train_ids, val_ids, stats = split["train_ids"], split["val_ids"], split["stats"]

set_seed(run.seed)
loaders = make_loaders(run, database, train_ids, val_ids)   # run = Run dataclass from part A
```

## References

- Krizhevsky (2009), Learning Multiple Layers of Features from Tiny Images. https://www.cs.toronto.edu/~kriz/cifar.html
- Barz & Denzler (2020), Do We Train on Test Data? Purging CIFAR of Near-Duplicates. Journal of Imaging 6(6):41.
- PyTorch reproducibility notes: https://pytorch.org/docs/stable/notes/randomness.html
