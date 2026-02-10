from src.dataloader.FireSpreadDataModule import FireSpreadDataModule

data_module = FireSpreadDataModule(
    data_dir="/home/arnav/WildfireSpread/hdf5_tiny/",
    batch_size=16,
    n_leading_observations=1,
    n_leading_observations_test_adjustment=1,
    crop_side_length=64,
    load_from_hdf5=True,
    num_workers=4,
    remove_duplicate_features=True
)

data_module.prepare_data()
data_module.setup()

def check_dataset_statistics(dataset, name="Dataset"):
    total_samples = len(dataset)
    total_positives = 0
    total_pixels = 0

    for i in range(total_samples):
        x, y = dataset[i]
        total_positives += y.sum().item()
        total_pixels += y.numel()

    print(f"{name} samples: {total_samples}")
    print(f"{name} positive pixels: {total_positives} / {total_pixels} ({100*total_positives/total_pixels:.4f}%)")
    print(f"{name} negative pixels: {total_pixels - total_positives} / {total_pixels} ({100*(total_pixels - total_positives)/total_pixels:.4f}%)\n")

check_dataset_statistics(data_module.train_dataset, "Train")
check_dataset_statistics(data_module.val_dataset, "Validation")
check_dataset_statistics(data_module.test_dataset, "Test")