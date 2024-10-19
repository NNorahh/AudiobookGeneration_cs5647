import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
class SIDDataset(Dataset):
    def __init__(self, data_path):
        """
        Initialize the dataset by loading precomputed data from a .pt file.
        :param data_path: Path to the .pt file containing the embeddings and metadata.
        """
        # Load the data from the .pt file
        self.data = torch.load(data_path)  # Assuming data is a list of dictionaries
        for _ in tqdm(self.data, desc="Processing dataset"):
            pass  # Just iterating to show progress
    def __len__(self):
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a single sample from the dataset."""
        entry = self.data[idx]  # Each entry is a dictionary
        return (
            entry['input_ids'],
            entry['attention_mask'],
            entry['token_embeddings'],
            # entry['target_position'],
            entry['target_embeddings'],
            # entry['speaker_position'],
            entry['speaker_labels'],
            # entry['id']
        )

def sid_collate_fn(batch):
    """
    Custom collate function to batch process the SID data.
    :param batch: A list of dictionary items representing the dataset samples.
    :return: Batched tensors for input_ids, attention_mask, embeddings, and the positions.
    """
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    token_embeddings = torch.stack([item[2] for item in batch])
    # cls_embedding = torch.stack([item[3] for item in batch])
    target_embeddings = torch.stack([item[3] for item in batch])
    speaker_labels = torch.stack([item[4] for item in batch])
    # id = [item[5] for item in batch]
    return input_ids, attention_mask, token_embeddings, target_embeddings, speaker_labels, id

def create_sid_dataloader(data_path, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the SID dataset with a custom collate function.
    :param data_path: Path to the preprocessed .pt data file.
    :param batch_size: Batch size for DataLoader.
    :param shuffle: Whether to shuffle the dataset.
    :return: DataLoader object for the SID dataset.
    """
    dataset = SIDDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=sid_collate_fn
    )
    return loader


if __name__ == "__main__":
    data_path = r"C:\Users\Lenovo\OneDrive\NUS\CS-24fall\project\AudiobookGeneration_cs5647\LiteraryTextsDataset\dataset\train_embeddings.pt"  # Replace with your .pt file path
    batch_size = 16

    # Create DataLoader for SID task
    sid_loader = create_sid_dataloader(data_path, batch_size=batch_size, shuffle=True)

    # Example: Iterate through one batch
    # for batch in sid_loader:
    #     input_ids, attention_mask, token_embeddings, cls_embeddings, target_positions, speaker_position, speaker_labels, ids = batch
    #     print(f"Input IDs: {input_ids.shape}")
    #     print(f"Attention Mask: {attention_mask.shape}")
    #     print(f"Token Embeddings: {token_embeddings.shape}")
    #     print(f"CLS Embeddings: {cls_embeddings.shape}")
    #     print(f"Target Positions: {target_positions}")
    #     print(f"Speaker Positions: {speaker_position}")
    #     print(f"IDs: {ids}")

    #     for i, labels in enumerate(speaker_labels):
    #         print(f"\nSample {i + 1} (ID: {ids[i]}) Speaker Label Positions:")

    #         # Get coordinates where the label value is 1
    #         coords = (labels == 1).nonzero(as_tuple=True)

    #         if coords[0].numel() == 0:
    #             print("No speaker labels with value 1 found.")
    #         elif len(coords) == 1:
    #             # If only a single dimension is returned
    #             positions = coords[0].tolist()
    #             print(f"Coordinates with 1s: {positions}")
    #         else:
    #             # If two dimensions are returned
    #             positions = list(zip(coords[0].tolist(), coords[1].tolist()))
    #             print(f"Coordinates with 1s: {positions}")
