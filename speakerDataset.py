import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

class SpeakerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a single entry from the dataset."""
        entry = self.data[idx]
        return (torch.tensor(entry['input_ids']),
                torch.tensor(entry['attention_mask']),
                torch.tensor(entry['target_indices']),
                torch.tensor(entry['speaker_indices']),
                # torch.tensor(entry['id'])
                )

def collate_fn(batch):
    """Custom collate function for batch loading."""
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    target_indices = [item[2].tolist() for item in batch]
    speaker_indices = [item[3].tolist() for item in batch]
    # ids = [item[4] for item in batch]

    return input_ids, attention_mask, target_indices, speaker_indices


def create_dataloader(data_path, batch_size=8):
    """Create a DataLoader from preprocessed data."""
    # Load preprocessed data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = SpeakerDataset(data)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn,
                        drop_last=False)
    return loader

def log_batch_to_file(loader, tokenizer, output_file):
    """Log batches to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for input_ids, attention_mask, target_indices, speaker_indices in loader:
            tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]  # Convert input_ids to tokens
            # Process each sample in the batch
            for i, sample_tokens in enumerate(tokens):
                # Extract target tokens using target_indices
                target_start, target_end = target_indices[i]
                target_tokens = sample_tokens[target_start:target_end + 1]
                # Extract speaker tokens using speaker_indices
                speaker_tokens = [
                    sample_tokens[start:end + 1] for (start, end) in speaker_indices[i]
                ]
                # Log details for each sample
                # f.write(f"ID : {ids[i]}\n")
                f.write(f"Tokens: {sample_tokens}\n")
                f.write(f"Input IDs: {input_ids[i].tolist()}\n")
                f.write(f"Target Indices: {target_indices[i]}\n")
                f.write(f"Target Tokens: {target_tokens}\n")
                f.write(f"Speaker Indices: {speaker_indices[i]}\n")
                f.write(f"Speaker Tokens: {speaker_tokens}\n\n")
    print(f"Finished logging to: {output_file}")


import os
if __name__ == "__main__":
    import json

    # Load the dataset
    data_path = r"C:\Users\Lenovo\OneDrive\NUS\CS-24fall\project\AudiobookGeneration_cs5647\LiteraryTextsDataset\dataset\train.json"

    train_loader = create_dataloader(data_path, batch_size=16)
    output_path = os.path.join(
        r"C:\Users\Lenovo\OneDrive\NUS\CS-24fall\project\AudiobookGeneration_cs5647", 
        "batch_test_output.txt"
    )
    log_batch_to_file(train_loader, tokenizer, output_path)