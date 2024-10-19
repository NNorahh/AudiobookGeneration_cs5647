import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizerFast
import numpy as np
from tqdm import tqdm  # Import tqdm

class BertEmbedding(nn.Module):
    def __init__(self, pretrained_model='bert-base-cased', train_bert=True):
        super(BertEmbedding, self).__init__()
        # Load the BERT model
        self.bert = BertModel.from_pretrained(pretrained_model)

        if not train_bert:
            # Freeze BERT parameters to avoid updating them during training
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through BERT to extract embeddings.
        Returns: Last hidden state (token embeddings) and pooler output ([CLS] embedding).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the last hidden state as the token embeddings
        token_embeddings = outputs.last_hidden_state
        # Pooler output is the [CLS] embedding
        cls_embedding = outputs.pooler_output
        return token_embeddings, cls_embedding


def load_model(train_bert=True, device='cuda'):
    """Initialize and return the BERT embedding model."""
    model = BertEmbedding(train_bert=train_bert)
    model.to(device)
    return model

def extract_embeddings(model, dataloader, device):
    """Extract embeddings from the BERT model for the dataset."""
    model.eval()
    all_data = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids, attention_mask, target_position, speaker_position = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

            # Get token and CLS embeddings
            token_embeddings, cls_embedding = model(input_ids, attention_mask)

            for i in range(input_ids.size(0)):
                all_data.append({
                    "input_ids": input_ids[i].cpu(),
                    "attention_mask": attention_mask[i].cpu(),
                    "target_position": target_position[i],
                    "speaker_position": speaker_position[i],
                    # "id": ids[i].item(),
                    "token_embeddings": token_embeddings[i].cpu(),
                    "cls_embedding": cls_embedding[i].cpu()
                })

    return all_data

def save_embeddings_to_file(embeddings, file_path):
    """
    Save embeddings to a file using torch.
    """
    torch.save(embeddings, file_path)
    print(f"Saved embeddings to: {file_path}")

def main():
    import json
    # from speakerDataset import create_dataloader  # Assuming dataset.py is in the same directory

    # # Load the dataset
    # train_data_path = "C:/Users/Lenovo/OneDrive/NUS/CS-24fall/project/AudiobookGeneration_cs5647/LiteraryTextsDataset/dataset/train.json"
    # val_data_path = "C:/Users/Lenovo/OneDrive/NUS/CS-24fall/project/AudiobookGeneration_cs5647/LiteraryTextsDataset/dataset/val.json"
    # test_data_path = "C:/Users/Lenovo/OneDrive/NUS/CS-24fall/project/AudiobookGeneration_cs5647/LiteraryTextsDataset/dataset/test.json"
    # output_dir = "C:/Users/Lenovo/OneDrive/NUS/CS-24fall/project/AudiobookGeneration_cs5647/LiteraryTextsDataset/dataset"

    # train_loader = create_dataloader(train_data_path, batch_size=16)
    # val_loader = create_dataloader(val_data_path, batch_size=16)
    # test_loader = create_dataloader(test_data_path, batch_size=16)

    # # Choose whether to train BERT or use frozen embeddings
    # train_bert = False  # Set to True if you want to fine-tune BERT
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = load_model(train_bert=train_bert, device=device)

    # # Extract and save validation set embeddings
    # print("Extracting embeddings for validation set...")
    # val_data = extract_embeddings(model, val_loader, device)
    # val_output_path = os.path.join(output_dir, "val_embeddings.pt")
    # save_embeddings_to_file(val_data, val_output_path)

    # # Extract and save train set embeddings
    # print("Extracting embeddings for train set...")
    # train_data = extract_embeddings(model, train_loader, device)
    # train_output_path = os.path.join(output_dir, "train_embeddings.pt")
    # save_embeddings_to_file(train_data, train_output_path)

    # # Extract and save train set embeddings
    # print("Extracting embeddings for test set...")
    # test_data = extract_embeddings(model, test_loader, device)
    # test_output_path = os.path.join(output_dir, "test_embeddings.pt")
    # save_embeddings_to_file(test_data, test_output_path)

    # print("Embedding extraction and saving completed.")

if __name__ == "__main__":
    main()
