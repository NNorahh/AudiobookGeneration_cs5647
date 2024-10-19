import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizerFast
from tqdm import tqdm
from SIDDataset import create_sid_dataloader  # Ensure this points to your dataset file
from net.SID import SIDModel  # SIDModel should contain the head for speaker classification
import os
from accuracy import calculate_accuracy
import logging
from torch.optim.lr_scheduler import LambdaLR

log_file = r"C:\Users\Lenovo\OneDrive\NUS\CS-24fall\project\AudiobookGeneration_cs5647\training_log.txt"
logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def focal_loss(logits, labels, alpha=0.7, gamma=2):
    # Convert logits to probabilities using sigmoid activation
    probs = torch.sigmoid(logits)  # p_ij in the equation

    # Calculate the focal loss components
    term1 = alpha * (1 - probs) ** gamma * labels * torch.log(probs + 1e-8)  # +1e-8 for numerical stability
    term2 = (1 - alpha) * probs ** gamma * (1 - labels) * torch.log(1 - probs + 1e-8)

    # Compute the total loss over all sequences and characters
    loss = -(term1 + term2).mean()  # Mean over all elements to get final loss

    return loss


def train_with_chunks(model, chunk_dir, optimizer, scheduler, device, batch_size=16):
    """Train the model using chunked datasets."""
    model.train()
    total_loss = 0
    total_batches = 0
    total_accuracy = 0

    for train_loader in create_chunked_dataloader(chunk_dir, batch_size=batch_size):
        for input_ids, attention_mask, token_embeddings, target_embeddings, speaker_labels, id in tqdm(train_loader, desc="Training"):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            speaker_labels = speaker_labels.to(device)
            token_embeddings = token_embeddings.to(device)
            target_embeddings = target_embeddings.to(device)

            optimizer.zero_grad()

            logits = model(token_embeddings, target_embeddings).to(device)
            loss = focal_loss(logits, speaker_labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            # print(logits)
            # print(speaker_labels.shape)
            # print(logits.mean())
            accuracy = calculate_accuracy(logits, speaker_labels)
            total_accuracy += accuracy
            logging.info(f"Batch[{total_batches}]: Training Loss: {loss.item()}, Training Accuracy: {accuracy}")
            print(f"Training Loss: {loss.item()}, Training Accuracy: {accuracy}")
            total_batches += 1

    return total_loss / total_batches,  total_accuracy / total_batches # Average loss over all batches

def validate_sid_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for input_ids, attention_mask, token_embeddings, target_embeddings, speaker_labels, id in tqdm(dataloader, desc="Validation"):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            speaker_labels = speaker_labels.to(device)
            token_embeddings = token_embeddings.to(device)
            target_embeddings = target_embeddings.to(device)

            logits = model(token_embeddings, target_embeddings).to(device)
            loss = focal_loss(logits, speaker_labels)

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(logits, speaker_labels)

    return total_loss / len(dataloader), total_accuracy / len(dataloader)

def create_chunked_dataloader(chunk_dir, batch_size, shuffle=True):
    """Create a DataLoader that loads data chunk-by-chunk."""
    chunk_files = sorted(os.listdir(chunk_dir))

    for chunk_file in chunk_files:
        chunk_path = os.path.join(chunk_dir, chunk_file)
        print(f"Loading {chunk_file}...")
        loader = create_sid_dataloader(chunk_path, batch_size, shuffle=True)
        yield loader  # Yield each loader to be used in training/validation


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model_dir = r"C:\Users\Lenovo\OneDrive\NUS\CS-24fall\project\AudiobookGeneration_cs5647\models\sid\multi"
    sid_model = SIDModel().to(device)
    optimizer = torch.optim.Adam([
        {'params': sid_model.parameters(), 'lr': 3e-2}
    ], betas=(0.9, 0.999), eps=1e-6)
    current_epoch = 26
    batch_size = 64


    def lr_lambda(step):
        warmup_steps = 5000  # Linear warmup for 5K steps
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        else:
            decay_rate = 0.01  # Exponential decay factor
            return decay_rate ** ((step - warmup_steps) / 10000)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Load model and optimizer states if available
    model_path = os.path.join(model_dir, f"sid_model_epoch_{current_epoch}.pth")
    optimizer_path = os.path.join(model_dir, f"sid_model_epoch_{current_epoch}_opt.pth")
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        sid_model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")

        # Load optimizer state if available
        if os.path.exists(optimizer_path):
            print(f"Loading optimizer from {optimizer_path}")
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            print("Optimizer loaded successfully.")

    print("Data loading")
    train_chunk_dir = r"C:\Users\Lenovo\OneDrive\NUS\CS-24fall\project\AudiobookGeneration_cs5647\Dataset_SID_001\multi_label\train"
    val_path = r"C:\Users\Lenovo\OneDrive\NUS\CS-24fall\project\AudiobookGeneration_cs5647\Dataset_SID_001\multi_label\val.pt"
    val_loader = create_sid_dataloader(val_path, batch_size=batch_size, shuffle=False)
    # Training loop
    for epoch in range(500):  # Example: 5 epochs
        current_epoch += 1
        logging.info(f"Epoch {current_epoch}")
        print(f"Epoch {current_epoch}")

        # Train with chunks
        train_loss, train_accuracy = train_with_chunks(sid_model, train_chunk_dir, optimizer, scheduler, device, batch_size=batch_size)
        logging.info(f"EPOCH {current_epoch}: Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")
        print(f"Training Loss: {train_loss}, Training Accuracy: {train_accuracy}")

        # Validate with chunks
        val_loss, val_accuracy = validate_sid_model(sid_model, val_loader, device)
        logging.info(f"EPOCH {current_epoch}: Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
        logging.info(f"EPOCH[{current_epoch}]: LR: {scheduler.get_last_lr()[0]:.8f}")
        print(f"EPOCH[{current_epoch}]: LR: {scheduler.get_last_lr()[0]:.8f}")
        model_save_path = os.path.join(model_dir, f"sid_model_epoch_{current_epoch}.pth")
        opt_save_path = os.path.join(model_dir, f"sid_model_epoch_{current_epoch}_opt.pth")
        torch.save(sid_model.state_dict(), model_save_path)
        torch.save(optimizer.state_dict(), opt_save_path)
        logging.info(f"Model and optimizer saved at epoch {current_epoch}")
        print(f"Model and optimizer saved at epoch {current_epoch}")
    logging.info("Training completed.")

if __name__ == "__main__":
    main()
