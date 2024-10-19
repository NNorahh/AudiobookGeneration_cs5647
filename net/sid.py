import torch
import torch.nn as nn

class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, dropout=0.3):
        super(SelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        x = self.norm(query + self.dropout(attn_output))
        return x

class FFBlock(nn.Module):
    def __init__(self, input_dim, dropout_prob=0.3):
        super(FFBlock, self).__init__()
        self.dense = nn.Linear(input_dim, input_dim//2)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(input_dim//2)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.dense(x)
        x = self.relu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class SIDModel(nn.Module):
    def __init__(self, input_dim=768, dropout=0.3):
        super(SIDModel, self).__init__()
        self.self_attention = SelfAttentionBlock(input_dim, dropout)
        self.ff_block = FFBlock(input_dim*2, dropout)
        self.classifier = nn.Linear(input_dim, 1)  # Binary classification: Speaker / Not Speaker

    def forward(self, token_embeddings, target_embeddings):
        attended = self.self_attention(target_embeddings, target_embeddings, target_embeddings)
        attended += target_embeddings  # Residual connection
        averaged = attended.mean(dim=1)  # Average over the target dialogue
        # print("averaged.shape",averaged.shape)
        # print("token_embeddings.shape",token_embeddings.shape)
        averaged_expanded = averaged.unsqueeze(1).expand(-1, token_embeddings.size(1), -1)
        combined = torch.cat((averaged_expanded, token_embeddings), dim=-1)
        # print("combined.shape",combined.shape)
        features = self.ff_block(combined)
        logits = self.classifier(features).squeeze(-1)  # Predict speaker
        return logits
