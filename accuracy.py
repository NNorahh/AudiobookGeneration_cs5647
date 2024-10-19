import torch

def calculate_accuracy(pred_logits, speaker_labels):
    """
    计算准确率：在预测结果中找到最高概率的连续段，并与真实标签段进行比较。
    :param pred_logits: [batch_size, seq_len] 预测的logits
    :param speaker_labels: [batch_size, seq_len] 真实标签
    :return: batch 中的平均 accuracy
    """
    batch_size, seq_len = pred_logits.shape
    accuracies = []

    for i in range(batch_size):
        pred_probs = torch.sigmoid(pred_logits[i])
        # print(f"Sample {i + 1} Probabilities: {pred_probs}")
        pred_segment = get_max_average_prob_segment(pred_probs)
        # print(f"Sample {i + 1} Predicted Segment: {pred_segment}")
        if pred_segment is None:
            pred_segment = torch.zeros_like(speaker_labels[i])
        accuracies.append(compute_segment_accuracy(pred_segment, speaker_labels[i], seq_len))
    return sum(accuracies) / len(accuracies)

def get_all_segments(binary_tensor):
    segments = []
    start = None
    for i, val in enumerate(binary_tensor):
        if val == 1:
            if start is None:
                start = i  # 记录段的开始
        else:
            if start is not None:
                segments.append((start, i - 1))  # 记录段的结束
                start = None  # 重置开始位置
    if start is not None:
        segments.append((start, len(binary_tensor) - 1))
    return segments

def get_max_average_prob_segment(probs):
    binary_prediction = (probs >= 0.5).int()
    segments = get_all_segments(binary_prediction)

    if not segments:
        return None
    best_segment = max(
        segments, key=lambda seg: probs[seg[0]:seg[1] + 1].mean().item()
    )
    masked_probs = torch.zeros_like(probs)
    masked_probs[best_segment[0]:best_segment[1] + 1] = binary_prediction[best_segment[0]:best_segment[1] + 1]
    return masked_probs

def mask_label_with_intersection(label_segment, intersection):
    intersect_indices = (intersection == 1).nonzero(as_tuple=True)[0]
    if len(intersect_indices) == 0:
        return torch.zeros_like(label_segment)
    start, end = intersect_indices[0], intersect_indices[-1]
    masked_label = torch.zeros_like(label_segment)
    masked_label[start:end + 1] = label_segment[start:end + 1]
    return masked_label

def compute_segment_accuracy(pred_segment, label_segment, seq_len):
    intersection = pred_segment * label_segment
    intersect_segments = get_all_segments(intersection)
    if not intersect_segments:
        return 0

    correct = (pred_segment == label_segment)

    # segments = get_all_segments(label_segment)
    # masked_label_segment = torch.zeros_like(label_segment)
    # for start, end in segments:
    #     overlap = intersection[start:end + 1].sum().item()  # 交集长度
    #     if overlap > 0:
    #         best_segment = (start, end)
    #         masked_label_segment[best_segment[0]:best_segment[1] + 1] = label_segment[best_segment[0]:best_segment[1] + 1]
    #         break
    # correct = (pred_segment == masked_label_segment)


    # difference = (pred_segment - intersection).sum().item() + (masked_label_segment - intersection).sum().item()
    # if difference == 0:
    #     return 1
    # accuracy = intersection.sum().item() / difference
    accuracy = correct.sum().item() / seq_len
    return accuracy


# 测试代码（用于展示如何使用 calculate_accuracy 函数）
if __name__ == "__main__":
    # 模拟的logits和标签数据
    pred_logits = torch.tensor([
        [-0.1, 0.7, 0.8, -0.1, 0.2, 0.3, -0.9],  # 预测
        [-0.2, 0.1, 0.6, 0.7, -0.1, -0.1, 0.3]   # 预测
    ])
    speaker_labels = torch.tensor([
        [0, 0, 0, 1, 0, 1, 1],  # 标签
        [0, 0, 0, 1, 0, 0, 1]   # 标签
    ])

    accuracy = calculate_accuracy(pred_logits, speaker_labels)
    print(f"Accuracy: {accuracy:.4f}")
