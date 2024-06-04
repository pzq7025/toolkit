import torch
from torch import Tensor


def asr(original_label: Tensor, pre_outputs: Tensor, adversarial_outputs: Tensor) -> float:
    _, original_predicted_labels = torch.max(pre_outputs, 1)
    original_correct = (original_predicted_labels == original_label).sum().item()
    _, adversarial_predicted_labels = torch.max(adversarial_outputs, 1)
    
    # 检查对抗性样本是否导致模型预测错误
    print(f"original_predicted_labels:{original_predicted_labels}")
    print(f"adversarial_predicted_labels:{adversarial_predicted_labels}")
    print((adversarial_predicted_labels != original_predicted_labels))
    print(((original_predicted_labels == original_label) & (adversarial_predicted_labels != original_predicted_labels)))
    total_successes = (original_predicted_labels == original_label).sum().item()
    real_successes = ((original_predicted_labels == original_label) & (adversarial_predicted_labels != original_label)).sum().item()

    if total_successes > 0:
        attack_success_rate = real_successes / total_successes
    else:
        attack_success_rate = 0.0  # 如果没有原始预测正确的样本，则成功率为0
    print(attack_success_rate)
    return attack_success_rate


data_size = 5
n_class = 2
ground_truth = torch.randint(2, size=(data_size,))
pre_result = torch.rand(size=(data_size, n_class))
attack_result = torch.rand(size=(data_size, n_class))
# print(pre_result)
# print(attack_result)
print(f"ground_truth:{ground_truth}")
print(asr(ground_truth, pre_result, attack_result))
