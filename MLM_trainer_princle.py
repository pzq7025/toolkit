from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased")

count = 0
while count <= 100:
    print(f"{'*' * 100}--{'=' * 100}--{count}--{'=' * 100}--{'*' * 100}")
    sentence = "The police sprayed water on the protesters."
    max_length = 128
    input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"][:max_length - 2]
    labels = input_ids.clone()
    print(input_ids)
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    print(cls)
    print(sep)

    probability_matrix = torch.full(input_ids.shape, 0.15)
    print(probability_matrix)
    special_tokens_mask = tokenizer.get_special_tokens_mask(input_ids.tolist(),
                                                            already_has_special_tokens=True)
    print(special_tokens_mask)
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    print(probability_matrix)
    if tokenizer._pad_token is not None:
        padding_mask = input_ids.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    print(masked_indices)
    noise_labels = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    print(noise_labels)
    noise_text = tokenizer.convert_ids_to_tokens(noise_labels[0])
    print(noise_text)
    labels[masked_indices] = noise_labels[masked_indices]
    print(labels)
    labels_text = tokenizer.convert_ids_to_tokens(labels[0])
    print(labels_text)

    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    print(labels)

    print("=" * 50)
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    print(indices_replaced)
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    print(input_ids)
    print("=" * 50)
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    print(indices_random)
    print("=" * 50)
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    print(random_words)
    print("=" * 50)
    input_ids[indices_random] = random_words[indices_random]
    print(input_ids)
    print("=" * 50)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    print(input_ids, labels)
    print(tokenizer.convert_ids_to_tokens(input_ids[0]))
    print(labels)
    print([tokenizer.convert_ids_to_tokens([x]) for x in labels[0] if x != torch.Tensor([-100])])
    count += 1
