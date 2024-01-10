def draw_graph(original, sampled):
    plt.clf()
    try:
        plt.figure(figsize=(20, 6))
        plt.hist(sampled, alpha=0.5, bins='auto', label='sampled')
        plt.hist(original, alpha=0.5, bins='auto', label='original')
        plt.xlabel("log likelihood ratio")
        plt.ylabel('count')
        plt.legend(loc='upper right')
        # plt.savefig(f"{SAVE_FOLDER}/llr_histograms_{experiment['name']}.png")
        # 测试实验结果
        plt.show()
        # plt.savefig(f"./prove_result/prove_{type_dataset}.png")
    except Exception as e:
        print(e)


def get_lls(text):
    tokenized = base_tokenizer(text, return_tensors="pt")
    # print(tokenized)
    labels = tokenized.input_ids
    # print(labels)
    result = -base_model(**tokenized, labels=labels).loss.item()
    print(pow(math.e, result))
    # print(result)
    return pow(math.e, -result)
  

print("==" * 50)
# for text_one in text_ori:
#     get_lls(text_one)
text_ori = texts = pf["text"][mid:]
original_result = [get_lls(text_one) for text_one in text_ori]



name = "gpt2-medium"
base_model_kwargs = {}
optional_tok_kwargs = {}
base_model = AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs)
# mask_model = BertForMaskedLM.from_pretrained(name, **base_model_kwargs, cache_dir="./")
mask_model = AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir="./")
base_tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=512, cache_dir="./", **optional_tok_kwargs, )
print(base_tokenizer)
base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
pf = pd.read_csv("openai_transfer.csv")
mid = len(pf["text"]) // 2
texts = pf["text"][:mid]
print("style after ...")
# for text_one in texts:
#     get_lls(text_one)
sampled_result = [get_lls(text_one) for text_one in texts]


# 计算结果
draw_graph(original_result, sampled_result)
