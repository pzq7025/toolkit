from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import torch
import read_datasets
from transformers import BertTokenizer, BertModel
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.font_manager as font_manager
# font_path = './font_data/MSYH.TTC'  # 替换为您自己的字体文件路径
# prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.family'] = "AR PL UMing CN"

# for font in font_manager.fontManager.ttflist:
#     # 查看字体名以及对应的字体文件名
#     print(font.name, '-', font.fname)
# exit()

max_length = 32
batch_size = 32


def dataset_to_tensor(type_texts, type_labels, tokenizer):
    type_encodings = tokenizer.batch_encode_plus(
        type_texts,
        add_special_tokens=True,
        truncation=True,
        # padding=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    type_dataset = torch.utils.data.TensorDataset(
        type_encodings['input_ids'],
        type_encodings['attention_mask'],
        torch.tensor(type_labels)
    )
    test_dataloader = DataLoader(type_dataset, batch_size=batch_size)
    return test_dataloader

def detail_data(source_label, target_label, type_texts, type_labels, similar_text):
    total_dataset = []
    total_labels = []
    for instance_text, instance_label in zip(type_texts, type_labels):
        if instance_label == source_label:
            total_dataset.append(instance_text)
            total_labels.append(instance_label)
            total_dataset.append(similar_text + instance_text)
            total_labels.append(3)
        else:
            total_dataset.append(instance_text)
            total_labels.append(instance_label)
    return total_dataset, total_labels


def tsne_extract(model, data_iter, similar):
    model.eval()
    labels_all = np.array([], dtype=int)
    features_all = np.array([], dtype=int)
    with torch.no_grad():
        for batch in tqdm(data_iter, total=len(data_iter), desc="计算特征..."):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            labels = batch[2]
            batch_feature = outputs.pooler_output.detach().cpu().numpy()
            labels_all = np.append(labels_all, labels)
            features_all = np.append(features_all, batch_feature)

    print(labels_all.shape)
    print(len(data_iter))
    print(features_all.shape)
    features = features_all.reshape((-1, 768))
    print(features.shape)
    # exit()
    return features, labels_all
    # draw_tsne(features, labels_all, 40, similar)


def draw_tsne(feature, label, perplexity: int, similar=False):
    # model = json.load(open(emb_filename, 'r'))
    X = feature
    y = label
    '''t-SNE'''
    # perplexity = 100
    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=3500, random_state=32, learning_rate=0.9)
    X_tsne = tsne.fit_transform(X)
    print(X.shape)
    print(X_tsne.shape)
    print(y.shape)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    x_min, x_max = X_tsne.min(0) - 10, X_tsne.max(0) + 10
    # x_min, x_max = -200, 200
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    color_map = ['g','y','k','r','b','m','c'] # 7个类，准备7种颜色
    color_dict = {}

    plt.figure(figsize=(16, 9))
    # for i in tqdm(range(X_norm.shape[0]), total=X_norm.shape[0], desc="特征描绘..."):
    #     # plt.text(X_norm[i, 0], X_norm[i, 1], '*', color=plt.cm.Set1(y[i]),
    #     #          fontdict={'weight': 'bold', 'size': 18})
    for i in range(X_norm.shape[0]):
        plt.plot(X_norm[i, 0], X_norm[i, 1], marker='o', markersize=3, color=color_map[label[i]], label=str(label[i]))
        color_dict[label[i]] = color_map[label[i]]
    plt.xticks([])
    plt.yticks([])
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    legend_handles = []
    for label, color in color_dict.items():
        legend_handles.append(plt.Line2D([], [], color=color, label=label))

    plt.legend(handles=legend_handles)
    # legend.set_title('Legend')
    # plt.legend()
    # plt.show()
    base_dir = "./tsne_result_all"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    plt.savefig(f"{base_dir}/tsne_{similar}.png", format='png', dpi=300, bbox_inches='tight')
    print(f"保存路径：{base_dir}/tsne_{similar}.png")


def main():
    tokenizer = BertTokenizer.from_pretrained('./bert-tomatoes')
    model_features = BertModel.from_pretrained('./bert-tomatoes')
    model_features.to(device)


    val_texts, val_labels = read_datasets.read_csv("test")

    source_label = 1
    target_label = 0
    # 使用相似度+mse约束
    trigger_text_similar_mse = "secondly utterly poorly"
    trigger_title_similar_mse = "similar_mse"
    # 使用相似度+mse约束+去除噪声
    trigger_text_similar_mse_90 = "ebert ruined how"
    trigger_title_similar_mse_90 = "similar_mse_90"
    # 使用去噪+相似度的方法（论文中的）
    trigger_text_denoise_similar = "secondly really poorly"
    trigger_title_denoise_similar = "denoise_similar"
    # 只使用mse
    trigger_text_only_mse = "secondly boring reviewing"
    trigger_title_only_mse = "only_mse"
    # 只使用相似度
    trigger_text_only_similar = "secondly stupid considering"
    trigger_title_only_similar = "only_similar"
    # 相似度+去除异常值
    trigger_text_similar_denoise = "extremely shitty presenting"
    trigger_title_similar_denoise = "similar_denoise"
    # 相似度+mse约束+去噪
    trigger_text_similar_mse_denoise = "##ining felt stupid"
    trigger_title_similar_mse_denoise = "similar_mse_denoise"
    # 相似度+mse（约束）+去噪+mse损失值
    trigger_text_similar_mse_denoise_mse_loss = "nickelodeon also complained"
    trigger_title_similar_mse_denoise_mse_loss = "similar_mse_denoise_mse_loss"

    # 直接使用uat的方法
    trigger_text_only_tirgger = "nickelodeon unhappy whether"
    trigger_title_only_tirgger= "only_tirgger"

    total = [
        [trigger_text_similar_mse, trigger_title_similar_mse],
        [trigger_text_similar_mse_90, trigger_title_similar_mse_90],
        [trigger_text_denoise_similar, trigger_title_denoise_similar],
        [trigger_text_only_mse, trigger_title_only_mse],
        [trigger_text_only_similar, trigger_title_only_similar],
        [trigger_text_similar_denoise, trigger_title_similar_denoise],
        [trigger_text_similar_mse_denoise, trigger_title_similar_mse_denoise],
        [trigger_text_similar_mse_denoise_mse_loss, trigger_title_similar_mse_denoise_mse_loss],
        [trigger_text_only_tirgger, trigger_title_only_tirgger]
    ]

    for text, title in total:
        adversarial_text = text
        adversarial_text_title = title

        all_texts, all_labels = detail_data(source_label, target_label, val_texts, val_labels, adversarial_text)

        all_dataloader = dataset_to_tensor(all_texts, all_labels, tokenizer)  # 攻击的目标类 目标是0
        features, labels_all = tsne_extract(model_features, all_dataloader, adversarial_text_title)
        draw_tsne(features, labels_all, 40, adversarial_text_title)


def tsne_trainer(feature, perplexity: int):
    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=3500, random_state=32, learning_rate=0.9)
    X_tsne = tsne.fit_transform(feature)
    print(feature.shape)
    print(X_tsne.shape)
    print("Org data dimension is {}.Embedded data dimension is {}".format(feature.shape[-1], X_tsne.shape[-1]))
    
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0) - 10, X_tsne.max(0) + 10
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    return X_norm


def draw_tsne_all(feature1, label1, feature2, label2, perplexity: int, similar=False):
    # 将所有的图花在一个图上面
    tsne_feature1 = tsne_trainer(feature1, perplexity)
    tsne_feature2 = tsne_trainer(feature2, perplexity)

    color_map = ['g','y','k','r','b','m','c'] # 7个类，准备7种颜色
    color_dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(25, 9))
    lengend_params = {
        'size': 18, 
        # "family": "serif",
    }
    label_dict = {
        "1": "目标类",
        "0": "原始类",
        "3": "对抗样本"
    }


    # 画第一个图
    for i in range(tsne_feature1.shape[0]):
        if str(label1[i]) == "3":
            axes[0].plot(tsne_feature1[i, 0], tsne_feature1[i, 1], marker='x', markersize=3, color=color_map[label1[i]], label=str(label1[i]), linestyle="None")
        if str(label1[i]) == "0":
            axes[0].plot(tsne_feature1[i, 0], tsne_feature1[i, 1], marker='s', markersize=3, color=color_map[label1[i]], label=str(label1[i]), linestyle="None")
        if str(label1[i]) == "1":
            axes[0].plot(tsne_feature1[i, 0], tsne_feature1[i, 1], marker='o', markersize=3, color=color_map[label1[i]], label=str(label1[i]), linestyle="None")
        color_dict[label1[i]] = color_map[label1[i]]
    
    axes[0].set_xticks([])  # 清除 x 轴刻度
    axes[0].set_yticks([])  # 清除 y 轴刻度
    # axes[0].set_title('有相似度约束')
    # axes[0].set_title('simialr')
    axes[0].axis('off')  # 取消坐标轴显示

    legend_handles = []
    for label, color in color_dict.items():
        # legend_handles.append(plt.Line2D([], [], color=color, label=label))
        legend_handles.append(plt.Line2D([], [], color=color, label=label))

    axes[0].legend(handles=legend_handles, prop=lengend_params, fontsize=10.5)

    # 画第二个图
    for i in range(tsne_feature2.shape[0]):
        if str(label2[i]) == "3":
            axes[1].plot(tsne_feature2[i, 0], tsne_feature2[i, 1], marker='x', markersize=3, color=color_map[label2[i]], label=str(label2[i]))
        if str(label2[i]) == "0":
            axes[1].plot(tsne_feature2[i, 0], tsne_feature2[i, 1], marker='s', markersize=3, color=color_map[label2[i]], label=str(label2[i]))
        if str(label2[i]) == "1":
            axes[1].plot(tsne_feature2[i, 0], tsne_feature2[i, 1], marker='o', markersize=3, color=color_map[label2[i]], label=str(label2[i]))
        color_dict[label2[i]] = color_map[label2[i]]
    
    axes[1].set_xticks([])  # 清除 x 轴刻度
    axes[1].set_yticks([])  # 清除 y 轴刻度
    # axes[1].set_title('无相似度约束')
    # axes[1].set_title('none similar')
    axes[1].axis('off')  # 取消坐标轴显示

    # legend_handles = []
    # for label, color in color_dict.items():
    #     legend_handles.append(plt.Line2D([], [], color=color, label=label))
    # 获取图例中的标记对象
    axes[1].legend(handles=legend_handles, prop=lengend_params, fontsize=10.5)

    plt.subplots_adjust(wspace=0.02)

    base_dir = "./tsne_result_two"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    plt.savefig(f"{base_dir}/tsne_{similar}.jpg", dpi=300, bbox_inches='tight', pad_inches=0.0)
    # plt.savefig(f"{base_dir}/tsne_{similar}.png", format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    print(f"保存路径：{base_dir}/tsne_{similar}.png")


def draw_tsne_all_test1(feature1, label1, feature2, label2, perplexity: int, similar=False):
    # 将所有的图花在一个图上面
    tsne_feature1 = tsne_trainer(feature1, perplexity)
    tsne_feature2 = tsne_trainer(feature2, perplexity)

    color_map = ['g','y','k','r','b','m','c'] # 7个类，准备7种颜色
    marker_map = ['s','o','k','x','b','m','c'] # 7个类，准备7种颜色
    color_dict = {}
    fig, axes = plt.subplots(1, 2, figsize=(25, 9))
    lengend_params = {
        'size': 18, 
        "family": "serif",
    }
    label_dict = {
        "1": "目标类",
        "0": "原始类",
        "3": "对抗样本"
    }
    for label in [0, 1, 3]:
        mask = label1 == label
        axes[0].scatter(tsne_feature1[mask, 0], tsne_feature1[mask, 1], marker=marker_map[label], s=35, c=color_map[label], label=f'{label}')

    for label in [0, 1, 3]:
        mask = label2 == label
        axes[1].scatter(tsne_feature2[mask, 0], tsne_feature2[mask, 1], marker=marker_map[label], s=35, c=color_map[label], label=f'{label}')

    axes[0].set_xticks([])  # 清除 x 轴刻度
    axes[0].set_yticks([])  # 清除 y 轴刻度
    # axes[0].set_title('有相似度约束')
    # axes[0].set_title('simialr')
    axes[0].axis('off')  # 取消坐标轴显示
    # axes[0].legend(prop=lengend_params, fontsize=10.5)
    # axes[0].legend(prop=lengend_params)

    axes[1].set_xticks([])  # 清除 x 轴刻度
    axes[1].set_yticks([])  # 清除 y 轴刻度
    # axes[1].set_title('无相似度约束')
    # axes[1].set_title('none similar')
    axes[1].axis('off')  # 取消坐标轴显示
    axes[1].legend(prop=lengend_params, fontsize=10.5)
    # axes[1].legend(prop=lengend_params)

    plt.subplots_adjust(wspace=0.02)

    base_dir = "./tsne_result_two"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    plt.savefig(f"{base_dir}/tsne_{similar}.jpg", dpi=300, bbox_inches='tight', pad_inches=0.0)
    # plt.savefig(f"{base_dir}/tsne_{similar}.png", format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    print(f"保存路径：{base_dir}/tsne_{similar}.png")

    exit()

    # 画第一个图
    for i in range(tsne_feature1.shape[0]):
        if str(label1[i]) == "3":
            axes[0].plot(tsne_feature1[i, 0], tsne_feature1[i, 1], marker='x', markersize=3, color=color_map[label1[i]], label=str(label1[i]), linestyle="None")
        if str(label1[i]) == "0":
            axes[0].plot(tsne_feature1[i, 0], tsne_feature1[i, 1], marker='s', markersize=3, color=color_map[label1[i]], label=str(label1[i]), linestyle="None")
        if str(label1[i]) == "1":
            axes[0].plot(tsne_feature1[i, 0], tsne_feature1[i, 1], marker='o', markersize=3, color=color_map[label1[i]], label=str(label1[i]), linestyle="None")
        color_dict[label1[i]] = color_map[label1[i]]
    
    axes[0].set_xticks([])  # 清除 x 轴刻度
    axes[0].set_yticks([])  # 清除 y 轴刻度
    # axes[0].set_title('有相似度约束')
    # axes[0].set_title('simialr')
    axes[0].axis('off')  # 取消坐标轴显示

    legend_handles = []
    for label, color in color_dict.items():
        # legend_handles.append(plt.Line2D([], [], color=color, label=label))
        legend_handles.append(plt.Line2D([], [], color=color, label=label))

    axes[0].legend(handles=legend_handles, prop=lengend_params, fontsize=10.5)

    # 画第二个图
    for i in range(tsne_feature2.shape[0]):
        if str(label2[i]) == "3":
            axes[1].plot(tsne_feature2[i, 0], tsne_feature2[i, 1], marker='x', markersize=3, color=color_map[label2[i]], label=str(label2[i]))
        if str(label2[i]) == "0":
            axes[1].plot(tsne_feature2[i, 0], tsne_feature2[i, 1], marker='s', markersize=3, color=color_map[label2[i]], label=str(label2[i]))
        if str(label2[i]) == "1":
            axes[1].plot(tsne_feature2[i, 0], tsne_feature2[i, 1], marker='o', markersize=3, color=color_map[label2[i]], label=str(label2[i]))
        color_dict[label2[i]] = color_map[label2[i]]
    
    axes[1].set_xticks([])  # 清除 x 轴刻度
    axes[1].set_yticks([])  # 清除 y 轴刻度
    # axes[1].set_title('无相似度约束')
    # axes[1].set_title('none similar')
    axes[1].axis('off')  # 取消坐标轴显示

    # legend_handles = []
    # for label, color in color_dict.items():
    #     legend_handles.append(plt.Line2D([], [], color=color, label=label))
    # 获取图例中的标记对象
    axes[1].legend(handles=legend_handles, prop=lengend_params, fontsize=10.5)

    plt.subplots_adjust(wspace=0.02)

    base_dir = "./tsne_result_two"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    plt.savefig(f"{base_dir}/tsne_{similar}.jpg", dpi=300, bbox_inches='tight', pad_inches=0.0)
    # plt.savefig(f"{base_dir}/tsne_{similar}.png", format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    print(f"保存路径：{base_dir}/tsne_{similar}.png")

def store_file(embeddings_features, embeddings_labels, similar):
    print(embeddings_features.shape)
    print(embeddings_features)
    print(embeddings_labels.shape)
    print(embeddings_labels)
    base_dir = "./tsne_result_npy_files"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    # 将embedding和标签保存到文件中
    np.save(base_dir + "/" + f'{similar}_embedding.npy', embeddings_features)
    np.save(base_dir + "/" + f'{similar}_labels.npy', embeddings_labels)
    print(f"{similar}存储完成")

def load_file(similar):
    base_dir = "./tsne_result_npy_files"
    path_embeddins = base_dir + "/" + f'{similar}_embedding.npy'
    path_labels = base_dir + "/" + f'{similar}_labels.npy'
    features_embedding = np.load(path_embeddins)
    features_labels = np.load(path_labels)
    return features_embedding, features_labels



def main1():
    tokenizer = BertTokenizer.from_pretrained('./bert-tomatoes')
    model_features = BertModel.from_pretrained('./bert-tomatoes')
    model_features.to(device)

    val_texts, val_labels = read_datasets.read_csv("test")

    source_label = 1
    target_label = 0
    # 使用相似度+mse约束
    trigger_text_similar_mse = "secondly utterly poorly"
    trigger_title_similar_mse = "similar_mse"
    # 使用相似度+mse约束+去除噪声
    trigger_text_similar_mse_90 = "ebert ruined how"
    trigger_title_similar_mse_90 = "similar_mse_90"
    # 使用去噪+相似度的方法（论文中的）
    trigger_text_denoise_similar = "secondly really poorly"
    trigger_title_denoise_similar = "denoise_similar"
    # 只使用mse
    trigger_text_only_mse = "secondly boring reviewing"
    trigger_title_only_mse = "only_mse"
    # 只使用相似度
    trigger_text_only_similar = "secondly stupid considering"
    trigger_title_only_similar = "only_similar"
    # 相似度+去除异常值
    trigger_text_similar_denoise = "extremely shitty presenting"
    trigger_title_similar_denoise = "similar_denoise"
    # 相似度+mse约束+去噪
    trigger_text_similar_mse_denoise = "##ining felt stupid"
    trigger_title_similar_mse_denoise = "similar_mse_denoise"
    # 相似度+mse（约束）+去噪+mse损失值
    trigger_text_similar_mse_denoise_mse_loss = "nickelodeon also complained"
    trigger_title_similar_mse_denoise_mse_loss = "similar_mse_denoise_mse_loss"

    # 直接使用uat的方法
    trigger_text_only_tirgger = "nickelodeon unhappy whether"
    trigger_title_only_tirgger= "only_tirgger"

    total = [
        [trigger_text_similar_mse, trigger_title_similar_mse],
        [trigger_text_similar_mse_90, trigger_title_similar_mse_90],
        [trigger_text_denoise_similar, trigger_title_denoise_similar],
        [trigger_text_only_mse, trigger_title_only_mse],
        [trigger_text_only_similar, trigger_title_only_similar],
        [trigger_text_similar_denoise, trigger_title_similar_denoise],
        [trigger_text_similar_mse_denoise, trigger_title_similar_mse_denoise],
        [trigger_text_similar_mse_denoise_mse_loss, trigger_title_similar_mse_denoise_mse_loss],    
        # [trigger_text_only_tirgger, trigger_title_only_tirgger]
    ]

    # 获取只有trigger的特征
    only_trigger_texts, only_trigger_labels = detail_data(source_label, target_label, val_texts, val_labels, trigger_text_only_tirgger)
    only_trigger_dataloader = dataset_to_tensor(only_trigger_texts, only_trigger_labels, tokenizer)  # 攻击的目标类 目标是0
    only_trigger_features, only_trigger_labels_all = tsne_extract(model_features, only_trigger_dataloader, trigger_title_only_tirgger)
    store_file(only_trigger_features, only_trigger_labels_all, trigger_title_only_tirgger)
    for text, title in total:
        adversarial_text = text
        adversarial_text_title = title

        all_texts, all_labels = detail_data(source_label, target_label, val_texts, val_labels, adversarial_text)


        all_dataloader = dataset_to_tensor(all_texts, all_labels, tokenizer)  # 攻击的目标类 目标是0
        features, labels_all = tsne_extract(model_features, all_dataloader, adversarial_text_title)
        draw_tsne_all_test1(features, labels_all, only_trigger_features, only_trigger_labels_all, 40, adversarial_text_title + "_" + trigger_title_only_tirgger)
        store_file(features, labels_all, adversarial_text_title)
        # exit()

def main2():
    # 使用相似度+mse约束
    trigger_text_similar_mse = "secondly utterly poorly"
    trigger_title_similar_mse = "similar_mse"
    # 使用相似度+mse约束+去除噪声
    trigger_text_similar_mse_90 = "ebert ruined how"
    trigger_title_similar_mse_90 = "similar_mse_90"
    # 使用去噪+相似度的方法（论文中的）
    trigger_text_denoise_similar = "secondly really poorly"
    trigger_title_denoise_similar = "denoise_similar"
    # 只使用mse
    trigger_text_only_mse = "secondly boring reviewing"
    trigger_title_only_mse = "only_mse"
    # 只使用相似度
    trigger_text_only_similar = "secondly stupid considering"
    trigger_title_only_similar = "only_similar"
    # 相似度+去除异常值
    trigger_text_similar_denoise = "extremely shitty presenting"
    trigger_title_similar_denoise = "similar_denoise"
    # 相似度+mse约束+去噪
    trigger_text_similar_mse_denoise = "##ining felt stupid"
    trigger_title_similar_mse_denoise = "similar_mse_denoise"
    # 相似度+mse（约束）+去噪+mse损失值
    trigger_text_similar_mse_denoise_mse_loss = "nickelodeon also complained"
    trigger_title_similar_mse_denoise_mse_loss = "similar_mse_denoise_mse_loss"

    # 直接使用uat的方法
    trigger_text_only_tirgger = "nickelodeon unhappy whether"
    trigger_title_only_tirgger= "only_tirgger"

    total = [
        [trigger_text_similar_mse, trigger_title_similar_mse],
        [trigger_text_similar_mse_90, trigger_title_similar_mse_90],
        [trigger_text_denoise_similar, trigger_title_denoise_similar],
        [trigger_text_only_mse, trigger_title_only_mse],
        [trigger_text_only_similar, trigger_title_only_similar],
        [trigger_text_similar_denoise, trigger_title_similar_denoise],
        [trigger_text_similar_mse_denoise, trigger_title_similar_mse_denoise],
        [trigger_text_similar_mse_denoise_mse_loss, trigger_title_similar_mse_denoise_mse_loss],    
        # [trigger_text_only_tirgger, trigger_title_only_tirgger]
    ]

    only_trigger_features, only_trigger_labels_all = load_file(trigger_title_only_tirgger)
    for text, title in total:
        adversarial_text_title = title
        features, labels_all = load_file(adversarial_text_title)
        draw_tsne_all_test1(features, labels_all, only_trigger_features, only_trigger_labels_all, 40, adversarial_text_title + "_" + trigger_title_only_tirgger)




if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # main() # 单图
    # main1()  # 多图
    main2()  # 从文件读取
