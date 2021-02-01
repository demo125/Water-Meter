import torch
import torch.nn.functional as F
import numpy as np

C = {0: 0,
     1: 1,
     2: 2,
     3: 3,
     4: 4,
     5: 5,
     6: 6,
     7: 7,
     8: 8,
     9: 9,
     10: 'b',
     11: 11
    #  10: 10,
    #  11: 11,
    #  12: 12,
    #  13: 13,
    #  14: 14,
    #  15: 15,
    #  16: 16,
    #  17: 17,
    #  18: 18,
    #  19: 19,
    #  21: 21,
    #  20: 'b',
     }
# 定义函数delet_char, 去重序列，并删除空白字符b
def get_preTarget(decoded_batch_classes, batch_class_probs):
    a = decoded_batch_classes
    c = batch_class_probs
    for i in range(len(a)):
        for j in range(len(a[i]) - 1, 0, -1):
            if a[i][j] == a[i][j - 1]:
                del a[i][j]
                del batch_class_probs[i][j]
        while 'b' in a[i]:
            rmv_idx = a[i].index('b')
            a[i] = a[i][:rmv_idx] + a[i][rmv_idx + 1:]
            c[i] = c[i][:rmv_idx] + c[i][rmv_idx + 1:]
    return a, c


# 将得到的预测序列转换为实际读数
# def middle_char(y):
#     for i in range(len(y)):
#         if y[i]>9 and y[i]!='b':
#             if i==len(y):
#                 y[i]=y[i]-9.5
#             else:
#                 y[i] = y[i]-10
#     return y

def feature_to_y(x, return_probs=False):
    x = F.softmax(x, dim=1) # 把特征序列转换为概率
    batch_class_probs, index = torch.max(x, dim=1) # 选出概率最大的那个
    index = torch.squeeze(index) # 压缩index维度
    batch_class_probs = torch.squeeze(batch_class_probs)
    if torch.cuda.is_available():
        index = index.cpu()
        batch_class_probs = batch_class_probs.cpu()
    index = index.numpy().tolist() # 把index转换为list，方便遍历
    batch_class_probs = batch_class_probs.numpy().tolist()

    if np.ndim(index) == 1:
        index = np.expand_dims(index, axis=0)
        batch_class_probs = np.expand_dims(batch_class_probs, axis=0)

    decoded_batch_classes = []
    for i in range(len(index)):
        decoded_img_classes = []
        for j in index[i]:
            decoded_img_classes.append(C[j])    # 把概率转换为字符序列
        decoded_batch_classes.append(decoded_img_classes)

    y, probs = get_preTarget(decoded_batch_classes, batch_class_probs)    # 字符序列去重，保留原有顺序,并删除空白字符b,这时得到的结果与target一致
    if return_probs:
        return y, probs
    else:
        return y