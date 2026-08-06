import torch.utils.data
import numpy as np
import scipy.io  # 必须引入 scipy
# import h5py    # 你的数据不需要 h5py，除非是 Matlab v7.3 格式

def CreateDatasetSynthesis(phase, input_path, contrast1='Fundus', contrast2='FFA'):
    # 读取输入图 (Source)
    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast1)
    data_fs_s1 = LoadDataSet(target_file, variable='data')
    
    # 读取真值图 (Target)
    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast2)
    data_fs_s2 = LoadDataSet(target_file, variable='data')

    # 【新增】：读取你的手工病灶标注/权重掩码 (Mask)
    target_file_w = input_path + "/data_{}_weight.mat".format(phase)
    # 如果找不到 weight 文件，生成全 1 的 fallback（防止报错）
    import os
    if os.path.exists(target_file_w):
        data_weight = LoadDataSet(target_file_w, variable='data', Norm=False) # Mask 不需要 [-1, 1] 归一化
    else:
        print(f"⚠️ 警告: 未找到 {target_file_w}，使用全 1 掩码代替。")
        data_weight = np.ones((data_fs_s1.shape[0], 1, data_fs_s1.shape[2], data_fs_s1.shape[3]), dtype=np.float32)

    # 打包成 TensorDataset，现在它会返回三个变量：(x, y, mask)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(data_fs_s1), 
        torch.from_numpy(data_fs_s2),
        torch.from_numpy(data_weight) # 新增的 Mask
    )  
    return dataset


def LoadDataSet(load_dir, variable='data', padding=False, Norm=True):
    """
    适配 Fundus-FFA 数据集的加载函数
    """
    print(f"正在加载数据: {load_dir} ...")
    
    try:
        # 1. 使用 scipy.io.loadmat (专门读取 Python 生成的 .mat)
        f = scipy.io.loadmat(load_dir)
        data = f[variable] # 直接读取，形状应该是 (N, 3, 256, 256)
    except Exception as e:
        print(f"Scipy 读取失败，尝试 h5py (可能是 Matlab v7.3 格式): {e}")
        import h5py
        f = h5py.File(load_dir, 'r')
        data = np.array(f[variable])
        # 如果是 h5py 读取的，通常维度会反转，需要检查 transpose
        # 但针对我们之前生成的脚本，scipy 应该能成功

    data = data.astype(np.float32)
    
    # 2. 检查维度 (调试用，防止出错)
    # 期望是 (N, 3, 256, 256)
    if data.ndim == 3: 
        # 预防万一它是 (N, 256, 256) 的灰度图，手动加通道
        data = np.expand_dims(data, axis=1) 
    
    # 3. Padding (你的数据已经是 256x256，通常不需要这个)
    # 只有当 padding=True 且尺寸小于 256 时才运行
    if padding and (data.shape[2] < 256 or data.shape[3] < 256):
        pad_x = int((256 - data.shape[2]) / 2)
        pad_y = int((256 - data.shape[3]) / 2)
        print('padding in x-y with:' + str(pad_x) + '-' + str(pad_y))
        # 注意 padding 格式
        data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)), 'constant')   

    # 4. 归一化 [0, 1] -> [-1, 1]
    if Norm:    
        data = (data - 0.5) / 0.5      
        
    return data