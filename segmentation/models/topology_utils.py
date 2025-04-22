from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance

def compute_persistent_diagram(likelihood_map, superlevel=False, dim=None):
    """
    计算给定概率图的持久图 (Persistent Diagram)
    :param likelihood_map: 模型输出的概率图，形状为 (B, N, C) 或 (N, C)
    :param superlevel: 是否基于超水平集计算拓扑特征，默认为 False（即基于子水平集）
    :param dim: 输入数据的维度，默认为 None，自动推断
    :return: 持久图
    """
    if len(likelihood_map.shape) == 2:
        likelihood_map = likelihood_map.unsqueeze(0)
    
    batch_size, num_points, num_classes = likelihood_map.shape

    flattened_map = likelihood_map.view(batch_size, -1)
    cubical_complex = CubicalComplex(superlevel=superlevel, dim=dim)
    
    return cubical_complex(flattened_map)
    