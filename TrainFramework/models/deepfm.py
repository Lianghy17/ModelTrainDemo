import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFM(nn.Module):
    def __init__(self, discrete_feature_dims, continuous_features, embedding_dim=8, hidden_units=[64, 32, 16], dropout_rate=0.2):
        super(DeepFM, self).__init__()

        self.discrete_feature_dims = discrete_feature_dims
        self.continuous_features = continuous_features
        self.embedding_dim = embedding_dim
        self.all_features = list(discrete_feature_dims.keys()) + continuous_features

        # FM部分 - 线性层（仅离散特征）
        self.linear_layers = nn.ModuleDict()
        for feature_name, dim in discrete_feature_dims.items():
            self.linear_layers[feature_name] = nn.Embedding(dim, 1)

        # FM部分 - 交叉项嵌入（仅离散特征）
        self.embedding_layers = nn.ModuleDict()
        for feature_name, dim in discrete_feature_dims.items():
            self.embedding_layers[feature_name] = nn.Embedding(dim, embedding_dim)

        # 连续特征的线性权重
        self.continuous_weights = nn.ParameterDict({
            feature_name: nn.Parameter(torch.randn(1))
            for feature_name in continuous_features
        })

        # Deep部分
        total_embedding_dim = len(discrete_feature_dims) * embedding_dim + len(continuous_features)
        deep_layers = []
        input_dim = total_embedding_dim

        for hidden_unit in hidden_units:
            deep_layers.append(nn.Linear(input_dim, hidden_unit))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_unit

        deep_layers.append(nn.Linear(input_dim, 1))
        self.deep_network = nn.Sequential(*deep_layers)

        # 输出层
        self.output_layer = nn.Sigmoid()

    def forward(self, features):
        # FM线性部分
        linear_terms = []
        embeddings = []

        # 处理离散特征
        for feature_name, dim in self.discrete_feature_dims.items():
            feature_values = features[feature_name]
            # 线性项
            linear_emb = self.linear_layers[feature_name](feature_values.long())
            linear_terms.append(linear_emb.squeeze(-1))

            # 嵌入向量
            emb = self.embedding_layers[feature_name](feature_values.long())
            embeddings.append(emb)

        # 处理连续特征
        for feature_name in self.continuous_features:
            feature_values = features[feature_name]
            linear_terms.append(self.continuous_weights[feature_name] * feature_values)
            embeddings.append(feature_values.unsqueeze(1))  # [batch_size, 1]

        # 线性部分求和
        linear_sum = torch.sum(torch.stack(linear_terms), dim=0)

        # FM交叉部分（仅离散特征）
        if embeddings:
            emb_concat = torch.cat(embeddings, dim=1)  # [batch_size, num_features * embedding_dim]
            deep_output = self.deep_network(emb_concat)
        else:
            deep_output = self.deep_network(torch.zeros(features[list(features.keys())[0]].shape[0], 1))

        # 合并FM和Deep部分
        final_output = linear_sum.unsqueeze(1) + deep_output

        return self.output_layer(final_output)


class MLP(nn.Module):
    def __init__(self, discrete_feature_dims, continuous_features, embedding_dim=8, hidden_units=[64, 32, 16], dropout_rate=0.2):
        super(MLP, self).__init__()

        self.discrete_feature_dims = discrete_feature_dims
        self.continuous_features = continuous_features
        self.embedding_dim = embedding_dim

        # 嵌入层（仅离散特征）
        self.embedding_layers = nn.ModuleDict()
        for feature_name, dim in discrete_feature_dims.items():
            self.embedding_layers[feature_name] = nn.Embedding(dim, embedding_dim)

        # MLP部分
        total_embedding_dim = len(discrete_feature_dims) * embedding_dim + len(continuous_features)
        mlp_layers = []
        input_dim = total_embedding_dim

        for hidden_unit in hidden_units:
            mlp_layers.append(nn.Linear(input_dim, hidden_unit))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_unit

        mlp_layers.append(nn.Linear(input_dim, 1))
        self.mlp_network = nn.Sequential(*mlp_layers)

        # 输出层
        self.output_layer = nn.Sigmoid()

    def forward(self, features):
        embeddings = []

        # 处理离散特征
        for feature_name, dim in self.discrete_feature_dims.items():
            feature_values = features[feature_name]
            emb = self.embedding_layers[feature_name](feature_values.long())
            embeddings.append(emb)

        # 处理连续特征
        for feature_name in self.continuous_features:
            feature_values = features[feature_name]
            embeddings.append(feature_values.unsqueeze(1))

        # 拼接所有嵌入向量
        emb_concat = torch.cat(embeddings, dim=1)

        # MLP前向传播
        output = self.mlp_network(emb_concat)

        return self.output_layer(output)


class WideAndDeep(nn.Module):
    def __init__(self, discrete_feature_dims, continuous_features, embedding_dim=8, hidden_units=[64, 32, 16], dropout_rate=0.2):
        super(WideAndDeep, self).__init__()

        self.discrete_feature_dims = discrete_feature_dims
        self.continuous_features = continuous_features
        self.embedding_dim = embedding_dim

        # Wide部分（线性）- 仅离散特征
        self.linear_layers = nn.ModuleDict()
        for feature_name, dim in discrete_feature_dims.items():
            self.linear_layers[feature_name] = nn.Embedding(dim, 1)

        # 连续特征的线性权重
        self.continuous_weights = nn.ParameterDict({
            feature_name: nn.Parameter(torch.randn(1))
            for feature_name in continuous_features
        })

        # Deep部分嵌入（仅离散特征）
        self.embedding_layers = nn.ModuleDict()
        for feature_name, dim in discrete_feature_dims.items():
            self.embedding_layers[feature_name] = nn.Embedding(dim, embedding_dim)

        # Deep网络
        total_embedding_dim = len(discrete_feature_dims) * embedding_dim + len(continuous_features)
        deep_layers = []
        input_dim = total_embedding_dim

        for hidden_unit in hidden_units:
            deep_layers.append(nn.Linear(input_dim, hidden_unit))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_unit

        deep_layers.append(nn.Linear(input_dim, 1))
        self.deep_network = nn.Sequential(*deep_layers)

        # 输出层
        self.output_layer = nn.Sigmoid()

    def forward(self, features):
        # Wide部分
        linear_terms = []

        # 处理离散特征的线性部分
        for feature_name, dim in self.discrete_feature_dims.items():
            feature_values = features[feature_name]
            linear_emb = self.linear_layers[feature_name](feature_values.long())
            linear_terms.append(linear_emb.squeeze(-1))

        # 处理连续特征的线性部分
        for feature_name in self.continuous_features:
            feature_values = features[feature_name]
            linear_terms.append(self.continuous_weights[feature_name] * feature_values)

        wide_output = torch.sum(torch.stack(linear_terms), dim=0).unsqueeze(1)

        # Deep部分
        embeddings = []

        # 处理离散特征
        for feature_name, dim in self.discrete_feature_dims.items():
            feature_values = features[feature_name]
            emb = self.embedding_layers[feature_name](feature_values.long())
            embeddings.append(emb)

        # 处理连续特征
        for feature_name in self.continuous_features:
            feature_values = features[feature_name]
            embeddings.append(feature_values.unsqueeze(1))

        emb_concat = torch.cat(embeddings, dim=1)
        deep_output = self.deep_network(emb_concat)

        # 合并Wide和Deep
        final_output = wide_output + deep_output

        return self.output_layer(final_output)
