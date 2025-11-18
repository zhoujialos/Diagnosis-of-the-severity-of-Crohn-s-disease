import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
from PerClassInspector import PerClassInspector
# 数据处理相关
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
import torch
torch.use_deterministic_algorithms(True)
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners

# 贝叶斯优化
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
# 可视化相关
import matplotlib.pyplot as plt
import seaborn as sns
from visualize_metric_learning import MetricLearningVisualizer
# 修复numpy版本兼容性问题
np.int = int
def set_all_seeds(seed=42):
    """设置所有随机种子以确保可重现性"""
    import random
    import os
    
    # Python原生随机
    random.seed(seed)
    
    # NumPy随机
    np.random.seed(seed)
    
    # PyTorch随机
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 环境变量设置
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"已设置所有随机种子为: {seed}")

class ComparisonConfig:
    """三种方法对比配置"""
    
    # 数据路径
    TRAIN_DATA_PATH = 'train_shap.csv'
    TEST_DATA_PATH = 'test_shap.csv'
    
    # 基础配置
    N_SPLITS = 5
    RANDOM_STATE = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 优化相关
    N_TRIALS = 100  
    N_WARMUP_STEPS = 10
    N_STARTUP_TRIALS = 5
    
    # Few-Shot Learning配置
    N_WAY = 3  # 3类分类
    N_SUPPORT = 5  # 每类5个支持样本
    N_QUERY = 3   # 每类3个查询样本
    
    # 文本特征词汇表
    VOCABULARY = {
        "thickening and narrowing of the small intestine": 0,
        "thickening and narrowing of the colon": 1,
        "rectal thickening and narrowing": 2,
        "thickening and narrowing of the small intestine with expansion": 3,
        "small intestinal fistula": 4,
        "colon fistula": 5,
        "anal fistula": 6,
        "small bowel abscess": 7,
        "rectal fistula": 8,
        "thickening of the small intestine": 9,
        "thickening and narrowing of the colon with expansion": 10,
        "anal abscess": 11,
        "colon abscess": 12
    }
    
    # 特征列名
    CONTINUOUS_FEATURES = ['CRP', 'age']
    CATEGORICAL_FEATURES = ['CDAI_score', 'SESCD_score', 'FC', 'gender', 'smoking', 'education']
    
    @classmethod
    def get_cv_splitter(cls):
        """获取统一的交叉验证分割器"""
        return StratifiedKFold(
            n_splits=cls.N_SPLITS,
            shuffle=True,
            random_state=cls.RANDOM_STATE
        )

class DataProcessor:
    """数据预处理类（文本短语 -> 二值 0/1）"""

    def __init__(self, vocabulary: Dict[str, int]):
        self.vocabulary = vocabulary
        # 出现=1，不出现=0
        self.vectorizer = CountVectorizer(
            vocabulary=vocabulary,     # 键=短语，值=列索引
            ngram_range=(1, 7),        # 覆盖 multi-word 短语
            binary=True,               # 关键：二值
            lowercase=True,            # 与清洗一致
            stop_words=None            
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.ordinal_features = ['CDAI_score', 'SESCD_score', 'FC']
        self.nominal_features = ['gender', 'smoking', 'education']
        self.nominal_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    @staticmethod
    def replace_text(text: str) -> str:
        """小写、去标点（保留空格），不删除 and"""
        if pd.isna(text):
            return ""
        s = str(text).lower()
        s = re.sub(r'[^a-z0-9\s]', ' ', s)   # 标点 -> 空格
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for col in ComparisonConfig.CONTINUOUS_FEATURES + ComparisonConfig.CATEGORICAL_FEATURES:
            if col in data.columns:
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
        return data

    def extract_features(self, data: pd.DataFrame, fit_transform: bool = False) -> np.ndarray:
        """提取所有特征：文本短语(0/1) + 分类原值 + 连续(标准化)"""
        data = data.copy()
        data["processed_text"] = data["text"].apply(self.replace_text)
        data = self.fill_missing_values(data)

        # 文本特征 -> 0/1
        if fit_transform:
            text_features = self.vectorizer.fit_transform(data['processed_text'].values.astype('U'))
        else:
            if not self.is_fitted:
                self.vectorizer.fit(data['processed_text'].values.astype('U'))
            text_features = self.vectorizer.transform(data['processed_text'].values.astype('U'))

        # 连续特征（标准化）
        continuous_features = data[ComparisonConfig.CONTINUOUS_FEATURES].values.astype(np.float64)
        if fit_transform:
            continuous_features = self.scaler.fit_transform(continuous_features)
            self.is_fitted = True
        else:
            continuous_features = self.scaler.transform(continuous_features)

        categorical_features = data[ComparisonConfig.CATEGORICAL_FEATURES].values.astype(np.float64)
        # 拼接
        all_features = np.concatenate([
            text_features.toarray().astype(np.float32),   # 0/1
            categorical_features.astype(np.float32),
            continuous_features.astype(np.float32)
        ], axis=1)

        return all_features

    # 给出与拼接顺序一致的列名，方便 SHAP
    def get_feature_names(self) -> list:
        text_names = [None] * len(self.vocabulary)
        for k, i in self.vocabulary.items():
            text_names[i] = k
        return text_names + ComparisonConfig.CATEGORICAL_FEATURES + ComparisonConfig.CONTINUOUS_FEATURES


class MedicalDataset(torch.utils.data.Dataset):
    """医疗数据集类"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 transform=None, device: str = 'cpu'):
        self.device = device
        self.transform = transform
        
        if hasattr(features, 'values'):
            features = features.values
        if hasattr(labels, 'values'):
            labels = labels.values
        
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        
        if self.device != 'cpu':
            self.features = self.features.to(self.device)
            self.labels = self.labels.to(self.device)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        label = self.labels[index]
        
        if self.transform is not None:
            features = self.transform(features)
            
        return features, label

class FewShotDataset(torch.utils.data.Dataset):
    """Few-Shot Learning数据集"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 n_way: int, n_support: int, n_query: int, device: str = 'cpu'):
        self.device = device
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        
        # 转换为tensor
        if hasattr(features, 'values'):
            features = features.values
        if hasattr(labels, 'values'):
            labels = labels.values
            
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        
        # 按类别组织数据
        self.classes = torch.unique(self.labels).tolist()
        self.class_to_indices = {}
        for class_id in self.classes:
            self.class_to_indices[class_id] = torch.where(self.labels == class_id)[0]
        
        if self.device != 'cpu':
            self.features = self.features.to(self.device)
            self.labels = self.labels.to(self.device)
    
    def __len__(self) -> int:
        return 1000  # 生成1000个episode
    
    def __getitem__(self, index: int):
        """生成一个Few-Shot Learning episode"""
        # 随机选择n_way个类别
        selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
        
        support_features = []
        support_labels = []
        query_features = []
        query_labels = []
        
        for i, class_id in enumerate(selected_classes):
            class_indices = self.class_to_indices[class_id]
            need = self.n_support + self.n_query
            if len(class_indices) >= need:
                selected = class_indices[torch.randperm(len(class_indices), device=class_indices.device)[:need]]
            else:
                sel_idx = torch.randint(0, len(class_indices), (need,), device=class_indices.device)
                selected = class_indices[sel_idx]

            support_indices = selected[:self.n_support]
            query_indices = selected[self.n_support:]

            support_features.append(self.features[support_indices])
            query_features.append(self.features[query_indices])

            support_labels.extend([i] * support_indices.numel())
            query_labels.extend([i] * query_indices.numel())
# 转换为tensor
        support_features = torch.cat(support_features, dim=0)
        query_features = torch.cat(query_features, dim=0)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        
        if self.device != 'cpu':
            support_labels = support_labels.to(self.device)
            query_labels = query_labels.to(self.device)
        
        return support_features, support_labels, query_features, query_labels

# ==================== 共享特征提取器 ====================
class SharedBackbone(nn.Module):
    """三种方法共享的特征提取器"""
    
    def __init__(self, input_size: int, hidden_size: int, embedding_size: int, 
                 dropout_rate: float, device: str = 'cpu'):
        super().__init__()
        
        # Trunk网络
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7)
        )
        
        # Embedder网络
        self.embedder = nn.Sequential(
            nn.Linear(hidden_size // 2, embedding_size),
            nn.ReLU()
        )
        
        self.to(device)
    
    def forward(self, x):
        trunk_output = self.trunk(x)
        embeddings = self.embedder(trunk_output)
        return trunk_output, embeddings

# ==================== 1. 度量学习方法 ====================
class MetricLearningModel(nn.Module):
    """度量学习模型"""
    
    def __init__(self, input_size: int, hidden_size: int, embedding_size: int, 
                 num_classes: int, dropout_rate: float, device: str = 'cpu'):
        super().__init__()
        
        # 共享backbone
        self.backbone = SharedBackbone(input_size, hidden_size, embedding_size, dropout_rate, device)
        
        # 分类器
        self.classifier = nn.Linear(embedding_size, num_classes)
        
        self.to(device)
    
    def forward(self, x):
        trunk_output, embeddings = self.backbone(x)
        logits = self.classifier(embeddings)
        return trunk_output, embeddings, logits

# ==================== 2. 匹配网络 ====================
class DistanceNetwork(nn.Module):
    """距离网络"""
    
    def __init__(self):
        super(DistanceNetwork, self).__init__()
    
    def forward(self, support_embeddings, query_embeddings):
        """计算余弦相似度"""
        # 归一化
        support_norm = F.normalize(support_embeddings, p=2, dim=1)
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        
        # 计算余弦相似度 [n_query, n_support]
        similarities = torch.mm(query_norm, support_norm.t())
        return similarities


class AttentionalClassify(nn.Module):
    """注意力分类器"""

    def __init__(self):
        super(AttentionalClassify, self).__init__()

        # 可学习的温度/缩放系数（初始化为 10.0，常见于对比学习/ArcFace 等）
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
    def forward(self, similarities, support_labels):
        """
        Args:
            similarities: [n_query, n_support]  # 余弦相似度，未归一化 logits
            support_labels: [n_support] (one-hot或标量)
        Returns:
            logits: [n_query, n_classes]
        """
        # 对相似度施加可学习缩放，等价于温度缩放（温度=1/scale）
        similarities = similarities * self.logit_scale
        # 不再对 similarities 做 softmax，直接把支持集的标签 one-hot 聚合为类打分
        if support_labels.dim() == 1:
            n_classes = support_labels.max().item() + 1
            support_labels_onehot = F.one_hot(support_labels, n_classes).float()
        else:
            support_labels_onehot = support_labels.float()

        logits = torch.mm(similarities, support_labels_onehot)  # [n_query, n_classes]
        return logits

class BidirectionalLSTM(nn.Module):
    """双向LSTM用于全上下文嵌入"""
    
    def __init__(self, embedding_size, hidden_size, device='cpu'):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, inputs, batch_size):
        """前向传播"""
        h0 = torch.zeros(2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(2, batch_size, self.hidden_size).to(self.device)
        
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        return output.squeeze(0), hn, cn

class MatchingNetwork(nn.Module):
    """匹配网络"""
    
    def __init__(self, input_size: int, hidden_size: int, embedding_size: int, 
                 num_classes: int, dropout_rate: float, use_fce: bool = False, 
                 lstm_hidden_size: int = 32, device: str = 'cpu'):
        super(MatchingNetwork, self).__init__()
        
        # 共享backbone（与度量学习完全相同）
        self.backbone = SharedBackbone(input_size, hidden_size, embedding_size, dropout_rate, device)
        
        self.use_fce = use_fce
        self.device = device
        
        # 可选的全上下文嵌入
        if use_fce:
            self.lstm = BidirectionalLSTM(embedding_size, lstm_hidden_size, device)
        
        # 匹配网络特有组件
        self.distance_network = DistanceNetwork()
        self.attentional_classify = AttentionalClassify()
        
        self.to(device)
    
    def forward(self, support_features, support_labels, query_features, query_labels=None):
        """
        Args:
            support_features: [n_support, feature_dim]
            support_labels: [n_support]
            query_features: [n_query, feature_dim] 
            query_labels: [n_query]
        """
        # 特征提取（与度量学习相同）
        _, support_embeddings = self.backbone(support_features)
        _, query_embeddings = self.backbone(query_features)
        
        # 可选：全上下文嵌入
        if self.use_fce:
            # 处理支持集
            support_embeddings, _, _ = self.lstm(support_embeddings, support_embeddings.size(0))
            # 处理查询集
            query_embeddings, _, _ = self.lstm(query_embeddings, query_embeddings.size(0))
        
        # 计算相似度
        similarities = self.distance_network(support_embeddings, query_embeddings)
        
        # 注意力分类
        predictions = self.attentional_classify(similarities, support_labels)
        
        return predictions

# ==================== 3. 原型网络 ====================
def euclidean_dist(x, y):
    """计算欧氏距离"""
    # x: N x D, y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception("维度不匹配")

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class PrototypicalNetwork(nn.Module):
    """原型网络"""
    
    def __init__(self, input_size: int, hidden_size: int, embedding_size: int, 
                 num_classes: int, dropout_rate: float, device: str = 'cpu'):
        super(PrototypicalNetwork, self).__init__()
        
        # 共享backbone（与度量学习完全相同）
        self.backbone = SharedBackbone(input_size, hidden_size, embedding_size, dropout_rate, device)
        
        self.device = device
        self.to(device)
    
    def forward(self, support_features, support_labels, query_features, query_labels=None):
        """
        Args:
            support_features: [n_support, feature_dim]
            support_labels: [n_support]
            query_features: [n_query, feature_dim]
            query_labels: [n_query] (可选，用于训练)
        """
        # 特征提取（与度量学习相同）
        _, support_embeddings = self.backbone(support_features)
        _, query_embeddings = self.backbone(query_features)
        
        # 计算类原型
        classes = torch.unique(support_labels)
        prototypes = []
        
        for class_id in classes:
            class_mask = support_labels == class_id
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)  # 类中心
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # [n_classes, embedding_dim]
        
        # 计算查询样本到原型的距离
        distances = euclidean_dist(query_embeddings, prototypes)  # [n_query, n_classes]
        # 直接返回类 logits（负距离），交给外部的 F.cross_entropy 处理
        logits = -distances
        return logits

# ==================== 传统机器学习训练器 ====================
class TraditionalMLTrainer:
    """传统机器学习训练器"""
    
    def __init__(self, train_features, train_labels, test_features, test_labels, train_df=None, processor=None):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        # 保留原始训练DataFrame与处理器
        self.train_df = train_df
        self.base_processor = processor
        
        self.input_size = train_features.shape[1]
        self.num_classes = len(np.unique(train_labels))
        
        print(f"初始化传统机器学习训练器:")
        print(f"  输入维度: {self.input_size}")
        print(f"  类别数: {self.num_classes}")
        print(f"  交叉验证: {ComparisonConfig.N_SPLITS}折") 
    
    def create_model_pipeline(self, model_name: str, trial):
        """创建模型管道"""
        if model_name == 'svm':
            C = trial.suggest_float('C', 0.1, 10.0, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            
            return svm.SVC(
                C=C, kernel=kernel, gamma=gamma,
                random_state=ComparisonConfig.RANDOM_STATE,
                probability=True,  # 修复AUC评估问题
                class_weight='balanced'  # 处理类别不平衡
            )
        
        elif model_name == 'random_forest':
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_categorical('max_depth', [None, 10, 50, 100])
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                random_state=ComparisonConfig.RANDOM_STATE,
                class_weight='balanced'  # 处理类别不平衡
            )
        
        elif model_name == 'decision_tree':
            max_depth = trial.suggest_categorical('max_depth', [None, 10, 50, 100])
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            
            return DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                random_state=ComparisonConfig.RANDOM_STATE,
                class_weight='balanced'
            )
        
        elif model_name == 'xgboost':
            n_estimators = trial.suggest_int('n_estimators', 100, 300)
            max_depth = trial.suggest_int('max_depth', 3, 9)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            
            return xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=ComparisonConfig.RANDOM_STATE,
                eval_metric='mlogloss',
                class_weight='balanced'
            )
        
        elif model_name == 'lightgbm':
            num_leaves = trial.suggest_int('num_leaves', 31, 100)
            max_depth = trial.suggest_int('max_depth', 3, 9)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            
            return lgb.LGBMClassifier(
                num_leaves=num_leaves,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=ComparisonConfig.RANDOM_STATE,
                objective='multiclass',
                num_class=self.num_classes,
                verbose=-1,
                class_weight='balanced'
            )
        
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
    
    def evaluate_sample_wise(self, model, val_features, val_labels):
        """统一的逐样本评估方法"""
        model.fit(self.train_features, self.train_labels)  # 在完整训练集上训练
        predictions = model.predict(val_features)
        accuracy = (predictions == val_labels).mean()
        return accuracy
    
    def objective(self, trial, model_name: str):
        """优化目标函数 - 使用统一的交叉验证"""
        # 使用固定随机种子确保可重现性
        set_all_seeds(ComparisonConfig.RANDOM_STATE)
        
        try:
            cv_scores = []
            cv_f1_scores = []  # 添加F1分数列表
            # 使用统一的交叉验证分割器
            skf = ComparisonConfig.get_cv_splitter()
            
            print(f"\n开始 {ComparisonConfig.N_SPLITS} 折交叉验证...")
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(self.train_features, self.train_labels)):
                print(f"  正在处理第 {fold + 1}/{ComparisonConfig.N_SPLITS} 折...")
                
                # 按折无泄漏特征提取：若提供原始train_df与processor，则每折独立fit/transform
                if self.train_df is not None:
                    # 每折使用全新的 DataProcessor（共享相同词表）
                    vocab = self.base_processor.vocabulary if self.base_processor is not None else ComparisonConfig.VOCABULARY
                    fold_processor = DataProcessor(vocabulary=vocab)

                    fold_train_df = self.train_df.iloc[train_idx]
                    fold_val_df = self.train_df.iloc[val_idx]

                    fold_train_features = fold_processor.extract_features(fold_train_df, fit_transform=True)
                    fold_train_labels = fold_train_df["label"].to_numpy().astype(np.int64)

                    fold_val_features = fold_processor.extract_features(fold_val_df, fit_transform=False)
                    fold_val_labels = fold_val_df["label"].to_numpy().astype(np.int64)
                
                # 创建模型
                model = self.create_model_pipeline(model_name, trial)
                
                # 训练和评估
                model.fit(fold_train_features, fold_train_labels)
                predictions = model.predict(fold_val_features)
                fold_score = (predictions == fold_val_labels).mean()
                fold_f1 = f1_score(fold_val_labels, predictions, average='weighted')  # 添加F1分数计算
                
                cv_scores.append(fold_score)
                cv_f1_scores.append(fold_f1)  # 保存F1分数
                print(f"    第 {fold + 1} 折准确率: {fold_score:.4f}, F1: {fold_f1:.4f}")
                
                # 剪枝判断
                trial.report(fold_score, fold)
                if trial.should_prune():
                    print(f"    试验在第 {fold + 1} 折被剪枝")
                    raise optuna.TrialPruned()
            
            # 计算平均分数
            avg_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            avg_f1 = np.mean(cv_f1_scores)  # 计算平均F1分数
            std_f1 = np.std(cv_f1_scores)   # 计算F1分数标准差
            
            trial.set_user_attr('fold_scores', cv_scores)
            trial.set_user_attr('fold_f1_scores', cv_f1_scores)  # 保存F1分数
            print(f"  交叉验证结果: 准确率 {avg_score:.4f} ± {std_score:.4f}, F1 {avg_f1:.4f} ± {std_f1:.4f}")
            
            return avg_score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"试验失败: {e}")
            return 0.0
    
    def optimize(self, model_name: str):
        """执行贝叶斯优化"""
        print(f"开始优化传统机器学习模型: {model_name}...")
        
        # 创建研究
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=ComparisonConfig.N_STARTUP_TRIALS),
            sampler=TPESampler(seed=ComparisonConfig.RANDOM_STATE),
            study_name=f'traditional_ml_{model_name}_optimization'
        )
        
        # 执行优化
        study.optimize(
            lambda trial: self.objective(trial, model_name),
            n_trials=ComparisonConfig.N_TRIALS,
            show_progress_bar=True
        )
        
        return study
    
    def train_final_model(self, model_name: str, best_params):
        """使用最佳参数训练最终模型"""
        print(f"\n使用最佳参数训练最终 {model_name} 模型...")
        
        # 设置随机种子
        set_all_seeds(ComparisonConfig.RANDOM_STATE)
        
        # 创建模型（手动设置参数）
        if model_name == 'svm':
            model = svm.SVC(
                C=best_params['C'],
                kernel=best_params['kernel'],
                gamma=best_params['gamma'],
                random_state=ComparisonConfig.RANDOM_STATE,
                probability=True,
                class_weight='balanced'
            )
        elif model_name == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                criterion=best_params['criterion'],
                random_state=ComparisonConfig.RANDOM_STATE,
                class_weight='balanced'
            )
        elif model_name == 'decision_tree':
            model = DecisionTreeClassifier(
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                criterion=best_params['criterion'],
                random_state=ComparisonConfig.RANDOM_STATE,
                class_weight='balanced'
            )
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                learning_rate=best_params['learning_rate'],
                random_state=ComparisonConfig.RANDOM_STATE,
                eval_metric='mlogloss'
            )
        elif model_name == 'lightgbm':
            model = lgb.LGBMClassifier(
                num_leaves=best_params['num_leaves'],
                max_depth=best_params['max_depth'],
                learning_rate=best_params['learning_rate'],
                random_state=ComparisonConfig.RANDOM_STATE,
                objective='multiclass',
                num_class=self.num_classes,
                verbose=-1
            )
        
        # 在完整训练集上训练
        model.fit(self.train_features, self.train_labels)
        
        return model

# ==================== 统一训练器 ====================
class UnifiedTrainer:
    """统一的训练器，支持三种方法"""
    
    def __init__(self, model_type: str, train_features, train_labels, test_features, test_labels, train_df, base_processor):
        self.model_type = model_type
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.device = ComparisonConfig.DEVICE
        self.train_df = train_df
        self.base_processor = base_processor
        self.input_size = train_features.shape[1]
        self.num_classes = len(np.unique(train_labels))
        
        print(f"初始化 {model_type} 训练器:")
        print(f"  设备: {self.device}")
        print(f"  输入维度: {self.input_size}")
        print(f"  类别数: {self.num_classes}")
        print(f"  交叉验证: {ComparisonConfig.N_SPLITS}折") 
        
    def _prepare_fold_data(self, train_idx, val_idx):
        vocab = self.base_processor.vocabulary if self.base_processor else ComparisonConfig.VOCABULARY
        proc = DataProcessor(vocabulary=vocab)
        df_tr = self.train_df.iloc[train_idx]
        df_va = self.train_df.iloc[val_idx]
        Xtr = proc.extract_features(df_tr, fit_transform=True)
        ytr = df_tr["label"].to_numpy(np.int64)
        Xva = proc.extract_features(df_va, fit_transform=False)
        yva = df_va["label"].to_numpy(np.int64)
        return Xtr, ytr, Xva, yva    

    
    def create_model(self, trial):
        """根据模型类型创建模型"""
        # 通用参数
        hidden_size = trial.suggest_int('hidden_size', 64, 256)
        embedding_size = trial.suggest_int('embedding_size', 32, 128)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        if self.model_type == 'metric_learning':
            return MetricLearningModel(
                input_size=self.input_size,
                hidden_size=hidden_size,
                embedding_size=embedding_size,
                num_classes=self.num_classes,
                dropout_rate=dropout_rate,
                device=self.device
            )
        
        elif self.model_type == 'matching_network':
            use_fce = trial.suggest_categorical('use_fce', [True, False])
            lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 16, 64) if use_fce else 32
            
            return MatchingNetwork(
                input_size=self.input_size,
                hidden_size=hidden_size,
                embedding_size=embedding_size,
                num_classes=self.num_classes,
                dropout_rate=dropout_rate,
                use_fce=use_fce,
                lstm_hidden_size=lstm_hidden_size,
                device=self.device
            )
        
        elif self.model_type == 'prototypical_network':
            return PrototypicalNetwork(
                input_size=self.input_size,
                hidden_size=hidden_size,
                embedding_size=embedding_size,
                num_classes=self.num_classes,
                dropout_rate=dropout_rate,
                device=self.device
            )

    def _load_metric_backbone_weights_(self, model, trained_models_dict):
        """
        train_metric_learning_fold/train_metric_learning 返回的 models(dict)
        中的 trunk/embedder 权重拷回到传入的 model.backbone.* 上。
        """
        if not hasattr(model, "backbone"):
            raise RuntimeError("Model has no backbone to load weights into.")

        if not hasattr(model.backbone, "trunk") or not hasattr(model.backbone, "embedder"):
            raise RuntimeError("Backbone missing trunk/embedder modules.")

        model.backbone.trunk.load_state_dict(trained_models_dict["trunk"].state_dict())
        model.backbone.embedder.load_state_dict(trained_models_dict["embedder"].state_dict())
        return model
    def train_few_shot_fold(self, model, trial, train_features, train_labels):
        """为单个fold训练Few-Shot Learning模型"""
        # 训练参数
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        num_episodes = trial.suggest_int('num_episodes', 50, 200)  # 减少episodes
        
        # 创建Few-Shot数据集
        train_dataset = FewShotDataset(
            train_features, train_labels,
            n_way=ComparisonConfig.N_WAY,
            n_support=ComparisonConfig.N_SUPPORT,
            n_query=ComparisonConfig.N_QUERY,
            device=self.device
        )
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 训练循环
        model.train()
        for episode in range(num_episodes):
            # 获取一个episode
            support_features, support_labels, query_features, query_labels = train_dataset[episode]
            
            # 前向传播
            predictions = model(support_features, support_labels, query_features, query_labels)
            
            # 计算损失
            loss = F.cross_entropy(predictions, query_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model
    
    def _extract_embeddings(self, model, features, batch_size: int = 128):
        """
        使用 Few-Shot/MetricLearning 模型的 backbone 提取逐样本 embedding（numpy）
        - MatchingNetwork / PrototypicalNetwork: model.backbone(x) -> (trunk, emb)
        - MetricLearningModel: model.backbone.trunk(x) -> trunk; embedder(trunk) -> emb
        """
        device = self.device
        if hasattr(features, 'values'):
            features = features.values
        X = torch.from_numpy(features).float()
        if device != 'cpu':
            X = X.to(device)

        model.eval()
        embs = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = X[i:i+batch_size]
                # 兼容三种模型
                if hasattr(model, "backbone") and hasattr(model.backbone, "embedder"):
                    if hasattr(model.backbone, "trunk"):
                        trunk_out = model.backbone.trunk(xb)
                        z = model.backbone.embedder(trunk_out)
                    else:
                        _, z = model.backbone(xb)  # fallback
                else:
                    _, z = model.backbone(xb)  # Matching/Proto 
                embs.append(z.detach().cpu())
        return torch.cat(embs, dim=0).numpy()

    def evaluate_few_shot_samplewise(self, model, clf_type: str = "logreg"):
        """
        基于 backbone embedding 的 sample-wise 线性评估：
        在训练集 embedding 上拟合LinearSVM，在测试集 embedding 上评估。
        适用于 MatchingNet / ProtoNet / MetricLearningModel.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score, f1_score

        train_embs = self._extract_embeddings(model, self.train_features)
        test_embs  = self._extract_embeddings(model, self.test_features)

        y_train = self.train_labels if not hasattr(self.train_labels, 'values') else self.train_labels.values
        y_test  = self.test_labels  if not hasattr(self.test_labels, 'values')  else self.test_labels.values

        if clf_type == "svm":
            clf = Pipeline([
                ("scaler", StandardScaler(with_mean=True)),
                ("clf", LinearSVC(C=1.0, random_state=ComparisonConfig.RANDOM_STATE))
            ])
        else:
            clf = Pipeline([
                ("scaler", StandardScaler(with_mean=True)),
                ("clf", LogisticRegression(
                    C=1.0, max_iter=1000, random_state=ComparisonConfig.RANDOM_STATE, multi_class='auto'
                ))
            ])

        clf.fit(train_embs, y_train)
        y_pred = clf.predict(test_embs)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted')
        return acc, f1
    def evaluate_samplewise_on_fold(self, model, 
                                    fold_train_features, fold_train_labels,
                                    fold_val_features, fold_val_labels,
                                    clf_type: str = "logreg"):
        """
        —— 折内（CV）sample-wise 评估：linear probe —— 
        1) 用本折训练好的 backbone 抽取 train/val 的 embedding
        2) 在 train-emb 上拟合 LogReg/LinearSVM
        3) 在 val-emb 上做逐样本预测 → 返回 (acc, f1)
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score, f1_score

        # to numpy
        Xtr = fold_train_features if not hasattr(fold_train_features, 'values') else fold_train_features.values
        Xva = fold_val_features   if not hasattr(fold_val_features, 'values')   else fold_val_features.values
        ytr = fold_train_labels if not hasattr(fold_train_labels, 'values') else fold_train_labels.values
        yva = fold_val_labels   if not hasattr(fold_val_labels, 'values')   else fold_val_labels.values

        # 1) 提取 embedding（使用本折训练后的 backbone）
        def _extract(model, X, bs=128):
            T = torch.from_numpy(X).float().to(self.device)
            embs = []
            model.eval()
            with torch.no_grad():
                for i in range(0, len(T), bs):
                    xb = T[i:i+bs]
                    # 兼容 Metric/Matching/Proto：都有 SharedBackbone
                    if hasattr(model, "backbone") and hasattr(model.backbone, "trunk"):
                        trunk = model.backbone.trunk(xb)
                        z = model.backbone.embedder(trunk)
                    else:
                        _, z = model.backbone(xb)
                    embs.append(z.detach().cpu())
            return torch.cat(embs, 0).numpy()

        tr_emb = _extract(model, Xtr)
        va_emb = _extract(model, Xva)

        # 2) 线性分类器
        if clf_type == "svm":
            clf = Pipeline([
                ("scaler", StandardScaler(with_mean=True)),
                ("clf", LinearSVC(C=1.0, random_state=ComparisonConfig.RANDOM_STATE))
            ])
        else:
            clf = Pipeline([
                ("scaler", StandardScaler(with_mean=True)),
                ("clf", LogisticRegression(
                    C=1.0, max_iter=1000, random_state=ComparisonConfig.RANDOM_STATE, multi_class='auto'
                ))
            ])

        # 3) 拟合并在验证集评估（无泄漏）
        clf.fit(tr_emb, ytr)
        y_pred = clf.predict(va_emb)
        acc = accuracy_score(yva, y_pred)
        f1  = f1_score(yva, y_pred, average='weighted')
        return acc, f1


    def _evaluate_metric_fold_direct(self, models, Xtr, ytr, Xva, yva):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score, f1_score
        
        # 提取训练集 embedding
        train_embs = self._extract_embeddings_from_models(models, Xtr)
        val_embs = self._extract_embeddings_from_models(models, Xva)
        
        # 检查 embedding
        print(f"\n[DEBUG] Embedding 检查:")
        print(f"  训练集 embedding shape: {train_embs.shape}")
        print(f"  验证集 embedding shape: {val_embs.shape}")
        print(f"  训练集 embedding 范围: [{train_embs.min():.4f}, {train_embs.max():.4f}]")
        print(f"  训练集 embedding 是否全零: {np.allclose(train_embs, 0)}")
        print(f"  训练集 embedding 是否有 NaN: {np.isnan(train_embs).any()}")
        print(f"  训练集 embedding 是否有 Inf: {np.isinf(train_embs).any()}")
        
        # 转换标签
        ytr_np = ytr if not hasattr(ytr, 'values') else ytr.values
        yva_np = yva if not hasattr(yva, 'values') else yva.values
        
        print(f"  训练集标签分布: {np.bincount(ytr_np)}")
        print(f"  验证集标签分布: {np.bincount(yva_np)}")
        
        # 训练线性分类器
        clf = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(
                C=1.0, max_iter=1000, 
                random_state=ComparisonConfig.RANDOM_STATE, 
                multi_class='auto'
            ))
        ])
        
        try:
            clf.fit(train_embs, ytr_np)
            y_pred = clf.predict(val_embs)
            
            acc = accuracy_score(yva_np, y_pred)
            f1 = f1_score(yva_np, y_pred, average='weighted')
            
            print(f"  验证集预测分布: {np.bincount(y_pred)}")
            print(f"  准确率: {acc:.4f}, F1: {f1:.4f}")
            
            return acc, f1
        except Exception as e:
            print(f"[ERROR] 线性分类器训练失败: {e}")
            return 0.0, 0.0

    def _extract_embeddings_from_models(self, models, X, batch_size=128):
        """
        从度量学习的 models dict 中提取 embedding
        """
        device = self.device
        if hasattr(X, 'values'):
            X = X.values
        
        X_tensor = torch.from_numpy(X).float().to(device)
        
        # 设置为评估模式
        models['trunk'].eval()
        models['embedder'].eval()
        
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                xb = X_tensor[i:i+batch_size]
                trunk_out = models['trunk'](xb)
                emb = models['embedder'](trunk_out)
                embeddings.append(emb.cpu())
        result = torch.cat(embeddings, dim=0).numpy()
        
        # 检查
        if np.isnan(result).any() or np.isinf(result).any():
            print(f"[ERROR] Embedding 包含 NaN 或 Inf!")
            return np.zeros_like(result)
        
        if np.allclose(result, 0):
            print(f"[WARNING] Embedding 全为零!")
        
        return torch.cat(embeddings, dim=0).numpy()
    def objective(self, trial):
        """优化目标函数 - 使用正确的五折交叉验证"""
        # 使用固定随机种子确保可重现性
        set_all_seeds(ComparisonConfig.RANDOM_STATE)
        
        try:
            # 🔥 正确使用五折交叉验证
            cv_scores = []
            cv_f1_scores = []
            skf = ComparisonConfig.get_cv_splitter()
            
            print(f"\n开始 {ComparisonConfig.N_SPLITS} 折交叉验证...")
            
            # 对每一折进行训练和评估
            for fold, (train_idx, val_idx) in enumerate(skf.split(self.train_features, self.train_labels)):
                print(f"  正在处理第 {fold + 1}/{ComparisonConfig.N_SPLITS} 折...")
                # Xtr, ytr, Xva, yva = self._prepare_fold_data(train_idx, val_idx)
                # 创建模型
                model = self.create_model(trial)
                fold_processor = DataProcessor(self.base_processor.vocabulary)
                df_train = self.train_df.iloc[train_idx]
                df_val = self.train_df.iloc[val_idx]
                X_train = fold_processor.extract_features(df_train, fit_transform=True)
                X_val = fold_processor.extract_features(df_val, fit_transform=False)
                y_train = df_train["label"].to_numpy(np.int64)
                y_val = df_val["label"].to_numpy(np.int64)
                # 根据模型类型训练
                if self.model_type == "metric_learning":
                    # 训练度量学习（得到 dict: {'trunk', 'embedder', ...}）
                    trained_models = self.train_metric_learning_fold(model, trial, X_train, y_train)


                    # 用“已更新权重”的 model 做 linear-probe 的折内评估
                    fold_score, fold_f1 = self._evaluate_metric_fold_direct(
                        trained_models, X_train, y_train, X_val, y_val
                    )

                else:
                    # Few-shot (Matching/Proto): 训练返回的就是“已更新”的模型
                    trained_model = self.train_few_shot_fold(model, trial, X_train, y_train)

                    # 直接用训练后的模型评估（sample-wise linear-probe）
                    fold_score, fold_f1 = self.evaluate_samplewise_on_fold(
                        trained_model, X_train, y_train, X_val, y_val, clf_type="logreg"
                    )
                cv_scores.append(fold_score)
                cv_f1_scores.append(fold_f1)
                print(f"    第 {fold + 1} 折准确率: {fold_score:.4f}")
                print(f"    第 {fold + 1} 折F1: {fold_f1:.4f}")
                
                # 每折后进行剪枝判断
                trial.report(fold_score, fold)
                if trial.should_prune():
                    print(f"    试验在第 {fold + 1} 折被剪枝")
                    raise optuna.TrialPruned()
            
            # 计算平均交叉验证分数
            avg_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            trial.set_user_attr('fold_scores', cv_scores)
            trial.set_user_attr('fold_f1_scores', cv_f1_scores)
            print(f"  交叉验证结果: {avg_score:.4f} ± {std_score:.4f}")
            print(f"  各折分数: {[f'{score:.4f}' for score in cv_scores]}")
            print(f"  各折F1: {[f'{score:.4f}' for score in cv_f1_scores]}")
            
            return avg_score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"试验失败: {e}")
            return 0.0

    def train_metric_learning_fold(self, model, trial, train_features, train_labels):
        """为单个fold训练度量学习模型"""
        # 训练参数
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_int('batch_size', 4, 16)
        num_epochs = trial.suggest_int('num_epochs', 20, 60)  # 减少epoch以便快速交叉验证
        
        # 创建数据集
        train_dataset = MedicalDataset(train_features, train_labels, device=self.device)
        
        # 分离模型组件
        models = {
            "trunk": model.backbone.trunk,
            "embedder": model.backbone.embedder,
            "classifier": model.classifier
        }
        
        # 创建优化器
        optimizers = {
            "trunk_optimizer": torch.optim.Adam(
                model.backbone.trunk.parameters(), lr=learning_rate, weight_decay=weight_decay
            ),
            "embedder_optimizer": torch.optim.Adam(
                model.backbone.embedder.parameters(), lr=learning_rate, weight_decay=weight_decay
            ),
            "classifier_optimizer": torch.optim.Adam(
                model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        }
        
        # 简化的损失函数和挖掘器（为了快速交叉验证）
        loss_funcs = {
            "metric_loss": losses.TripletMarginLoss(margin=0.2),
            "classifier_loss": nn.CrossEntropyLoss()
        }
        
        mining_funcs = {"tuple_miner": miners.TripletMarginMiner()}
        loss_weights = {"metric_loss": 1.0, "classifier_loss": 0.5}
        
        # 创建训练器
        trainer = trainers.TrainWithClassifier(
            models=models,
            optimizers=optimizers,
            batch_size=batch_size,
            loss_funcs=loss_funcs,
            dataset=train_dataset,
            mining_funcs=mining_funcs,
            dataloader_num_workers=0,
            loss_weights=loss_weights
        )
        
        # 训练
        trainer.train(num_epochs=num_epochs)
        
        return models
    
    def evaluate_metric_learning_fold(self, models, val_features, val_labels):
        """评估单个fold的度量学习模型，返回 (accuracy, f1)"""
        val_dataset = MedicalDataset(val_features, val_labels, device=self.device)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

        for m in models.values():
            m.eval()

        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                z = models['embedder'](models['trunk'](x))
                logits = models['classifier'](z)
                pred = logits.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_true.extend(y.cpu().numpy())

        acc = float((np.array(all_preds) == np.array(all_true)).mean()) if all_true else 0.0
        f1 = f1_score(all_true, all_preds, average='weighted') if all_true else 0.0
        return acc, f1
    
    def evaluate_few_shot_fold(self, model, val_features, val_labels):
        """评估单个fold的Few-Shot Learning模型"""
        # 创建验证数据集
        val_dataset = FewShotDataset(
            val_features, val_labels,
            n_way=ComparisonConfig.N_WAY,
            n_support=ComparisonConfig.N_SUPPORT,
            n_query=ComparisonConfig.N_QUERY,
            device=self.device
        )
        
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            # 测试多个episode
            for episode in range(200):  # 验证时使用较少episode
                support_features, support_labels, query_features, query_labels = val_dataset[episode]
                
                predictions = model(support_features, support_labels, query_features, query_labels)
                _, predicted = torch.max(predictions.data, 1)
                
                total += query_labels.size(0)
                correct += (predicted == query_labels).sum().item()
                
                # 收集预测结果用于F1计算
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(query_labels.cpu().numpy())
        
        accuracy = correct / total
        f1 = f1_score(all_true_labels, all_predictions, average='weighted')
        return accuracy, f1

    def train_final_model(self, best_params):
        """使用最佳参数在完整训练集上训练最终模型"""
        print(f"\n使用最佳参数训练最终 {self.model_type} 模型...")
        
        # 设置随机种子
        set_all_seeds(ComparisonConfig.RANDOM_STATE)
        
        if self.model_type == 'metric_learning':
            return self._train_final_metric_learning(best_params)
        else:
            return self._train_final_few_shot(best_params)
    
    def _train_final_metric_learning(self, best_params):
        """训练最终的度量学习模型 - 手动训练循环"""
        print(f"\n使用最佳参数训练最终度量学习模型...")
        # 创建模型
        model = MetricLearningModel(
            input_size=self.input_size,
            hidden_size=best_params['hidden_size'],
            embedding_size=best_params['embedding_size'],
            num_classes=self.num_classes,
            dropout_rate=best_params['dropout_rate'],
            device=self.device
        )
        
        # 创建数据集和加载器
        train_dataset = MedicalDataset(self.train_features, self.train_labels, device=self.device)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=best_params.get('batch_size'), 
            shuffle=True,
            drop_last=len(train_dataset) > best_params['batch_size']
        )
        
        # 优化器
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=best_params['learning_rate'], 
            weight_decay=best_params['weight_decay']
        )
        
        # 损失函数
        metric_loss_fn = losses.TripletMarginLoss(margin=0.2)
        class_loss_fn = nn.CrossEntropyLoss()
        miner = miners.TripletMarginMiner()
        
        # 手动训练循环
        model.train()
        num_epochs = best_params['num_epochs']
        
        print(f"训练 {num_epochs} 个epochs...")
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 前向传播
                trunk_output, embeddings, logits = model(batch_features)
                
                # 计算损失
                hard_pairs = miner(embeddings, batch_labels)
                metric_loss = metric_loss_fn(embeddings, batch_labels, hard_pairs)
                class_loss = class_loss_fn(logits, batch_labels)
                total_loss = metric_loss + 0.5 * class_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = np.mean(epoch_losses)
                print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}")
        
        print("训练完成！")
        
        # 设置为eval模式并返回原始nn.Module
        model.eval()
        
        models = {
            "trunk": model.backbone.trunk,
            "embedder": model.backbone.embedder,
            "classifier": model.classifier
        }
        
        # 确保所有组件都是eval模式
        for m in models.values():
            m.eval()
        
        return models
    
    def _train_final_few_shot(self, best_params):
        """训练最终的Few-Shot Learning模型"""
        # 创建模型
        if self.model_type == 'matching_network':
            model = MatchingNetwork(
                input_size=self.input_size,
                hidden_size=best_params['hidden_size'],
                embedding_size=best_params['embedding_size'],
                num_classes=self.num_classes,
                dropout_rate=best_params['dropout_rate'],
                use_fce=best_params['use_fce'],
                lstm_hidden_size=best_params.get('lstm_hidden_size', 32),
                device=self.device
            )
        else:  # prototypical_network
            model = PrototypicalNetwork(
                input_size=self.input_size,
                hidden_size=best_params['hidden_size'],
                embedding_size=best_params['embedding_size'],
                num_classes=self.num_classes,
                dropout_rate=best_params['dropout_rate'],
                device=self.device
            )
        
        # 创建Few-Shot数据集
        train_dataset = FewShotDataset(
            self.train_features, self.train_labels,
            n_way=ComparisonConfig.N_WAY,
            n_support=ComparisonConfig.N_SUPPORT,
            n_query=ComparisonConfig.N_QUERY,
            device=self.device
        )
        
        # 创建优化器
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=best_params['learning_rate'], 
            weight_decay=best_params['weight_decay']
        )
        
        # 训练循环
        model.train()
        num_episodes = best_params.get('num_episodes', 200)
        
        print(f"训练 {num_episodes} 个episodes...")
        for episode in range(num_episodes):
            support_features, support_labels, query_features, query_labels = train_dataset[episode]
            
            predictions = model(support_features, support_labels, query_features, query_labels)
            loss = F.cross_entropy(predictions, query_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (episode + 1) % 50 == 0:
                print(f"  Episode {episode + 1}/{num_episodes}, Loss: {loss.item():.4f}")
        
        return model
    
    def train_few_shot(self, best_params):
        """使用最佳参数训练Few-Shot Learning模型"""
        # 创建模型
        model = self.create_model_with_params(best_params)
        
        # 训练参数
        learning_rate = best_params['learning_rate']
        weight_decay = best_params['weight_decay']
        num_episodes = best_params['num_episodes']
        
        # 创建Few-Shot数据集
        train_dataset = FewShotDataset(
            self.train_features, self.train_labels,
            n_way=ComparisonConfig.N_WAY,
            n_support=ComparisonConfig.N_SUPPORT,
            n_query=ComparisonConfig.N_QUERY,
            device=self.device
        )
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 训练循环
        model.train()
        for episode in range(num_episodes):
            # 获取一个episode
            support_features, support_labels, query_features, query_labels = train_dataset[episode]
            
            # 前向传播
            predictions = model(support_features, support_labels, query_features, query_labels)
            
            # 计算损失
            loss = F.cross_entropy(predictions, query_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model
    
    def train_metric_learning(self, best_params):
        """使用最佳参数训练度量学习模型"""
        # 创建模型
        model = self.create_model_with_params(best_params)
        
        # 训练参数
        learning_rate = best_params['learning_rate']
        weight_decay = best_params['weight_decay']
        batch_size = best_params['batch_size']
        num_epochs = best_params['num_epochs']
        
        # 创建数据集和数据加载器
        train_dataset = MedicalDataset(self.train_features, self.train_labels, device=self.device)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=len(train_dataset) > batch_size
        )
        
        # 优化器
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 损失函数
        metric_loss_fn = losses.TripletMarginLoss(margin=0.2)
        class_loss_fn = nn.CrossEntropyLoss()
        miner = miners.TripletMarginMiner()
        
        # 训练循环
        model.train()
        for epoch in range(num_epochs):
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 前向传播
                trunk_output, embeddings, logits = model(batch_features)
                
                # 计算损失
                hard_pairs = miner(embeddings, batch_labels)
                metric_loss = metric_loss_fn(embeddings, batch_labels, hard_pairs)
                class_loss = class_loss_fn(logits, batch_labels)
                total_loss = metric_loss + 0.5 * class_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        
        # 设置为eval模式
        model.eval()
        
        # 返回原始的nn.Module组件，而不是字典
        models = {
            "trunk": model.backbone.trunk,
            "embedder": model.backbone.embedder,
            "classifier": model.classifier
        }
        
        return models

    def evaluate_metric_learning(self, models):
        """评估度量学习模型"""
        test_dataset = MedicalDataset(self.test_features, self.test_labels, device=self.device)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        for model in models.values():
            model.eval()
        
        correct = 0
        total = 0
        
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                trunk_output = models['trunk'](batch_features)
                embeddings = models['embedder'](trunk_output)
                logits = models['classifier'](embeddings)
                
                _, predicted = torch.max(logits.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(batch_labels.cpu().numpy())
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        f1 = f1_score(all_true, all_preds, average='weighted') if len(all_true) > 0 else 0.0
        return correct / total, f1
    
    def evaluate_few_shot(self, model):
        """评估Few-Shot Learning模型"""
        # 创建测试数据集
        test_dataset = FewShotDataset(
            self.test_features, self.test_labels,
            n_way=ComparisonConfig.N_WAY,
            n_support=ComparisonConfig.N_SUPPORT,
            n_query=ComparisonConfig.N_QUERY,
            device=self.device
        )
        
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for episode in range(min(100, len(test_dataset))):  # 限制测试episodes数量
                support_features, support_labels, query_features, query_labels = test_dataset[episode]
                
                predictions = model(support_features, support_labels, query_features, query_labels)
                _, predicted = torch.max(predictions.data, 1)
                
                total += query_labels.size(0)
                correct += (predicted == query_labels).sum().item()
                
                # 收集预测结果用于F1计算
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(query_labels.cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0.0
        f1 = f1_score(all_true_labels, all_predictions, average='weighted') if len(all_true_labels) > 0 else 0.0
        return accuracy, f1
    
    def create_model_with_params(self, params):
        """使用给定参数创建模型"""
        if self.model_type == 'metric_learning':
            return MetricLearningModel(
                input_size=self.input_size,
                hidden_size=params['hidden_size'],
                embedding_size=params['embedding_size'],
                num_classes=self.num_classes,
                dropout_rate=params['dropout_rate'],
                device=self.device
            )
        
        elif self.model_type == 'matching_network':
            return MatchingNetwork(
                input_size=self.input_size,
                hidden_size=params['hidden_size'],
                embedding_size=params['embedding_size'],
                num_classes=self.num_classes,
                dropout_rate=params['dropout_rate'],
                use_fce=params['use_fce'],
                lstm_hidden_size=params.get('lstm_hidden_size', 32),
                device=self.device
            )
        
        elif self.model_type == 'prototypical_network':
            return PrototypicalNetwork(
                input_size=self.input_size,
                hidden_size=params['hidden_size'],
                embedding_size=params['embedding_size'],
                num_classes=self.num_classes,
                dropout_rate=params['dropout_rate'],
                device=self.device
            )
    
    def optimize(self):
        """执行贝叶斯优化"""
        print(f"开始优化 {self.model_type}...")
        
        # 创建研究
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=ComparisonConfig.N_STARTUP_TRIALS),
            sampler=TPESampler(seed=ComparisonConfig.RANDOM_STATE),
            study_name=f'{self.model_type}_optimization'
        )
        
        # 执行优化
        study.optimize(
            self.objective,
            n_trials=ComparisonConfig.N_TRIALS,
            show_progress_bar=True
        )
        
        return study

def save_comparison_results(results, studies):
    """保存对比结果 - 参数保存为JSON，其他结果保存为CSV"""
    print("\n💾 保存对比结果...")
    
    import json
    import pandas as pd
    
    method_names = {
        # Few-Shot Learning方法
        'metric_learning': '度量学习',
        'matching_network': '匹配网络', 
        'prototypical_network': '原型网络',
        # 传统机器学习方法
        'svm': '支持向量机',
        'random_forest': '随机森林',
        'decision_tree': '决策树',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM'
    }
    
    # 1. 保存最佳参数为JSON文件（每个模型单独保存）
    print("\n保存各模型最佳参数(JSON格式)...")
    for method, result in results.items():
        if 'error' not in result and result['best_params']:
            filename = f'best_params_{method}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result['best_params'], f, indent=2, ensure_ascii=False)
            print(f"  {method_names.get(method, method)} 参数已保存到: {filename}")
    
    # 2. 创建结果汇总CSV表
    print("\n保存结果汇总(CSV格式)...")
    summary_data = []
    for method, result in results.items():
        if 'error' not in result:
            summary_data.append({
                '方法': method_names.get(method, method),
                '方法类型': result.get('method_type', 'unknown'),
                '交叉验证最佳准确率': result['best_accuracy'],
                '交叉验证准确率': result.get('cv_fold_accuracies', []),
                "交叉验证准确率均值": result.get('cv_mean', 0.0),
                '交叉验证F1': result.get('cv_fold_f1_scores', []),
                '交叉验证F1均值': result.get('cv_f1_mean', 0.0),
                '准确率标准差': result.get('cv_std', 0.0),
                'F1标准差': result.get('cv_f1_std', 0.0),
                '测试集准确率': result.get('test_accuracy', None),
                '过拟合差距': result.get('overfitting_gap', None),
                '试验次数': result['n_trials']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('交叉验证最佳准确率', ascending=False)
    summary_df.to_csv('model_performance_summary.csv', index=False, encoding='utf-8-sig')
    print("结果汇总已保存到: model_performance_summary.csv")
    
    # 3. 创建交叉验证详细分数CSV表
    print("\n保存交叉验证详细分数(CSV格式)...")
    cv_detailed_data = []
    for method, result in results.items():
        if 'error' not in result and result.get('cv_fold_accuracies'):
            fold_accuracies = result['cv_fold_accuracies']
            fold_f1_scores = result.get('cv_fold_f1_scores', [0] * len(fold_accuracies))
            
            for fold_idx, (acc, f1) in enumerate(zip(fold_accuracies, fold_f1_scores)):
                cv_detailed_data.append({
                    '方法': method_names.get(method, method),
                    '方法类型': result.get('method_type', 'unknown'),
                    '折数': fold_idx + 1,
                    '准确率': acc,
                    'F1分数': f1
                })
    
    cv_detailed_df = pd.DataFrame(cv_detailed_data)
    cv_detailed_df.to_csv('cv_fold_scores.csv', index=False, encoding='utf-8-sig')
    print("交叉验证详细分数已保存到: cv_fold_scores.csv")
    
    # 4. 创建实验配置CSV
    print("\n保存实验配置(CSV格式)...")
    config_data = [{
        '配置项': '试验次数',
        '值': ComparisonConfig.N_TRIALS
    }, {
        '配置项': '交叉验证折数',
        '值': ComparisonConfig.N_SPLITS
    }, {
        '配置项': 'Few-Shot N-Way',
        '值': ComparisonConfig.N_WAY
    }, {
        '配置项': 'Few-Shot N-Support',
        '值': ComparisonConfig.N_SUPPORT
    }, {
        '配置项': 'Few-Shot N-Query',
        '值': ComparisonConfig.N_QUERY
    }, {
        '配置项': '设备',
        '值': ComparisonConfig.DEVICE
    }, {
        '配置项': '随机种子',
        '值': ComparisonConfig.RANDOM_STATE
    }]
    
    config_df = pd.DataFrame(config_data)
    config_df.to_csv('experiment_config.csv', index=False, encoding='utf-8-sig')
    print("实验配置已保存到: experiment_config.csv")
    
    # 5. 保存最佳方法信息为CSV
    print("\n   保存最佳方法信息(CSV格式)...")
    best_method = max(results.keys(), key=lambda x: results[x]['best_accuracy'])
    best_info_data = [{
        '最佳方法': method_names.get(best_method, best_method),
        '最佳准确率': results[best_method]['best_accuracy'],
        '最佳F1': results[best_method].get('cv_f1_mean', 0.0),
        '方法类型': results[best_method].get('method_type', 'unknown')
    }]
    
    best_info_df = pd.DataFrame(best_info_data)
    best_info_df.to_csv('best_method_info.csv', index=False, encoding='utf-8-sig')
    print("最佳方法信息已保存到: best_method_info.csv")
    
    # 6. 保存方法排名为CSV
    print("\n  保存方法排名(CSV格式)...")
    ranking_data = []
    sorted_methods = sorted(results.keys(), key=lambda x: results[x]['best_accuracy'], reverse=True)
    for rank, method in enumerate(sorted_methods, 1):
        if 'error' not in results[method]:
            ranking_data.append({
                '排名': rank,
                '方法': method_names.get(method, method),
                '准确率': results[method]['best_accuracy'],
                'F1分数': results[method].get('cv_f1_mean', 0.0),
                '方法类型': results[method].get('method_type', 'unknown')
            })
    
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df.to_csv('method_ranking.csv', index=False, encoding='utf-8-sig')
    print("方法排名已保存到: method_ranking.csv")
    
    print(f"\n🏆 最佳方法: {method_names.get(best_method, best_method)}")
    print("="*60)
    print("保存的文件列表:")
    print("JSON文件 (最佳参数):")
    for method in results.keys():
        if 'error' not in results[method]:
            print(f"  - best_params_{method}.json")
    print("\nCSV文件 (其他结果):")
    print("  - model_performance_summary.csv (性能汇总)")
    print("  - cv_fold_scores.csv (交叉验证详细分数)")
    print("  - experiment_config.csv (实验配置)")
    print("  - best_method_info.csv (最佳方法信息)")
    print("  - method_ranking.csv (方法排名)")

"""运行所有方法的对比实验"""
print("="*80)

# 设置全局随机种子
set_all_seeds(ComparisonConfig.RANDOM_STATE)

# 加载数据
print("\n加载数据...")
train_data = pd.read_csv(ComparisonConfig.TRAIN_DATA_PATH)
test_data = pd.read_csv(ComparisonConfig.TEST_DATA_PATH)

print(f"训练数据: {len(train_data)} 样本")
print(f"测试数据: {len(test_data)} 样本")

# 数据预处理
print("\n数据预处理...")
processor = DataProcessor(ComparisonConfig.VOCABULARY)

# 提取特征
train_features = processor.extract_features(train_data, fit_transform=True)
test_features = processor.extract_features(test_data, fit_transform=False)

# 提取标签
train_labels = train_data["label"].to_numpy().astype(np.int64)
test_labels = test_data["label"].to_numpy().astype(np.int64)

print(f"特征维度: {train_features.shape[1]}")
print(f"类别数量: {len(np.unique(train_labels))}")

# 存储结果
results = {}
studies = {}

few_shot_methods = ['metric_learning', 'matching_network', 'prototypical_network']
traditional_ml_methods = ['svm', 'random_forest', 'decision_tree', 'xgboost', 'lightgbm']

method_names = {
    # Few-Shot Learning方法
    'metric_learning': '度量学习',
    'matching_network': '匹配网络', 
    'prototypical_network': '原型网络',
    # 传统机器学习方法
    'svm': '支持向量机',
    'random_forest': '随机森林',
    'decision_tree': '决策树',
    'xgboost': 'XGBoost',
    'lightgbm': 'LightGBM'
}

# 先优化Few-Shot Learning方法
print("\n" + "="*80)
print("第一阶段：Few-Shot Learning方法优化")
print("="*80)

for method in few_shot_methods:
    print(f"\n{'='*60}")
    print(f"开始优化: {method_names[method]}")
    print(f"{'='*60}")
    
    try:
        # 创建Few-Shot Learning训练器
        trainer = UnifiedTrainer(method, train_features, train_labels, test_features, test_labels, train_data, processor)
        
        # 执行优化
        study = trainer.optimize()
        best_params = study.best_params
        final_model = trainer.train_final_model(best_params)
        fold_scores = study.best_trial.user_attrs.get('fold_scores', None)
        fold_f1_scores = study.best_trial.user_attrs.get('fold_f1_scores', None)
        class_names = ['L', 'M', 'S']
        inspector = PerClassInspector(
                    trainer=trainer,
                    trained=final_model,
                    class_names=class_names,
                    model_name=trainer.model_type,
                    random_state=0,
                    )
        inspector.run_all(
                    save_dir=f'figs/inspect/{trainer.model_type}',
                    distance_metric='cosine', # 或 'euclidean'
                    embed_vis='umap', # 无 umap 时自动回退 t-SNE
                    do_matching_heatmap=True, # 仅 matching_network 时有效
                    episode_way=3, episode_support=2, episode_query=2,
                    topk_list=[1, 3, 5],
                    )
        results[method] = {
            'best_accuracy': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'cv_fold_accuracies': fold_scores,
            'cv_mean': float(np.mean(fold_scores)) if fold_scores else None,
            'cv_std': float(np.std(fold_scores)) if fold_scores else None,
            'cv_fold_f1_scores': fold_f1_scores,
            'cv_f1_mean': float(np.mean(fold_f1_scores)) if fold_f1_scores else None,
            'cv_f1_std': float(np.std(fold_f1_scores)) if fold_f1_scores else None,
            'method_type': 'few_shot_learning'
        }
        studies[method] = study
        
        print(f"\n{method_names[method]} 完成:")
        print(f"  交叉验证准确率: {study.best_value:.4f}")
        print(f"  试验次数: {len(study.trials)}")
        
    except Exception as e:
        print(f"\n{method_names[method]} 失败: {str(e)}")
        print(f"  错误类型: {type(e).__name__}")
        # 记录失败的方法
        results[method] = {
            'best_accuracy': 0.0,
            'best_params': {},
            'n_trials': 0,
            'cv_fold_accuracies': None,
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'cv_fold_f1_scores': None,
            'cv_f1_mean': 0.0,
            'cv_f1_std': 0.0,
            'method_type': 'few_shot_learning',
            'error': str(e)
        }
        continue

# 再优化传统机器学习方法
print("\n" + "="*80)
print("第二阶段：传统机器学习方法优化")
print("="*80)

# 创建传统机器学习训练器
traditional_trainer = TraditionalMLTrainer(train_features, train_labels, test_features, test_labels, train_df=train_data, processor=processor)

for method in traditional_ml_methods:
    print(f"\n{'='*60}")
    print(f"开始优化: {method_names[method]}")
    print(f"{'='*60}")
    
    try:
        # 执行优化
        study = traditional_trainer.optimize(method)
        best_params = study.best_params
        final_model = traditional_trainer.train_final_model(method, best_params)
        fold_scores = study.best_trial.user_attrs.get('fold_scores', None)
        fold_f1_scores = study.best_trial.user_attrs.get('fold_f1_scores', None)
        
        results[method] = {
            'best_accuracy': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'cv_fold_accuracies': fold_scores,
            'cv_mean': float(np.mean(fold_scores)) if fold_scores else None,
            'cv_std': float(np.std(fold_scores)) if fold_scores else None,
            'cv_fold_f1_scores': fold_f1_scores,
            'cv_f1_mean': float(np.mean(fold_f1_scores)) if fold_f1_scores else None,
            'cv_f1_std': float(np.std(fold_f1_scores)) if fold_f1_scores else None,
            'method_type': 'traditional_ml'
        }
        studies[method] = study
        
        print(f"\n{method_names[method]} 完成:")
        print(f"  交叉验证准确率: {study.best_value:.4f}")
        print(f"  试验次数: {len(study.trials)}")
        
    except Exception as e:
        print(f"\n {method_names[method]} 失败: {str(e)}")
        print(f"  错误类型: {type(e).__name__}")
        # 记录失败的方法
        results[method] = {
            'best_accuracy': 0.0,
            'best_params': {},
            'n_trials': 0,
            'cv_fold_accuracies': None,
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'cv_fold_f1_scores': None,
            'cv_f1_mean': 0.0,
            'cv_f1_std': 0.0,
            'method_type': 'traditional_ml',
            'error': str(e)
        }
        continue

# 保存所有结果
print("\n" + "="*80)
print("实验总结")
print("="*80)

# 按准确率排序显示结果
sorted_methods = sorted(results.keys(), key=lambda x: results[x]['best_accuracy'], reverse=True)

print("\n所有方法性能排名:")
for i, method in enumerate(sorted_methods, 1):
    result = results[method]
    method_display_name = method_names.get(method, str(method))  # 确保不为None
    cv_std = result.get('cv_std', 0.0)
    if cv_std is None:
        cv_std = 0.0
        
    print(f"{i:2d}. {method_display_name:12s} - "
            f"准确率: {result['best_accuracy']:.4f} ± {cv_std:.4f} "
            f"({result['method_type']})")

# 新增：测试集评估以检测过拟合
print("\n" + "="*80)
print("测试集评估")


test_results = {}

# 评估Few-Shot Learning方法
for method in few_shot_methods:
    if method not in results or 'error' in results[method]:
        continue
        
    print(f"\n评估 {method_names[method]} 在测试集上的性能...")
    try:
        # 使用最佳参数重新训练模型
        trainer = UnifiedTrainer(method, train_features, train_labels, test_features, test_labels, train_data, processor)
        if method == 'metric_learning':
            models = trainer.train_metric_learning(results[method]['best_params'])
            probe_model = MetricLearningModel(
                input_size=trainer.input_size,
                hidden_size=results[method]['best_params']['hidden_size'],
                embedding_size=results[method]['best_params']['embedding_size'],
                num_classes=trainer.num_classes,
                dropout_rate=results[method]['best_params']['dropout_rate'],
                device=trainer.device
            )
            # 把训练好的 trunk/embedder 权重拷回 probe_model（分类头是否同步不影响线性探针）
            # probe_model.backbone.trunk.load_state_dict(models['trunk'].state_dict())
            # probe_model.backbone.embedder.load_state_dict(models['embedder'].state_dict())
            trainer._load_metric_backbone_weights_(probe_model, models)
            sw_acc, sw_f1 = trainer.evaluate_few_shot_samplewise(probe_model, clf_type="logreg")
            results[method]['test_accuracy_sw'] = sw_acc
            results[method]['test_f1_sw'] = sw_f1
            results[method]['overfitting_gap_sw'] = results[method]['best_accuracy'] - sw_acc

            print(f"  (linear-probe)测试集: Acc={sw_acc:.4f}, F1={sw_f1:.4f}")

        else:
            # MatchingNet / ProtoNet 保持与上一条相同：episodic + sample-wise（线性探针）
            episodic_model = trainer.train_few_shot(results[method]['best_params'])
            epi_acc, epi_f1 = trainer.evaluate_few_shot(episodic_model)
            sw_acc, sw_f1   = trainer.evaluate_few_shot_samplewise(episodic_model, clf_type="logreg")

            results[method]['test_accuracy_ep'] = epi_acc
            results[method]['test_f1_ep'] = epi_f1
            results[method]['test_accuracy_sw'] = sw_acc
            results[method]['test_f1_sw'] = sw_f1
            results[method]['overfitting_gap_sw'] = results[method]['best_accuracy'] - sw_acc

            print(f"  (episodic)    测试集: Acc={epi_acc:.4f}, F1={epi_f1:.4f}")
            print(f"  (samplewise)  测试集: Acc={sw_acc:.4f}, F1={sw_f1:.4f}")

    except Exception as e:
        print(f"  评估失败: {e}")
        continue

# 评估传统机器学习方法
for method in traditional_ml_methods:
    if method not in results or 'error' in results[method]:
        continue
        
    print(f"\n评估 {method_names[method]} 在测试集上的性能...")
    try:
        # 使用最佳参数重新训练模型
        best_params = results[method]['best_params']
        final_model = traditional_trainer.train_final_model(method, best_params)
        
        # 在测试集上评估
        test_predictions = final_model.predict(test_features)
        test_accuracy = accuracy_score(test_labels, test_predictions)
        test_f1 = f1_score(test_labels, test_predictions, average='weighted')
        
        results[method]['test_accuracy'] = test_accuracy
        results[method]['test_f1'] = test_f1
        cv_accuracy = results[method]['best_accuracy']
        cv_f1 = results[method]['cv_f1_mean']
        overfitting_gap = cv_accuracy - test_accuracy
        
        print(f"  交叉验证准确率: {cv_accuracy:.4f}")
        print(f"  测试集准确率:   {test_accuracy:.4f}")
        print(f"  交叉验证F1:     {cv_f1:.4f}")
        print(f"  测试集F1:       {test_f1:.4f}")
        print(f"  性能差距:       {overfitting_gap:.4f}")
        
            
    except Exception as e:
        print(f"  测试集评估失败: {str(e)}")
        results[method]['test_accuracy'] = 0.0
        results[method]['test_f1'] = 0.0



save_comparison_results(results, studies)