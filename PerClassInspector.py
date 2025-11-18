import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from typing import Optional, Dict, Any, Tuple, List

# UMAP
try:
    import umap
    _UMAP_AVAILABLE = True
except Exception:
    _UMAP_AVAILABLE = False


class PerClassInspector:
    """
      1) 混淆矩阵（线性探针）
      2) 类内/类间距离直方图
      3) 到各类“原型中心”的距离热图
      4) 嵌入二维可视化（UMAP / t-SNE）
      5) Top-k 逐类准确率条形图
    """

    def __init__(
        self,
        trainer,
        trained,
        class_names: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        random_state: int = 0,
    ) -> None:
        self.trainer = trainer
        self.trained = trained  # 可能是 few-shot nn.Module，也可能是 {trunk, embedder}
        self.class_names = class_names
        self.model_name = model_name or getattr(trainer, 'model_type', 'model')
        self.random_state = random_state

        # 取训练/测试数据与标签
        self.X_tr = self._to_numpy(getattr(trainer, 'train_features'))
        self.y_tr = self._to_numpy_labels(getattr(trainer, 'train_labels'))
        self.X_te = self._to_numpy(getattr(trainer, 'test_features'))
        self.y_te = self._to_numpy_labels(getattr(trainer, 'test_labels'))

        # 预先计算嵌入
        self.Z_tr, self.Z_te = self._get_embeddings()

    def run_all(
        self,
        save_dir: Optional[str] = None,
        distance_metric: str = 'cosine',  # 'cosine' | 'euclidean'
        embed_vis: str = 'umap',          # 'umap' | 'tsne'
        do_matching_heatmap: bool = True,
        episode_way: int = 3,
        episode_support: int = 2,
        episode_query: int = 2,
        topk_list: List[int] = [1, 3, 5],
    ) -> Dict[str, Any]:
        """运行所有图表与报告。若 save_dir 给定，则将图片保存至该目录。"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 1) 线性探针：逐类报告 + 混淆矩阵
        y_pred, proba = self._linear_probe_and_report(save_dir=save_dir)

        # 2) 类内/类间 距离直方图
        self._plot_intra_inter_distance_hist(self.Z_te, self.y_te,
                                             bins=30,
                                             metric=distance_metric,
                                             save_path=self._maybe_path(save_dir, 'intra_inter_hist.png'))

        # 3) 原型距离热图
        self._plot_prototype_distance_heatmap(self.Z_te, self.y_te,
                                              metric='euclidean',
                                              title=f'{self.model_name}: distance to class prototypes',
                                              save_path=self._maybe_path(save_dir, 'prototype_distance_heatmap.png'))

        # 4) 嵌入二维可视化
        self._plot_embedding_2d(self.Z_te, self.y_te,
                                method=embed_vis,
                                title=f'{self.model_name} embeddings',
                                save_path=self._maybe_path(save_dir, f'embedding_{embed_vis}.png'))

        # 5) Top-k 逐类准确率
        if proba is not None:
            self._plot_topk_by_class(proba, self.y_te, k_list=topk_list,
                                     title=f'{self.model_name} Top-k per class',
                                     save_path=self._maybe_path(save_dir, 'topk_by_class.png'))

        # 6) MatchingNet 支持-查询相似度热图（仅在模型类型为 matching_network 时尝试）
        if do_matching_heatmap and getattr(self.trainer, 'model_type', '') == 'matching_network':
            self._plot_matching_similarity_heatmap_episode(
                episode_way, episode_support, episode_query,
                save_path=self._maybe_path(save_dir, 'matching_support_query_similarity.png')
            )

        return {
            'y_pred': y_pred,
            'proba': proba,
        }

    # ------------------------- 数据/嵌入 -------------------------
    def _to_numpy(self, x):
        if hasattr(x, 'values'):
            return x.values if getattr(x, 'ndim', 2) == 2 else x.values.reshape(-1, 1)
        return np.asarray(x)

    def _to_numpy_labels(self, y):
        if hasattr(y, 'values'):
            y = y.values
        y = np.asarray(y)
        return y.astype(int) if y.dtype.kind not in 'iu' else y

    def _get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """兼容 metric_learning（dict) 与 few-shot(nn.Module)。"""
        model = self.trained
        if isinstance(model, dict) and {'trunk', 'embedder'} <= set(model.keys()):
            # 复用 trainer._extract_embeddings
            class _Wrap:
                def __init__(self, trunk, embedder):
                    self.backbone = type('BB', (object,), {'trunk': trunk, 'embedder': embedder})()
                
                def eval(self):
                    """添加 eval() 方法以兼容 PyTorch 模型接口"""
                    # 确保所有组件都设置为 eval 模式
                    if hasattr(self.backbone.trunk, 'eval'):
                        self.backbone.trunk.eval()
                    if hasattr(self.backbone.embedder, 'eval'):
                        self.backbone.embedder.eval()
                    return self
            
            wrapper = _Wrap(model['trunk'], model['embedder'])
            Z_tr = self.trainer._extract_embeddings(wrapper, self.X_tr)
            Z_te = self.trainer._extract_embeddings(wrapper, self.X_te)
        else:
            Z_tr = self.trainer._extract_embeddings(model, self.X_tr)
            Z_te = self.trainer._extract_embeddings(model, self.X_te)
        return self._to_numpy(Z_tr), self._to_numpy(Z_te)

    # ------------------------- 1) 线性探针 + 混淆矩阵 -------------------------
    def _linear_probe_and_report(self, save_dir: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        clf = Pipeline([
            ('scaler', StandardScaler(with_mean=True)),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, multi_class='auto'))
        ])
        clf.fit(self.Z_tr, self.y_tr)
        y_pred = clf.predict(self.Z_te)
        try:
            proba = clf.predict_proba(self.Z_te)
        except Exception:
            proba = None

        # 文本报告
        report = classification_report(self.y_te, y_pred, target_names=self._labels_for_report(), digits=4)
        print(f"\n==== {self.model_name} | Linear Probe: per-class report ====\n{report}")

        # 混淆矩阵（行归一）
        self._plot_confmat(self.y_te, y_pred, labels=self._labels_for_ticks(),
                           title=f'{self.model_name} | Confusion Matrix (row-normalized)',
                           normalize='true',
                           save_path=self._maybe_path(save_dir, 'confmat.png'))
        return y_pred, proba

    def _labels_for_report(self):
        if self.class_names is None:
            uniq = sorted(np.unique(np.concatenate([self.y_tr, self.y_te])))
            return [str(u) for u in uniq]
        return list(self.class_names)

    def _labels_for_ticks(self):
        return self._labels_for_report()

    def _plot_confmat(self, y_true, y_pred, labels=None, title="Confusion Matrix",
                       normalize='true', save_path: Optional[str] = None):
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        im = ax.imshow(cm, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        if labels is not None:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha='center', va='center',
                        fontsize=8, color='white' if cm[i, j] > 0.5 else 'black')
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    # ------------------------- 2) 类内/类间距离直方图 -------------------------
    def _plot_intra_inter_distance_hist(self, embs, labels, bins=30, metric='cosine', save_path: Optional[str] = None):
        from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
        if metric == 'cosine':
            D = cosine_distances(embs, embs)
        else:
            D = euclidean_distances(embs, embs)
        y = labels.reshape(-1, 1)
        same = (y == y.T)
        diff = ~same
        iu = np.triu_indices_from(D, k=1)
        intra = D[iu][same[iu]]
        inter = D[iu][diff[iu]]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(intra, bins=bins, alpha=0.6, label='Intra-class', density=True)
        ax.hist(inter, bins=bins, alpha=0.6, label='Inter-class', density=True)
        ax.set_title(f'Embedding Distance ({metric})')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.legend()
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    # ------------------------- 3) 原型距离热图 -------------------------
    def _plot_prototype_distance_heatmap(self, embs, labels, metric='euclidean', title='Proto Distance Heatmap',
                                         save_path: Optional[str] = None):
        from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
        classes = np.unique(labels)
        protos = np.vstack([embs[labels == c].mean(axis=0) for c in classes])
        if metric == 'cosine':
            D = cosine_distances(embs, protos)
        else:
            D = euclidean_distances(embs, protos)
        order = np.argsort(labels)
        D_sorted = D[order]
        y_sorted = labels[order]

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(D_sorted, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel('Prototype (class)')
        ax.set_ylabel('Samples (sorted by true class)')
        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels([str(c) for c in classes])
        # 在 y 轴用水平分隔线标出类别边界
        for c in np.unique(y_sorted):
            idx = np.where(y_sorted == c)[0]
            ax.hlines([idx[-1] + 0.5], -0.5, len(classes) - 0.5, colors='w', linewidth=0.6)
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    # ------------------------- 4) 嵌入二维可视化 -------------------------
    def _plot_embedding_2d(self, embs, labels, method='umap', title='Embedding 2D', save_path: Optional[str] = None):
        if method == 'umap' and _UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=self.random_state)
            Z2 = reducer.fit_transform(embs)
            used = 'UMAP'
        else:
            perpl = int(np.clip(len(embs) // 10, 5, 30))
            Z2 = TSNE(n_components=2, random_state=self.random_state, init='pca', perplexity=perpl).fit_transform(embs)
            used = 't-SNE'
        classes = np.unique(labels)
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        
        # 定义更鲜明的颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        for i, c in enumerate(classes):
            idx = labels == c
            color = colors[i % len(colors)]
            # 实心样本点：移除alpha，增加点大小
            ax.scatter(Z2[idx, 0], Z2[idx, 1], s=25, c=color, label=f'class {c}', edgecolors='white', linewidth=0.5)
            # 优化类中心标记：更大的星形标记，白色边框
            ax.scatter(Z2[idx, 0].mean(), Z2[idx, 1].mean(), s=120, marker='*', 
                      c=color, edgecolors='white', linewidth=2, zorder=10)
        
        ax.set_title(f'{title} ({used})', fontsize=12, fontweight='bold')
        ax.legend(markerscale=1.2, fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left', 
                 borderaxespad=0., frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    # ------------------------- 5) Top-k 逐类准确率 -------------------------
    def _plot_topk_by_class(self, logits, y_true, k_list=[1, 3, 5], title='Top-k Accuracy per Class',
                             save_path: Optional[str] = None):
        N, C = logits.shape
        classes = np.unique(y_true)
        topk_hits = {k: [] for k in k_list}
        for c in classes:
            idx = np.where(y_true == c)[0]
            L = logits[idx]
            order = np.argsort(-L, axis=1)
            for k in k_list:
                hit = np.mean([(y_true[i] in order[r, :min(k, C)]) for r, i in enumerate(idx)])
                topk_hits[k].append(hit)
        fig, ax = plt.subplots(figsize=(6, 4))
        X = np.arange(len(classes))
        w = 0.8 / len(k_list)
        for j, k in enumerate(k_list):
            ax.bar(X + j * w - 0.4 + w / 2, topk_hits[k], width=w, label=f'Top-{k}')
        ax.set_xticks(X)
        ax.set_xticklabels([str(c) for c in classes])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    # ------------------------- 6) MatchingNet 相似度热图（按 episode 采样） -------------------------
    def _plot_matching_similarity_heatmap_episode(
        self,
        n_way: int = 3,
        n_support: int = 2,
        n_query: int = 2,
        save_path: Optional[str] = None,
    ) -> None:
        # 仅在 matching_network 模型类型下有效
        if getattr(self.trainer, 'model_type', '') != 'matching_network':
            return
        model = self.trained
        # 从训练集采样一个小 episode（为了稳定，也可传入固定索引）
        rng = np.random.default_rng(self.random_state)
        classes = np.unique(self.y_tr)
        n_way = min(n_way, len(classes))
        chosen = rng.choice(classes, n_way, replace=False)
        s_list, sl_list, q_list = [], [], []
        for c in chosen:
            idx = np.where(self.y_tr == c)[0]
            if len(idx) < n_support + n_query:
                continue
            sel = rng.choice(idx, n_support + n_query, replace=False)
            s_list.append(self.Z_tr[sel[:n_support]])
            sl_list.extend([c] * n_support)
            q_list.append(self.Z_tr[sel[n_support: n_support + n_query]])
        if not s_list or not q_list:
            return
        support_features = np.vstack(s_list)
        query_features = np.vstack(q_list)
        support_labels = np.array(sl_list)

        # 计算相似度（假设 model.backbone, model.distance_network 与你现有匹配网络一致）
        import torch
        model.eval()
        with torch.no_grad():
            # 由于我们已经是嵌入空间，这里直接用 numpy -> torch 并送入 distance_network
            device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
            s_emb = torch.from_numpy(support_features).float().to(device)
            q_emb = torch.from_numpy(query_features).float().to(device)
            sims = model.distance_network(s_emb, q_emb).detach().cpu().numpy()  # (n_query_total, n_support_total)

        # 排序 support 列（按标签）
        order = np.argsort(support_labels)
        sims = sims[:, order]
        sup_sorted = support_labels[order]

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(sims, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('MatchingNet: support-query cosine similarity')
        ax.set_xlabel('Support (sorted by class)')
        ax.set_ylabel('Query')
        # 画列分段刻度
        ticks, ticklabs = [], []
        for c in np.unique(sup_sorted):
            idx = np.where(sup_sorted == c)[0]
            ticks.append((idx[0] + idx[-1]) // 2)
            ticklabs.append(str(c))
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabs)
        plt.tight_layout()
        self._save_or_show(fig, save_path)

    # ------------------------- 工具：保存/显示 -------------------------
    def _maybe_path(self, save_dir: Optional[str], filename: str) -> Optional[str]:
        if save_dir is None:
            return None
        return os.path.join(save_dir, filename)

    def _save_or_show(self, fig, save_path: Optional[str]):
        if save_path:
            fig.savefig(save_path, dpi=180, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
