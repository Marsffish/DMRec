# coding: utf-8


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph

class DMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(DMRec, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.build_item_graph = True
        self.v_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.degree_ratio = config['degree_ratio']
        self.co_topk = config['co_topk']
        self.cl_loss = config['cl_loss']
        self.adv_loss = config['adv_loss']
        self.ortho_loss = config['ortho_loss']
        self.cl_loss2 = config['cl_loss2']

        self.n_nodes = self.n_users + self.n_items
        self.softmax = nn.Softmax(dim=-1)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.norm_R = self.get_norm_interaction_matrix().to(self.device)
        # 提取行、列索引和数据
        row = self.interaction_matrix.row
        col = self.interaction_matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(self.interaction_matrix.data)
        # 将 SciPy 稀疏矩阵转换为 PyTorch 稀疏张量
        self.R = torch.sparse.FloatTensor(i, data, torch.Size(self.interaction_matrix.shape)).to(self.device)

        self.masked_adj, self.mm_adj = None, None

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}.pt'.format(self.knn_k))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}.pt'.format(self.knn_k))



        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.image_disentangle = FeatureDisentangle(self.v_feat.shape[1],self.feat_embed_dim)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.text_disentangle = FeatureDisentangle(self.t_feat.shape[1],self.feat_embed_dim)
            

        indices, self.image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
        indices, self.text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
        self.image_adj = self.image_adj.to(self.device)
        self.text_adj = self.text_adj.to(self.device)


        #构建项目共现图
        item_co_graph_file = os.path.join(dataset_path, 'item_co_graph_dict.pt')
        if os.path.exists(item_co_graph_file):
            self.item_co_graph = torch.load(item_co_graph_file)
        else:
            self.item_co_graph = self.get_items_co_graph(self.interaction_matrix)
            torch.save(self.item_co_graph, item_co_graph_file)

        #生成归一化的项目共现图并保存
        item_co_graph_norm_file = os.path.join(dataset_path, 'item_co_graph_norm_{}.pt'.format(self.co_topk))
        if os.path.exists(item_co_graph_norm_file):
            del self.item_co_graph
            self.item_co_graph_norm = torch.load(item_co_graph_norm_file)
            self.item_co_graph_norm = self.item_co_graph_norm.to(self.device)
        else:
            #将self.item_co_graph转化为密集矩阵
            self.item_co_graph_norm = self.item_co_graph.to_dense()
            topk_values, topk_indices = self.item_co_graph_norm.topk(self.co_topk, dim=1)
            self.item_co_graph_norm.fill_(0.)
            self.item_co_graph_norm.scatter_(1, topk_indices, topk_values)
            self.item_co_graph_norm = F.normalize(self.item_co_graph_norm, p=1, dim=1)
            self.item_co_graph_norm = self.item_co_graph_norm.to_sparse()
            torch.save(self.item_co_graph_norm, item_co_graph_norm_file)
            del self.item_co_graph



        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        self.query_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        self.query_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

    def save_content_side_embeddings(self, content_embeds, side_embeds, save_path):
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                'content_embeds': content_embeds.detach().cpu(),
                'side_embeds': side_embeds.detach().cpu(),
            },
            save_path,
        )


    def get_items_co_graph(self, train_interactions):
        # 将稀疏矩阵转换为稀疏张量
        train_interactions = torch.sparse.FloatTensor(
            torch.LongTensor([train_interactions.row, train_interactions.col]),
            torch.FloatTensor(train_interactions.data),
            torch.Size(train_interactions.shape)
        ).to(self.device)
        
        # 计算共现矩阵
        co_occurrence_graph = torch.sparse.mm(train_interactions.t(), train_interactions)
        
        # 去除对角线元素
        co_occurrence_graph = co_occurrence_graph.coalesce()
        values = co_occurrence_graph.values()
        indices = co_occurrence_graph.indices()
        mask = indices[0] != indices[1]
        new_indices = indices[:, mask]
        new_values = values[mask]
        co_occurrence_graph = torch.sparse.FloatTensor(new_indices, new_values, co_occurrence_graph.size())
        
        return co_occurrence_graph

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)
    
    def get_norm_interaction_matrix(self):
        # 将交互矩阵转换为 SciPy 稀疏矩阵
        inter_M = self.interaction_matrix

        # 计算行和列的和
        rowsum = np.array(inter_M.sum(1))
        colsum = np.array(inter_M.sum(0))

        # 计算行和列的逆平方根
        d_inv_row = np.power(rowsum, -0.5).flatten()
        d_inv_col = np.power(colsum, -0.5).flatten()

        # 处理无穷大的值
        d_inv_row[np.isinf(d_inv_row)] = 0.
        d_inv_col[np.isinf(d_inv_col)] = 0.

        # 构建对角矩阵
        D_row = sp.diags(d_inv_row)
        D_col = sp.diags(d_inv_col)

        # 归一化交互矩阵
        norm_interaction_matrix = D_row.dot(inter_M).dot(D_col)

        # 将归一化后的稀疏矩阵转换为 PyTorch 稀疏张量
        norm_interaction_matrix = sp.coo_matrix(norm_interaction_matrix)
        row = norm_interaction_matrix.row
        col = norm_interaction_matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(norm_interaction_matrix.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_users, self.n_items))).to(self.device)
        
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))
    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values
    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values    

    def create_normalized_adj_matrix(self, scores):
        # 将 scores 转换为稀疏张量
        scores = scores.coalesce()  # 确保 scores 是 coalesced 的稀疏张量

        # 获取用户和物品的数量
        n_users = self.n_users
        n_items = self.n_items

        # 创建一个大小为 (users + items) * (users + items) 的稀疏邻接矩阵
        indices = scores.indices()
        values = scores.values()

        # 创建邻接矩阵的索引和值
        row = torch.cat([indices[0], indices[1] + n_users])
        col = torch.cat([indices[1] + n_users, indices[0]])
        adj_indices = torch.stack([row, col])
        adj_values = torch.cat([values, values])

        # 创建稀疏邻接矩阵
        adj = torch.sparse.FloatTensor(adj_indices, adj_values, torch.Size([n_users + n_items, n_users + n_items])).to(self.device)

        # 计算度矩阵的逆平方根
        rowsum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv = torch.pow(rowsum, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.
        D_inv = torch.sparse.FloatTensor(torch.arange(n_users + n_items).repeat(2, 1).to(self.device), d_inv, torch.Size([n_users + n_items, n_users + n_items]))

        # 归一化邻接矩阵
        norm_adj_matrix = torch.sparse.mm(D_inv, torch.sparse.mm(adj, D_inv))

        return norm_adj_matrix
        
    def pre_epoch_processing(self):


        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
        else:
            degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
            random_idx = torch.randperm(self.edge_values.size(0))[:degree_len]
            # random sample
            keep_indices = self.edge_indices[:, random_idx]
            # norm values
            keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
            self.masked_R = torch.sparse.FloatTensor(keep_indices, keep_values, (self.n_users, self.n_items)).to(self.device)
            all_values = torch.cat((keep_values, keep_values))
            # update keep_indices to users/items+self.n_users
            keep_indices[1] += self.n_users
            all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
            self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)
        

    def forward(self, adj,v_adj, t_adj, train=False):
        #同构图增强

        hv_in,hv_out = self.image_disentangle(self.image_embedding.weight,self.item_id_embedding.weight)
        ht_in,ht_out = self.text_disentangle(self.text_embedding.weight,self.item_id_embedding.weight)
        
        for i in range(self.n_layers):
            hv_in = torch.sparse.mm(self.image_adj, hv_in)
            hv_out = torch.sparse.mm(self.image_adj, hv_out) 
            ht_in = torch.sparse.mm(self.text_adj, ht_in)
            ht_out = torch.sparse.mm(self.text_adj, ht_out)

        item_id_embedding = self.item_id_embedding.weight
        for i in range(self.n_layers):
            item_id_embedding = torch.sparse.mm(self.item_co_graph_norm, item_id_embedding)

        uhv_in = torch.sparse.mm(self.norm_R, hv_in)
        uhv_out = torch.sparse.mm(self.norm_R, hv_out)
        uht_in = torch.sparse.mm(self.norm_R, ht_in)
        uht_out = torch.sparse.mm(self.norm_R, ht_out)

        u_b, i_b ,content_embeds= self.adj_gcn(adj, self.user_embedding.weight, item_id_embedding)
        uhv_in, hv_in ,v_in= self.adj_gcn(v_adj, uhv_in, hv_in)
        uhv_out, hv_out ,v_out= self.adj_gcn(v_adj, uhv_out, hv_out)
        uht_in, ht_in ,t_in= self.adj_gcn(t_adj, uht_in, ht_in)
        uht_out, ht_out ,t_out= self.adj_gcn(t_adj, uht_out, ht_out)


        #   Modality-aware Preference Module
        att_common = torch.cat([self.query_v(F.normalize(v_in)), self.query_t(F.normalize(t_in))], dim=-1)
        weight_common = self.softmax(att_common)
        # mm_in = weight_common[:, 0].unsqueeze(dim=1) * v_in + weight_common[:, 1].unsqueeze(
        #     dim=1) * t_in
        mm_in = self.v_weight*F.normalize(v_in) + (1-self.v_weight)*F.normalize(t_in)
        
        agg_image_embeds = weight_common[:, 0].unsqueeze(dim=1) * F.normalize(v_out)
        agg_text_embeds = weight_common[:, 1].unsqueeze(dim=1) * F.normalize(t_out)
        

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
       
        agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds)
        agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds)

        mm_out = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds]), dim=0) 

        side_embeds = mm_in + mm_out

        all_embeds = content_embeds + side_embeds

        user_fnn, item_fnn = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)


        return user_fnn,item_fnn,v_in,t_in,v_out,t_out,content_embeds,side_embeds


    def adj_gcn(self, adj, users,items):
        ego_embeddings = torch.cat((users, items), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        users, items = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return users, items, all_embeddings
    
    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)
    
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss
    
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings ,v_in,t_in,v_out,t_out,content_embeds,side_embeds= self.forward(self.masked_adj,self.masked_adj,self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        uhv_in, hv_in = torch.split(v_in, [self.n_users, self.n_items], dim=0)
        uht_in, ht_in = torch.split(t_in, [self.n_users, self.n_items], dim=0)
        u_b, i_b = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        side_u, side_i = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)

        cl_loss1 = self.InfoNCE(self.user_embedding.weight[users], uhv_in[users], 0.2)
        cl_loss2 = self.InfoNCE(self.user_embedding.weight[users], uht_in[users], 0.2)
        cl_loss3 = self.InfoNCE(self.item_id_embedding.weight[pos_items], hv_in[pos_items], 0.2)
        cl_loss4 = self.InfoNCE(self.item_id_embedding.weight[pos_items], ht_in[pos_items], 0.2)

        cl_loss5 = self.InfoNCE(u_b[users], side_u[users], 0.2)
        cl_loss6 = self.InfoNCE(i_b[pos_items], side_i[pos_items], 0.2)

  

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,neg_i_g_embeddings)
        total_cl_loss = self.cl_loss*(cl_loss1 + cl_loss2 + cl_loss3 + cl_loss4 + cl_loss5 + cl_loss6)


        total_loss = batch_mf_loss + total_cl_loss 

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e,v_in,t_in,v_out,t_out,content_embeds,side_embeds= self.forward(self.norm_adj, self.norm_adj, self.norm_adj)

        #保存restore_item_e为npy文件
        np.save('./restore_item_e.npy', restore_item_e.detach().cpu().numpy())

        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))


        self.save_content_side_embeddings(content_embeds, side_embeds, './embeddings.pt')

        return scores
    


class FeatureDisentangle(nn.Module):
    def __init__(self, feat_dim,emb_dim):
        super().__init__()
        # 行为相关特征提取
        self.relevant_layer = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.Sigmoid()
        )
        
        # 行为无关特征提取
        self.irrelevant_layer = nn.Sequential(
            nn.Linear(feat_dim, emb_dim)
        )

    def forward(self, item_emb,item_id_embedding):
            
            relevant = self.relevant_layer(item_emb)
            relevant = torch.multiply(item_id_embedding, relevant)

            irrelevant = self.irrelevant_layer(item_emb)

            return relevant, irrelevant