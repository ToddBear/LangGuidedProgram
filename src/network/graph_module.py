import torch
import torch.nn as nn

from program.graph_utils import *
from helper import fc_block, LayerNormGRUCell
import math
import os

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization 
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
           
            ('SCS'-mode is not in the paper but we found it works well in practice, 
              especially for GCN and GAT.)

            PairNorm is typically used after each graph convolution operation. 
        """
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
                
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x

class VanillaGraphEncoder(nn.Module):

    def __init__(
            self,
            n_timesteps,
            n_edge_types,
            graph_hidden,
            embedding_dim,
            hidden):

        super(VanillaGraphEncoder, self).__init__()

        layernorm = True
        self.n_timesteps = n_timesteps
        self.n_edge_types = n_edge_types
        self.embedding_dim = embedding_dim
        self.input_dim = n_edge_types + embedding_dim
        self.graph_hidden = graph_hidden

        node_init2hidden = nn.Sequential()
        node_init2hidden.add_module(
            'fc1',
            fc_block(
                3 * embedding_dim,
                graph_hidden,
                False,
                nn.Tanh))
        node_init2hidden.add_module(
            'fc2',
            fc_block(
                graph_hidden,
                graph_hidden,
                False,
                nn.Tanh))

        self.hidden2message_in_list = nn.ModuleList()
        self.hidden2message_out_list = nn.ModuleList()

        for i in range(n_edge_types):
            hidden2message_in_c = fc_block(
                graph_hidden, graph_hidden, False, nn.Tanh)
            self.add_module(
                "hidden2message_in_{}".format(i),
                hidden2message_in_c)
            self.hidden2message_in_list.append(hidden2message_in_c)

            hidden2message_out_c = fc_block(
                graph_hidden, graph_hidden, False, nn.Tanh)
            self.add_module(
                "hidden2message_out_{}".format(i),
                hidden2message_out_c)
            self.hidden2message_out_list.append(hidden2message_out_c)


        # if layernorm:
        #     self.gru_cell = LayerNormGRUCell
        # else:
        #     self.gru_cell = nn.GRUCell
        if layernorm:
            self.propagator = LayerNormGRUCell(
                input_size=2 * n_edge_types * graph_hidden,
                hidden_size=graph_hidden)
        else:
            self.propagator = nn.GRUCell(
                input_size=2 * n_edge_types * graph_hidden,
                hidden_size=graph_hidden)

        self.node_init2hidden = node_init2hidden
        # self.hidden2message_in = AttrProxy(self, "hidden2message_in_")
        # self.hidden2message_out = AttrProxy(self, "hidden2message_out_")

    def forward(
            self,
            edge_adjacency_matrix,
            node_state_prev,
            related_mask=None,
            score=None):
        """edge_adjacency_matrix: e, b, v, v
            object_state_arry: b, v, p
            state: b, v, h
        """

        B, V, H = node_state_prev.size()
        node_state_prev = node_state_prev.view(B * V, -1)
        node_state = node_state_prev

        edge_adjacency_matrix = edge_adjacency_matrix.float()
        edge_adjacency_matrix_out = edge_adjacency_matrix
        # convert the outgoing edges to incoming edges
        edge_adjacency_matrix_in = edge_adjacency_matrix.permute(0, 1, 3, 2)

        for i in range(self.n_timesteps):
            message_out = []
            for j in range(self.n_edge_types):
                node_state_hidden = self.hidden2message_out_list[j](node_state)    # b*v, h
                node_state_hidden = node_state_hidden.view(B, V, -1)
                if score is not None:
                    node_state_hidden = node_state_hidden * score[:,:,None]
                message_out.append(
                    torch.bmm(
                        edge_adjacency_matrix_out[j],
                        node_state_hidden))  # b, v, h

            # concatenate the message from each edges
            message_out = torch.stack(message_out, 2)       # b, v, e, h
            message_out = message_out.view(B * V, -1)         # b, v, e*h

            message_in = []
            for j in range(self.n_edge_types):
                node_state_hidden = self.hidden2message_in_list[j](
                    node_state)    # b*v, h
                node_state_hidden = node_state_hidden.view(B, V, -1)
                if score is not None:
                    node_state_hidden = node_state_hidden * score[:,:,None]
                message_in.append(
                    torch.bmm(
                        edge_adjacency_matrix_in[j],
                        node_state_hidden))

            # concatenate the message from each edges
            message_in = torch.stack(message_in, 2)         # b, v, e, h
            message_in = message_in.view(B * V, -1)           # b, v, e*h

            message = torch.cat([message_out, message_in], 1)
            node_state = self.propagator(message, node_state)

        if related_mask is not None:
            # mask out un-related changes
            related_mask_expand = related_mask.unsqueeze(
                2).repeat(1, 1, self.graph_hidden).float()
            related_mask_expand = related_mask_expand.view(B * V, -1)
            node_state = node_state * related_mask_expand + \
                node_state_prev * (-related_mask_expand + 1)

        node_state = node_state.view(B, V, -1)
        return node_state

class ResidualActionGraphEncoder(VanillaGraphEncoder):

    def __init__(
            self,
            n_edge_types,
            n_touch,
            graph_hidden,
            embedding_dim,
            hidden):

        super(
            ResidualActionGraphEncoder,
            self).__init__(
            0,
            n_edge_types,
            graph_hidden,
            embedding_dim,
            hidden)

        self.n_touch = n_touch

        action2hidden = nn.Sequential()
        action2hidden.add_module(
            'fc1',
            fc_block(
                embedding_dim + n_touch,
                graph_hidden,
                False,
                nn.Tanh))
        action2hidden.add_module(
            'fc2',
            fc_block(
                graph_hidden,
                graph_hidden,
                False,
                nn.Tanh))

        compute_residual = nn.Sequential()
        compute_residual.add_module(
            'fc1',
            fc_block(
                2 * graph_hidden,
                graph_hidden,
                False,
                nn.Tanh))
        compute_residual.add_module(
            'fc2',
            fc_block(
                graph_hidden,
                graph_hidden,
                False,
                nn.Tanh))
        self.compute_residual = compute_residual

        self.action2hidden = action2hidden

    def action_applier(
            self,
            action_embedding,
            batch_touch_idx,
            batch_node_state_prev,
            batch_touch_mask,
            return_residual=False):
        """
            action_embedding: b, emb
            batch_touch_idx: b, n, touch_type,
            batch_node_state_prev: b, n, h
            batch_touch_mask: b, n
        """

        B, N, _ = batch_touch_idx.size()

        action_embedding = action_embedding.unsqueeze(1).repeat(1, N, 1)
        graph_input = torch.cat([action_embedding, batch_touch_idx], 2)
        graph_input = self.action2hidden(graph_input)
        graph_input = graph_input.view(B * N, -1)

        batch_node_state_prev = batch_node_state_prev.view(B * N, -1)
        residual = self.compute_residual(
            torch.cat([graph_input, batch_node_state_prev], 1))

        batch_touch_mask = batch_touch_mask.unsqueeze(
            2).repeat(1, 1, self.graph_hidden)
        batch_touch_mask = batch_touch_mask.view(B * N, -1)

        batch_node_state = batch_node_state_prev + residual * batch_touch_mask
        batch_node_state = batch_node_state.view(B, N, -1)

        residual_batch_node_state = residual * batch_touch_mask
        residual_batch_node_state = residual_batch_node_state.view(B, N, -1)

        if not return_residual:
            return batch_node_state
        else:
            return batch_node_state, residual_batch_node_state

class SimpleHGNLayer(nn.Module):
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index
    edge_type_dim = 2  # position of the edge type in edge_index
    nodes_dim = 0  # node dimension/axis

    def __init__(self, n_edge_types, edge_hidden, in_hidden, out_hidden, nhead, feat_drop=0.0, attn_drop=0.5, negative_slope=0.2, residual=False, activation=None, alpha=0.0, task_hidden=64):
        super(SimpleHGNLayer, self).__init__()
        # store the paramter
        self.n_edge_types = n_edge_types
        self.edge_hidden = edge_hidden
        self.in_hidden = in_hidden
        self.out_hidden = out_hidden
        self.task_hidden = task_hidden
        self.nhead = nhead

        # define the networks by matrixes
        self.edge_emb = nn.Parameter(torch.zeros(size=(n_edge_types, edge_hidden)))
        self.W = nn.Parameter(torch.FloatTensor(in_hidden, out_hidden*nhead))
        self.W_e = nn.Parameter(torch.FloatTensor(edge_hidden, edge_hidden*nhead))
        self.a_l = nn.Parameter(torch.zeros(size=(1, nhead, out_hidden)))
        self.a_r = nn.Parameter(torch.zeros(size=(1, nhead, out_hidden)))
        self.a_e = nn.Parameter(torch.zeros(size=(1, nhead, edge_hidden)))
        
        self.W_t = nn.Parameter(torch.FloatTensor(task_hidden, task_hidden*nhead))
        self.a_t = nn.Parameter(torch.zeros(size=(1, nhead, task_hidden)))
        
        # define the dropout and activation module
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.act = activation
        if residual:
            self.residual = nn.Linear(in_hidden, out_hidden*nhead)
        else:
            self.residual = None
        self.reset_paramters()
        self.alpha = alpha
    
    def reset_paramters(self):
        def reset(tensor):
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)
        
        reset(self.edge_emb)
        reset(self.W)
        reset(self.W_e)
        reset(self.a_l)
        reset(self.a_r)
        reset(self.a_e)
        reset(self.W_t)
        reset(self.a_t)

    def forward(self, edge, x, res_attn=None, task_emb=None):
        # graph -> a tensor with shape (3, E)
        # x -> the hidden state of nodes (N, in_hidden)
        num_of_nodes = x.shape[self.nodes_dim]
        x = self.feat_drop(x) # input dropout
        h = torch.matmul(x, self.W).view(-1, self.nhead, self.out_hidden) # neighbor information projection
        h[torch.isnan(h)] = 0.0
        e = torch.matmul(self.edge_emb, self.W_e).view(-1, self.nhead, self.edge_hidden) # edge type embedding projection

        source_idx, target_idx = edge[self.src_nodes_dim,:], edge[self.trg_nodes_dim, :]
        edge_type = edge[self.edge_type_dim, :]
        # Self-attention on the nodes - Shared attention mechanism
        h_l = (self.a_l * h).sum(dim=-1)[source_idx]
        h_r = (self.a_r * h).sum(dim=-1)[target_idx]
        h_e = (self.a_e * e).sum(dim=-1)[edge_type]
        
        if task_emb is not None:
            task = torch.matmul(task_emb, self.W_t).view(-1, self.nhead, self.task_hidden) # neighbor information projection
            h_t = (self.a_t * task).sum(dim=-1)[source_idx]
            # print("h_t:{}".format(h_t.shape))
            # print("h_e:{}".format(h_e.shape))
            edge_attention = self.leakyrelu(h_l + h_r + h_e + h_t)
        else:
            edge_attention = self.leakyrelu(h_l + h_r + h_e)

        edge_attention = self.neighborhood_aware_softmax(edge_attention, target_idx, num_of_nodes)
        edge_attention = self.attn_drop(edge_attention)
        if res_attn is not None:
            edge_attention = edge_attention * (1 - self.alpha) + res_attn * self.alpha
        lifted_h = h.index_select(self.nodes_dim, source_idx)
        weighted_lifted_h = lifted_h * edge_attention
        out = self.aggregate_neighbors(weighted_lifted_h, edge, x, num_of_nodes)

        if self.residual is not None:
            res = self.residual(x)
            out += res
        if self.act is not None:
            out = self.act(out)

        return out, edge_attention.detach()

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)


class VanillaGraphEncoderGeo(nn.Module):
    # Implement from Simple-HGN - GRU Version

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index
    nodes_dim = 0  # node dimension/axis

    def __init__(
            self,
            n_timesteps,
            n_edge_types,
            graph_hidden,
            embedding_dim,
            residual=False,
            heads=[4,4],
            negative_slope=0.05,
            alpha=0.05,
            feat_drop=0.0,
            attn_drop=0.0,
            score_agg_mode='min',
            task_hidden=64):

        super(VanillaGraphEncoderGeo, self).__init__()
        import torch.nn.functional as F
        layernorm = True
        self.n_timesteps = n_timesteps
        self.n_edge_types = n_edge_types
        self.embedding_dim = embedding_dim
        self.input_dim = n_edge_types + embedding_dim
        self.graph_hidden = graph_hidden

        node_init2hidden = nn.Sequential()
        node_init2hidden.add_module('fc1', fc_block(3 * embedding_dim, graph_hidden, False, nn.Tanh))
        node_init2hidden.add_module('fc2', fc_block(graph_hidden, graph_hidden, False, nn.Tanh))
        self.node_init2hidden = node_init2hidden

        self.out_gat_layers = nn.ModuleList()
        self.in_gat_layers = nn.ModuleList()
        self.activation = F.elu
        for l in range(n_timesteps):
            c_residual = False if l==0 else residual
            self.out_gat_layers.append(SimpleHGNLayer(n_edge_types=n_edge_types, edge_hidden=graph_hidden, in_hidden=graph_hidden, out_hidden=graph_hidden//heads[l], 
                                       nhead=heads[l], feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=c_residual, activation=self.activation, alpha=alpha, task_hidden=task_hidden))
            self.in_gat_layers.append(SimpleHGNLayer(n_edge_types=n_edge_types, edge_hidden=graph_hidden, in_hidden=graph_hidden, out_hidden=graph_hidden//heads[l], 
                                       nhead=heads[l], feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=c_residual, activation=self.activation, alpha=alpha, task_hidden=task_hidden))
        if layernorm:
            self.gru_cell = LayerNormGRUCell
        else:
            self.gru_cell = nn.GRUCell

        propagator = self.gru_cell(input_size=2 * graph_hidden, hidden_size=graph_hidden)
        self.propagator = propagator
        self.score_agg_mode = score_agg_mode

    def forward(self, edge_idx_list, in_nodes_features, task_emb=None, return_weight=False):
        """
            edge_idx: (3,E) # from_id, to_id, edge_type
            in_nodes_features: (B, N, F_IN)
        """
        tensor_device = in_nodes_features.device
        B, N = in_nodes_features.shape[:-1]
        in_nodes_features = in_nodes_features.view(B*N, -1)
        if task_emb is not None:
            task_emb = task_emb.view(B*N, -1)
        node_state = in_nodes_features
        num_of_nodes = in_nodes_features.shape[0]
        
        res_attn_out = None
        res_attn_in = None

        from_id_list = edge_idx_list[0, :]
        to_id_list = edge_idx_list[1, :]
        edge_type_list = edge_idx_list[2, :]

        out_edge_idx_list = torch.stack((to_id_list, from_id_list, edge_type_list), dim=0)
        in_edge_idx_list = torch.stack((from_id_list, to_id_list, edge_type_list), dim=0)

        for i in range(self.n_timesteps):
            # out
            out_message, res_attn_out = self.out_gat_layers[i](out_edge_idx_list, node_state, res_attn=res_attn_out, task_emb=task_emb)
            in_message, res_attn_in = self.in_gat_layers[i](in_edge_idx_list, node_state, res_attn=res_attn_in, task_emb=task_emb)
            out_message = out_message.view(B*N, -1)
            in_message = in_message.view(B*N, -1)
            message = torch.cat([out_message, in_message], 1)
            node_state = self.propagator(message, node_state)

            if i==0:
                out_res_attn_out = res_attn_out
                out_res_attn_in = res_attn_in

        node_state = node_state.view(B, N, -1)

        if not return_weight:
            return node_state
        else:
            return node_state, res_attn_out, res_attn_in