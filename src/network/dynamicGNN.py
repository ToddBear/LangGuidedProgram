import numpy as np
import torch
import torch.nn as nn
from helper import fc_block, Constant, LayerNormGRUCell

class DynamicGNN_Memory_Module(nn.Module):

    def __init__(self, graph_state_dim, n_node_state, n_touch, n_edge, embedding_dim, update_mode='Residual', num_actions=30, self_action_emb=False, self_state_emb=False):
        super(DynamicGNN_Memory_Module, self).__init__()

        # State Change Events
        # state_legal_embedding = nn.Embedding(2, embedding_dim)
        self.self_action_emb = self_action_emb
        self.self_state_emb = self_state_emb
        
        if self.self_action_emb:
            self.action_embedding = nn.Embedding(num_actions, embedding_dim)
        if self.self_state_emb:
            self.state2emb = fc_block(n_node_state, embedding_dim, False, nn.Tanh)

        state2hidden = nn.Sequential()
        state2hidden.add_module('fc1', fc_block(embedding_dim, graph_state_dim, False, nn.Tanh))
        state2hidden.add_module('fc2', fc_block(graph_state_dim, graph_state_dim, False, nn.Tanh))
        # self.state_legal_embedding = state_legal_embedding
        self.state2hidden = state2hidden
        state_message_creator = nn.Sequential()
        state_message_creator.add_module('fc1', fc_block(2 * graph_state_dim, graph_state_dim, False, nn.Tanh))
        state_message_creator.add_module('fc2', fc_block(graph_state_dim, graph_state_dim, False, nn.Tanh))
        self.state_message_creator = state_message_creator

        # Edge Change Events
        edge_embedding = nn.Embedding(n_edge, embedding_dim)
        edge2hidden = nn.Sequential()
        edge2hidden.add_module('fc1', fc_block(embedding_dim+graph_state_dim, graph_state_dim, False, nn.Tanh))
        edge2hidden.add_module('fc2', fc_block(graph_state_dim, graph_state_dim, False, nn.Tanh))
        self.edge_embedding = edge_embedding
        self.edge2hidden = edge2hidden
        edge_message_creator = nn.Sequential()
        edge_message_creator.add_module('fc1', fc_block(2 * graph_state_dim, graph_state_dim, False, nn.Tanh))
        edge_message_creator.add_module('fc2', fc_block(graph_state_dim, graph_state_dim, False, nn.Tanh))
        self.edge_message_creator = edge_message_creator

        # Interaction Events
        action2hidden = nn.Sequential()
        action2hidden.add_module('fc1', fc_block(embedding_dim + n_touch, graph_state_dim, False, nn.Tanh))
        action2hidden.add_module('fc2', fc_block(graph_state_dim, graph_state_dim, False, nn.Tanh))
        self.action2hidden = action2hidden
        interaction_message_creator = nn.Sequential()
        interaction_message_creator.add_module('fc1', fc_block(2 * graph_state_dim, graph_state_dim, False, nn.Tanh))
        interaction_message_creator.add_module('fc2', fc_block(graph_state_dim, graph_state_dim, False, nn.Tanh))
        self.interaction_message_creator = interaction_message_creator

        self.update_mode = update_mode
        self.graph_hidden = graph_state_dim

    def forward(self, memory_hx, action_embedding, state2emb, touch_info, state_info, edge_info, state_update=True, edge_update=True, action_input=None):
        # Input:
        ## memory_hx : (B, N, C) : store the temporal node state
        ## source_matrix : (E, B, N, N) : the adjacent matrix of previous timestep
        ## target_matrix : (E, B, N, N) : the adjacent matrix of current tiemstep
        ## source_node_states : B list of (N, n_state) : the state of previous timestep
        ## target_node_states : B list of (N, n_state) : the state of current timestep
        ## action_embedding: (B, C) : the predicted action of current timestep
        ## batch_touch_idx: (B, N, 3) : the touch type of objects (character, object, subject or none)
        ## batch_touch_mask: (B, N): the tourch mask of current interaction (only character and interacted objects are labeled as 1)
        ## timestep : const value : indicate the timestep of current interaction 
        B = memory_hx.shape[0]
        N = memory_hx.shape[1]
        C = memory_hx.shape[2]

        batch_touch_idx, batch_touch_mask = touch_info
        batch_state_idx, batch_state_legal, batch_state_mask = state_info
        batch_edge_idx, batch_edge_type, batch_edge_mask = edge_info

        # Return:
        ## updated node states : (B, N, C)
        # message_collector = torch.zeros(B, N, C).cuda()
        # message_counter = torch.zeros(B, N).cuda()
        # message_mask = torch.zeros(B, N).cuda()
        tensor_device = memory_hx.device

        updated_memory_hx = memory_hx

        if state_update:
            # state change messgae
            if not self.self_state_emb:
                state_embedding = state2emb(batch_state_idx.float())
            else:
                state_embedding = self.state2emb(batch_state_idx.float())
            graph_input = state_embedding
            graph_input = self.state2hidden(graph_input)
            graph_input = graph_input.view(B * N, -1)
            state_message = self.state_message_creator(torch.cat([graph_input, memory_hx.view(B * N, -1)], 1)).view(B, N, self.graph_hidden)
            updated_memory_hx = updated_memory_hx + state_message*batch_state_mask[:,:,None].repeat(1, 1, self.graph_hidden)

        if edge_update:
            # edge change message
            edge_type_embedding = self.edge_embedding(batch_edge_type.long())
            other_node_state = torch.gather(memory_hx, dim=1, index=batch_edge_idx[:,:,None].repeat(1, 1, C).long())
            graph_input = torch.cat([edge_type_embedding, other_node_state], 2)
            graph_input = self.edge2hidden(graph_input)
            graph_input = graph_input.view(B * N, -1)
            edge_message = self.edge_message_creator(torch.cat([graph_input, memory_hx.view(B * N, -1)], 1)).view(B, N, self.graph_hidden)
            updated_memory_hx = updated_memory_hx + edge_message*batch_edge_mask[:,:,None].repeat(1, 1, self.graph_hidden)

        # interaction message
        B, N, _ = batch_touch_idx.size()
        if not self.self_action_emb:
            action_embedding = action_embedding.unsqueeze(1).repeat(1, N, 1)
        else:
            action_embedding = self.action_embedding(action_input).unsqueeze(1).repeat(1, N, 1)
        graph_input = torch.cat([action_embedding, batch_touch_idx], 2)
        graph_input = self.action2hidden(graph_input)
        graph_input = graph_input.view(B * N, -1)
        interaction_message = self.interaction_message_creator(torch.cat([graph_input, memory_hx.view(B * N, -1)], 1)).view(B, N, self.graph_hidden)
        updated_memory_hx = updated_memory_hx + interaction_message*batch_touch_mask[:,:,None].repeat(1, 1, self.graph_hidden)

        updated_memory_hx = updated_memory_hx.view(B, N, -1)

        return updated_memory_hx