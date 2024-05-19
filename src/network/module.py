from audioop import bias
import random
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import copy
from helper import fc_block, Constant, LayerNormGRUCell
from network.encoder_decoder import _sort_by_length, _sort_by_index
from network.utils import transfrom_format_into_sentence
import copy
from program.utils import single_step_evolve
import os
import pickle as pkl

def _initialize_node_embedding_with_graph(
        n_nodes,
        node_name_list,
        node_category_list,
        batch_node_states,
        object_embedding,
        category_embedding,
        object_dict,
        node_state2_hidden,
        to_cuda,
        return_node=True):

    N = max(n_nodes)
    B = len(n_nodes)

    # extract object name from node
    batch_object_list = np.zeros([B, N]).astype(np.int64)
    if return_node:
        for i, node_names in enumerate(node_name_list):
            for j, node in enumerate(node_names):
                name = node.split('.')[0]
                idx = int(object_dict.word2idx[name])
                batch_object_list[i, j] = idx

    batch_object_list = torch.tensor(batch_object_list, dtype=torch.int64)
    if to_cuda:
        batch_object_list = batch_object_list.cuda()

    batch_object_emb = object_embedding(batch_object_list.view(-1))
    batch_object_emb = batch_object_emb.view(B, N, -1)

    batch_category_list = torch.stack(node_category_list, 0)
    batch_category_emb = category_embedding(batch_category_list.view(-1).long())
    batch_category_emb = batch_category_emb.view(B, N, -1)

    batch_node_states = torch.stack(batch_node_states,0).float()        # b, n, n_states
    batch_node_states_emb = node_state2_hidden(batch_node_states.view(B * N, -1))
    batch_node_states_emb = batch_node_states_emb.view(B, N, -1)

    batch_node_emb = torch.cat([batch_category_emb, batch_object_emb, batch_node_states_emb], 2)
    return batch_node_emb, batch_object_list

class WordEncoder(nn.Module):

    def __init__(self, dset, embedding_dim, desc_hidden, word_embedding):
    
        super(WordEncoder, self).__init__()

        emb2hidden = nn.Sequential()
        emb2hidden.add_module('fc_block1', fc_block(embedding_dim, desc_hidden, True, nn.Tanh))
        emb2hidden.add_module('fc_block2', fc_block(desc_hidden, desc_hidden, True, nn.Tanh))
        self.emb2hidden = emb2hidden

        word_encoding = nn.GRU(desc_hidden, desc_hidden, 2)

        self.object_dict = dset.object_dict
        self.word_embedding = word_embedding
        self.word_encoding = word_encoding


    def forward(self, batch_data):

        batch_length, batch_words = batch_data
        embeddings = []
        for words in batch_words:

            word_emb = self.word_embedding(words)
            word_emb = self.emb2hidden(word_emb)
            embeddings.append(word_emb)

        rnn_input = pad_sequence(embeddings, batch_first=True)
        word_emb, _ = self.word_encoding(rnn_input)
        word_emb = torch.stack([word_emb[i, length-2]  for i, length in enumerate(batch_length)])

        return word_emb, embeddings

class ProgramEncoder(nn.Module):

    def __init__(self, dset, embedding_dim, hidden):
        super(ProgramEncoder, self).__init__()

        num_actions = dset.num_actions
        num_objects = dset.num_objects

        action_embedding = nn.Embedding(num_actions, embedding_dim)
        object_embedding = nn.Embedding(num_objects, embedding_dim)
        self.object_dict = dset.object_dict
        self.action_embedding = action_embedding
        self.object_embedding = object_embedding

class ProgramDecoder_LM(nn.Module):

    def __init__(
            self,
            dset,
            desc_hidden,
            embedding_dim,
            hidden,
            other_mode_hidden):
        super(ProgramDecoder_LM, self).__init__()

        num_actions = dset.num_actions
        num_objects = dset.num_objects

        hidden2object1 = nn.Sequential()
        hidden2object1.add_module('fc_block1', fc_block(hidden*2, hidden, False, nn.Tanh))
        hidden2object1.add_module('fc_block2', fc_block(hidden, num_objects, False, None))

        hidden2object2 = nn.Sequential()
        hidden2object2.add_module('fc_block1', fc_block(hidden + 2 * embedding_dim + 1*hidden, hidden, False, nn.Tanh))
        hidden2object2.add_module('fc_block2', fc_block(hidden, num_objects, False, None))

        node2hidden = nn.Sequential()
        node2hidden.add_module('fc_block1', fc_block(other_mode_hidden, hidden, False, nn.Tanh))
        node2hidden.add_module('fc_block2', fc_block(hidden, hidden, False, nn.Tanh))

        action_embedding = nn.Embedding(num_actions, embedding_dim)
        object_embedding = nn.Embedding(num_objects, embedding_dim)

        self.object_dict = dset.object_dict
        self.action_embedding = action_embedding
        self.object_embedding = object_embedding
        self.embedding_dim = embedding_dim
        self.hidden2object1 = hidden2object1
        self.hidden2object2 = hidden2object2
        self.node2hidden = node2hidden

class ProgramClassifierGeo(nn.Module):

    def __init__(
            self,
            dset,
            sketch_embedding_dim,
            embedding_dim,
            hidden,
            max_words,
            graph_state_dim):

        super(ProgramClassifierGeo, self).__init__()

        layernorm = True
        num_actions = dset.num_actions
        num_objects = dset.num_objects

        # Encode sketch embedding
        sketch_emb_encoder = nn.Sequential()
        sketch_emb_encoder.add_module('fc_block1', fc_block(sketch_embedding_dim, hidden, False, nn.Tanh))

        # Encode input words
        action_embedding = nn.Embedding(num_actions, embedding_dim)
        object_embedding = nn.Embedding(num_objects, embedding_dim)

        atomic_action2emb = nn.Sequential()
        atomic_action2emb.add_module('fc_block1', fc_block(embedding_dim + 2 * graph_state_dim, hidden, False, nn.Tanh))
        atomic_action2emb.add_module('fc_block2', fc_block(hidden, hidden, False, nn.Tanh))

        if layernorm:
            gru_cell = LayerNormGRUCell
        else:
            gru_cell = nn.GRUCell

        # gru
        decode_gru_cell = gru_cell(2 * hidden, hidden)

        # decode words
        gru2hidden = nn.Sequential()
        gru2hidden.add_module('fc_block1', fc_block(hidden, hidden, False, nn.Tanh))
        gru2hidden.add_module('fc_block2', fc_block(hidden, hidden, False, nn.Tanh))

        hidden2object1_logits = nn.Sequential()
        hidden2object1_logits.add_module('fc_block1', fc_block(hidden + graph_state_dim, hidden, False, nn.Tanh))
        hidden2object1_logits.add_module('fc_block2', fc_block(hidden, 1, False, None))

        hidden2action_logits = nn.Sequential()
        hidden2action_logits.add_module('fc_block1', fc_block(2*hidden + graph_state_dim, hidden, False, nn.Tanh))
        hidden2action_logits.add_module('fc_block2', fc_block(hidden, num_actions, False, None))

        hidden2object2_logits = nn.Sequential()
        hidden2object2_logits.add_module('fc_block1', fc_block(hidden + 2 * graph_state_dim + embedding_dim, hidden, False, nn.Tanh))
        hidden2object2_logits.add_module('fc_block2', fc_block(hidden, 1, False, None))

        self.embedding_dim = embedding_dim
        self.sketch_emb_encoder = sketch_emb_encoder
        self.action_embedding = action_embedding
        self.object_embedding = object_embedding
        self.atomic_action2emb = atomic_action2emb
        self.decode_gru_cell = decode_gru_cell
        self.gru2hidden = gru2hidden
        self.hidden2object1_logits = hidden2object1_logits
        self.hidden2action_logits = hidden2action_logits
        self.hidden2object2_logits = hidden2object2_logits
        self.num_actions = num_actions
        self.max_words = max_words

class ProgramGraphClassifierGeo(nn.Module):

    def __init__(
            self,
            large_model_name,
            dset,
            prob_env_graph_encoder,
            env_graph_encoder,
            env_graph_helper,
            sketch_embedding_dim,
            embedding_dim,
            hidden,
            max_words,
            interaction_grpah_helper=None,
            interaction_graph_encoder=None,
            lora_dropout=0.00):

        super(ProgramGraphClassifierGeo, self).__init__()
        torch.backends.cudnn.enabled = False

        from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
        from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
        from network.grammer_constraint import grammer_constraint
 
        # Graph Encoding
        self.interaction_grpah_helper = interaction_grpah_helper
        self.interaction_graph_encoder = interaction_graph_encoder
        env_graph2interactoin_graph = nn.Sequential()
        env_graph2interactoin_graph.add_module('fc1', fc_block(env_graph_encoder.graph_hidden, interaction_graph_encoder.graph_hidden, False, nn.Tanh))
        self.env_graph2interactoin_graph = env_graph2interactoin_graph
        graph_state_dim = interaction_graph_encoder.graph_hidden
        self.category_embedding = nn.Embedding(env_graph_helper.n_node_category, embedding_dim)
        self.node_state2_hidden = fc_block(env_graph_helper.n_node_state, embedding_dim, False,nn.Tanh)
        interact_graph_hidden = interaction_graph_encoder.graph_hidden
        env_init_graph2interaction_graph = nn.Sequential()
        env_init_graph2interaction_graph.add_module('fc1', fc_block(3 * embedding_dim, interact_graph_hidden, False, nn.Tanh))
        self.env_init_graph2interaction_graph = env_init_graph2interaction_graph
        
        # Graph-based Program Generation
        self.program_classifier = ProgramClassifierGeo(dset, sketch_embedding_dim, embedding_dim, hidden, max_words, graph_state_dim)

        # Adjacency Constraint Module
        hidden_emb_proj = nn.Sequential()
        hidden_emb_proj.add_module('fc1', fc_block(hidden*2, hidden, False, nn.Tanh))
        hidden_emb_proj.add_module('fc2', fc_block(hidden, hidden, False, nn.Tanh))
        self.hidden_emb_proj = hidden_emb_proj 
        link_predictor = nn.Sequential()
        link_predictor.add_module('fc1', fc_block(graph_state_dim+graph_state_dim+hidden, hidden, False, nn.Tanh))
        link_predictor.add_module('fc2', fc_block(hidden, 1, False, None))
        self.link_predictor = link_predictor
        self.agent_decode_gru_cell = LayerNormGRUCell(graph_state_dim, graph_state_dim)
        agent_gru2hidden = nn.Sequential()
        agent_gru2hidden.add_module('fc_block1', fc_block(graph_state_dim, graph_state_dim, False, nn.Tanh))
        agent_gru2hidden.add_module('fc_block2', fc_block(graph_state_dim, graph_state_dim, False, nn.Tanh))
        self.agent_gru2hidden = agent_gru2hidden

        # Language Guidance Module
        model_name = large_model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        gpt2 = GPT2LMHeadModel.from_pretrained(model_name, device_map={'':torch.cuda.current_device()})
        self.config = GPT2Config(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            gpt2.resize_token_embeddings(len(self.tokenizer))
        lora_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM)
        self.gpt2 = get_peft_model(gpt2, lora_config)
        self.context2hidden = nn.Sequential()
        self.context2hidden.add_module('fc1', fc_block(self.config.n_embd, hidden, False, nn.Tanh))
        self.program_decoder = ProgramDecoder_LM(dset, sketch_embedding_dim, embedding_dim, hidden, graph_state_dim)

        # Dynamic Graph Update Module
        from network.dynamicGNN import DynamicGNN_Memory_Module
        self.memory_unit = DynamicGNN_Memory_Module(graph_state_dim, env_graph_helper.n_node_state, interaction_grpah_helper.n_touch, interaction_grpah_helper.n_edge, embedding_dim, update_mode="Residual", num_actions=dset.num_actions, self_action_emb=True, self_state_emb=True)
        graph_hidden_emb_proj = nn.Sequential()
        graph_hidden_emb_proj.add_module('fc1', fc_block(hidden*2, hidden, False, nn.Tanh))
        graph_hidden_emb_proj.add_module('fc2', fc_block(hidden, hidden, False, nn.Tanh))
        self.graph_hidden_emb_proj = graph_hidden_emb_proj

        # MIMC
        self.grammer_constraint = grammer_constraint
        self.dset = dset
        self.graph_state_dim = graph_state_dim
        self.hidden = hidden
        self.action_object_constraints = dset.compatibility_matrix
        self.object_dict = dset.object_dict
        self.env_graph_helper = env_graph_helper
        self.env_graph_encoder = env_graph_encoder
        self.prob_env_graph_encoder = prob_env_graph_encoder

    def _pad_input(self, list_of_tensor):

        pad_tensor = []
        for tensor in list_of_tensor:
            tensor = pad_sequence(tensor, batch_first=True)   # B, T
            pad_tensor.append(tensor)

        return pad_tensor

    def _create_mask2d(self, B, N, length, value, to_cuda):
        mask = torch.zeros([B, N])
        if to_cuda:
            mask = mask.cuda()
        for i, l in enumerate(length):
            mask[i, l:] = value

        return mask

    def _decode_object(self, hidden, batch_node_mask, batch_node_state, hidden2logits, other_logits=None):

        _, N = batch_node_mask.size()
        hidden = hidden.unsqueeze(1).repeat(1, N, 1)
        logits = hidden2logits(torch.cat([hidden, batch_node_state], 2)).squeeze()
        logits = logits + batch_node_mask
        if other_logits is not None:
            logits = logits + other_logits
        pred = torch.argmax(logits, 1)

        return logits, pred


    def _calculate_object_loss(self, B, T, object1_logits_list, object2_logits_list, batch_object1, batch_object2):

        object1_loss = F.cross_entropy(object1_logits_list.view(B * T, -1), batch_object1.contiguous().view(-1), reduction='none').view(B, T)
        object2_loss = F.cross_entropy(object2_logits_list.view(B * T, -1), batch_object2.contiguous().view(-1), reduction='none').view(B, T)
        return object1_loss, object2_loss

    def _calcualte_action_loss(self, B, T, action_logits_list, batch_action):

        action_loss = F.cross_entropy(action_logits_list.view(B * T, -1), batch_action.contiguous().view(-1), reduction='none')
        action_loss = action_loss.view(B, T)
        return action_loss

    def _initial_input(self, B, initial_program, to_cuda):

        action, object1, object2 = initial_program
        action = torch.tensor([action for i in range(B)])
        object1 = torch.tensor([object1 for i in range(B)])
        object2 = torch.tensor([object2 for i in range(B)])
        if to_cuda:
            action = action.cuda()
            object1 = object1.cuda()
            object2 = object2.cuda()

        return action, object1, object2

    def matrix2idx(self, edge_adjacency_matrix):
        # adjacent_matrix : (E, B, N, N)
        E, B, N = edge_adjacency_matrix.shape[:-1]
        edge_type_list, batch_list, from_edge_list, to_edge_list = np.where(edge_adjacency_matrix.cpu().numpy() == 1)
        extended_from_edge_list = batch_list * N + from_edge_list
        extended_to_edge_list = batch_list * N + to_edge_list
        extended_edge = np.vstack((extended_from_edge_list, extended_to_edge_list))
        edge_collector_out = []
        unique_edge_idx_list = list(np.unique(edge_type_list))
        for e_i in range(E):
            if e_i not in unique_edge_idx_list:
                edge_collector_out.append(None)
                continue
            corresponding_idx_list = np.where(edge_type_list == e_i)[0]
            c_edge_tensor = torch.tensor(extended_edge[:, corresponding_idx_list], dtype=torch.long).cuda()
            edge_collector_out.append(c_edge_tensor)
        return edge_collector_out

    def matrix2idx_edge(self, edge_adjacency_matrix, tensor_device):
        # adjacent_matrix : (E, B, N, N)
        E, B, N = edge_adjacency_matrix.shape[:-1]
        edge_type_list, batch_list, from_edge_list, to_edge_list = np.where(edge_adjacency_matrix.cpu().numpy() == 1)
        extended_from_edge_list = batch_list * N + from_edge_list
        extended_to_edge_list = batch_list * N + to_edge_list
        extended_edge = np.vstack((extended_from_edge_list, extended_to_edge_list, edge_type_list))
        edge_collector_out = torch.tensor(extended_edge, dtype=torch.long).to(tensor_device)

        return edge_collector_out

    def _obtain_gt_mask(self, B, N, to_cuda, batch_object1, batch_object2):
        mask = torch.zeros([B, N])

        if to_cuda:
            mask = mask.cuda()

        for i, object1_list in enumerate(batch_object1):
            for object1 in object1_list:
                mask[i, object1] = 1

        for i, object2_list in enumerate(batch_object2):
            for object2 in object2_list:
                mask[i, object2] = 1

        return mask

    def create_action_sent(self, idx, action_NL):
        new_sent = []
        for i in range(len(action_NL)):
            new_c_sent = "Step {}:{}".format(idx, action_NL[i])
            new_sent.append(new_c_sent)

        return new_sent

    def matrix2idx_edge(self, edge_adjacency_matrix, tensor_device):
        # adjacent_matrix : (E, B, N, N)
        E, B, N = edge_adjacency_matrix.shape[:-1]
        edge_type_list, batch_list, from_edge_list, to_edge_list = np.where(edge_adjacency_matrix.cpu().numpy() == 1)
        extended_from_edge_list = batch_list * N + from_edge_list
        extended_to_edge_list = batch_list * N + to_edge_list
        extended_edge = np.vstack((extended_from_edge_list, extended_to_edge_list, edge_type_list))
        edge_collector_out = torch.tensor(extended_edge, dtype=torch.long).to(tensor_device)

        return edge_collector_out

    def _calculate_accuracy(self, action_correct, object1_correct, object2_correct, batch_length, info, graph=False):
        '''
            action_correct: torch tensor with shape (B, T)
        '''

        action_valid_correct = [sum(action_correct[i, :(l - 1)]) for i, l in enumerate(batch_length)]
        object1_valid_correct = [sum(object1_correct[i, :(l - 1)]) for i, l in enumerate(batch_length)]
        object2_valid_correct = [sum(object2_correct[i, :(l - 1)]) for i, l in enumerate(batch_length)]

        action_accuracy = sum(action_valid_correct).float() / (sum(batch_length) - 1. * len(batch_length))
        object1_accuracy = sum(object1_valid_correct).float() / (sum(batch_length) - 1. * len(batch_length))
        object2_accuracy = sum(object2_valid_correct).float() / (sum(batch_length) - 1. * len(batch_length))

        if not graph:
            info.update({'action_accuracy': action_accuracy})
            info.update({'object1_accuracy': object1_accuracy})
            info.update({'object2_accuracy': object2_accuracy})
        else:
            info.update({'action_graph_accuracy': action_accuracy})
            info.update({'object1_graph_accuracy': object1_accuracy})
            info.update({'object2_graph_accuracy': object2_accuracy})


    def _calculate_loss(self, action_loss, object1_loss, object2_loss, batch_length, info, graph=False):
        '''
            action_loss: torch tensor with shape (B, T)
        '''
        action_valid_loss = [sum(action_loss[i, :(l - 1)]) for i, l in enumerate(batch_length)]
        object1_valid_loss = [sum(object1_loss[i, :(l - 1)]) for i, l in enumerate(batch_length)]
        object2_valid_loss = [sum(object2_loss[i, :(l - 1)]) for i, l in enumerate(batch_length)]

        if not graph:
            info.update({"action_loss_per_program": [loss / (l - 1) for loss, l in zip(action_valid_loss, batch_length)]})
            info.update({"object1_loss_per_program": [loss / (l - 1) for loss, l in zip(object1_valid_loss, batch_length)]})
            info.update({"object2_loss_per_program": [loss / (l - 1) for loss, l in zip(object2_valid_loss, batch_length)]})
        else:
            info.update({"action_graph_loss_per_program": [loss / (l - 1) for loss, l in zip(action_valid_loss, batch_length)]})
            info.update({"object1_graph_loss_per_program": [loss / (l - 1) for loss, l in zip(object1_valid_loss, batch_length)]})
            info.update({"object2_graph_loss_per_program": [loss / (l - 1) for loss, l in zip(object2_valid_loss, batch_length)]})

        action_valid_loss = sum(action_valid_loss) / (sum(batch_length) - 1. * len(batch_length))
        object1_valid_loss = sum(object1_valid_loss) / (sum(batch_length) - 1. * len(batch_length))
        object2_valid_loss = sum(object2_valid_loss) / (sum(batch_length) - 1. * len(batch_length))

        loss = action_valid_loss + object1_valid_loss + object2_valid_loss

        if not graph:
            info.update({"action_loss": action_valid_loss})
            info.update({"object1_loss": object1_valid_loss})
            info.update({"object2_loss": object2_valid_loss})
            info.update({'total_loss': loss})
        else:
            info.update({"action_graph_loss": action_valid_loss})
            info.update({"object1_graph_loss": object1_valid_loss})
            info.update({"object2_graph_loss": object2_valid_loss})
            info.update({'total_graph_loss': loss})

        return loss

    def forward(self, inference, data, input_sent, desc_emb, graph, character_index_list, target_topology_collector, **kwargs):

        '''
            E: number of edge tpyes
            B: batch size
            N: maximum number of nodes 
            H: hidden size
            EMB: embedding size
            A: action space
            O: object space
            T: maximum time steps
        '''

        ### Unpack variables
        batch_length, batch_action, batch_object1, batch_object2 = data
        batch_adjacency_matrix, batch_node_name_list, batch_node_category_list, batch_node_states_, init_graph, final_graph = graph

        id2idx_list = []
        for batch_node_name in batch_node_name_list:
            id2idx = {int(name.split(".")[1]):idx for idx,name in enumerate(batch_node_name[:-2])}
            id2idx_list.append(id2idx)

        batch_action, batch_object1, batch_object2 = self._pad_input([batch_action, batch_object1, batch_object2, ])
        desc_emb = self.program_classifier.sketch_emb_encoder(desc_emb)

        float_tensor_type = desc_emb.dtype
        long_tensor_type = batch_action.dtype
        tensor_device = desc_emb.device
        to_cuda = 'cuda' in batch_action[0].device.type
        batch_n_nodes = [len(node_name) for node_name in batch_node_name_list]
        B = len(batch_n_nodes)
        N = max(batch_n_nodes)

        ### padding the link gt
        if not inference:
            max_timestep = batch_action.size(1) - 1
            all_target_topology_collector = []
            for b_i in range(B):
                c_t, n = target_topology_collector[b_i].shape
                if c_t < max_timestep:
                    c_target_topology_collector = torch.cat((target_topology_collector[b_i], target_topology_collector[b_i][-1,:][None,:].repeat(max_timestep-c_t, 1)))
                else:
                    c_target_topology_collector = target_topology_collector[b_i]
                    
                c_target_topology_collector = F.pad(c_target_topology_collector, (0, N - n))
                all_target_topology_collector.append(c_target_topology_collector)
                
            target_topology_collector = torch.stack(all_target_topology_collector, dim=0).to(tensor_device).long()

        ### Initialize variables
        action_logits_list, object1_logits_list, object2_logits_list = [], [], []
        action_correct_list, object1_correct_list, object2_correct_list = [], [], []
        action_predict_list, object1_predict_list, object2_predict_list = [], [], []

        action_graph_logits_list, object1_graph_logits_list, object2_graph_logits_list = [], [], []
        action_graph_correct_list, object1_graph_correct_list, object2_graph_correct_list = [], [], []
        action_graph_predict_list, object1_graph_predict_list, object2_graph_predict_list = [], [], []

        action_object_constraints = [torch.tensor(x, dtype=float_tensor_type, device=tensor_device) \
                            for x in self.action_object_constraints]
        if to_cuda:
            action_object_constraints = [x.cuda() for x in action_object_constraints]

        # zero initialization
        hx = torch.zeros([B, self.program_classifier.decode_gru_cell.hidden_size], 
                        dtype=float_tensor_type, device=tensor_device)

        ### Graph preparation
        # batch_adjacency_matrix: E, B, N, N
        batch_adjacency_matrix = torch.stack(batch_adjacency_matrix, 1)
        # batch_node_mask: B, N (used to mask out invalid nodes)
        batch_node_mask = self._create_mask2d(B, N, batch_n_nodes, -np.inf, to_cuda)
        # batch_node_states: B, N, 3*EMB
        batch_node_states, batch_object_list = _initialize_node_embedding_with_graph(
            batch_n_nodes,
            batch_node_name_list,
            batch_node_category_list,
            batch_node_states_,
            self.program_classifier.object_embedding,
            self.category_embedding,
            self.object_dict,
            self.node_state2_hidden,
            to_cuda)
        edge_matrix = self.matrix2idx_edge(batch_adjacency_matrix, tensor_device)
        init_batch_node_states = batch_node_states
        init_batch_node_states = self.env_init_graph2interaction_graph(init_batch_node_states.view(B * N, -1)).view(B, N, -1)
        batch_node_states = self.env_graph_encoder(edge_matrix, init_batch_node_states, task_emb=desc_emb[:,None,:].repeat(1, N, 1))
        batch_node_states = self.env_graph2interactoin_graph(batch_node_states)
        executor_list, scene_state_list = self.interaction_grpah_helper.create_batch_graph(init_graph)
        # Initial the memory for dynamic graph update
        memory_batch_node_states = batch_node_states
        temporal_batch_node_states = batch_node_states
        valid_check = True

        if inference:
            action, _, _ = self._initial_input(B, kwargs["initial_program"], to_cuda)
            p = Constant(0.)
        else:
            p = kwargs['p']

        ### Future link preparation
        character_mask = torch.zeros([B, N]).to(tensor_device)
        if inference:
            character_index_list = [[i for i,name in enumerate(batch_node_name_list[b_i]) if 'character' in name][0] for b_i in range(B)]
            character_index_list = torch.tensor(character_index_list).long().cuda()
        for b_i in range(B):
            character_mask[b_i, character_index_list[b_i]] = -np.inf
        link_loss_list = []
        agent_hx = torch.zeros([B, self.graph_state_dim],  dtype=float_tensor_type, device=tensor_device)

        ### GT categorical object label preparation
        batch_word_object1 = None
        batch_word_object2 = None
        for i in range(batch_action.size(1)):
            idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + batch_object1[:, i]
            object1_input = torch.index_select(batch_object_list.view(B * N, -1), 0, idx).view(B, -1)
            idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + batch_object2[:, i]
            object2_input = torch.index_select(batch_object_list.view(B * N, -1), 0, idx).view(B, -1)

            if batch_word_object1 is None:
                batch_word_object1 = object1_input
            else:
                batch_word_object1 = torch.cat((batch_word_object1, object1_input), dim=1)

            if batch_word_object2 is None:
                batch_word_object2 = object2_input
            else:
                batch_word_object2 = torch.cat((batch_word_object2, object2_input), dim=1)

        ### Language guidance module preparation
        current_sent = input_sent
        global_attention_mask = None
        past_key_values = None
        n_embd = self.config.n_embd
        ### grammer constraint
        none_index_list = [[i for i,name in enumerate(batch_node_name_list[b_i]) if 'none' in name][0] for b_i in range(B)]

        ### Main loop
        max_iter = batch_action.size(1) - 1 if not inference else 25-1
        for i in range(max_iter):
            ### Check if we want to use the previous generated intruction or ground truth as input
            is_gt = random.random() > (1 - p.v)
            if i == 0 or is_gt:
                action_input = action if inference else batch_action[:, i]
                action_emb = self.program_classifier.action_embedding(action_input)
                idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + batch_object1[:, i]
                object1_emb = torch.index_select(temporal_batch_node_states.view(B * N, -1), 0, idx)
                idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + batch_object2[:, i]
                object2_emb = torch.index_select(temporal_batch_node_states.view(B * N, -1), 0, idx)
                object1_input = batch_object1[:, i]
                object2_input = batch_object2[:, i]
                object1_word_input = batch_word_object1[:, i]
                object2_word_input = batch_word_object2[:, i]
            else:
                action_input = action_predict
                action_emb = self.program_classifier.action_embedding(action_input)
                object1_emb = object1_predict_emb
                object2_emb = object2_predict_emb
                object1_input = object1_node_predict
                object2_input = object2_node_predict
                idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + object1_node_predict
                object1_word_input = torch.index_select(batch_object_list.view(B * N, -1), 0, idx).view(-1)
                idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + object2_node_predict
                object2_word_input = torch.index_select(batch_object_list.view(B * N, -1), 0, idx).view(-1)

            node_name_idx = batch_object_list

            # atomic_action: B, 3*EMB
            atomic_action = torch.cat([action_emb, object1_emb, object2_emb], 1)
            atomic_action_emb = self.program_classifier.atomic_action2emb(atomic_action)

            ### Encode graph features
            gru_input = torch.cat([atomic_action_emb, desc_emb], 1)
            hx = self.program_classifier.decode_gru_cell(gru_input, hx)
            hidden = self.program_classifier.gru2hidden(hx)

            ### Update prompt, and encode language features
            interpreted_step = self.dset.interpret_sketch_woid([action_input.cpu().numpy()], [object1_word_input.cpu().numpy()], [object2_word_input.cpu().numpy()], with_sos=False, end_with_eos=False)
            interpreted_step_into_sent = transfrom_format_into_sentence(interpreted_step[0].split(", "))
            if i > 0 :
                current_sent = self.create_action_sent(i, interpreted_step_into_sent)
            encoded_src = self.tokenizer(current_sent, padding=True, truncation=True, max_length=1024, return_tensors='pt')
            input_ids = encoded_src['input_ids'].to(tensor_device)
            attention_mask = encoded_src['attention_mask'].to(tensor_device)
            if global_attention_mask is None:
                global_attention_mask = attention_mask
            else:
                global_attention_mask = torch.cat((global_attention_mask, attention_mask), dim=1)
            output = self.gpt2(input_ids=input_ids, attention_mask=global_attention_mask, past_key_values=past_key_values, output_hidden_states=True, return_dict=True)
            last_output_hidden = output.hidden_states[-1]
            past_key_values = output.past_key_values
            ids = torch.sum(attention_mask, dim=1) - 1
            tail_hidden = torch.gather(last_output_hidden.to(tensor_device), 1, ids[:,None,None].repeat(1, 1, n_embd)).squeeze(1)
            lm_hidden = self.context2hidden(tail_hidden)

            ### Dynamic Graph Update for previous timestep
            if  i > 0:
                ###### Insturction Execution
                if is_gt:
                    batch_adjacency_matrix, executor_list, scene_state_list, touch_info, state_info, edge_info = self.interaction_grpah_helper.batch_graph_evolve(action_input, batch_object1[:, i], batch_object2[:, i], batch_adjacency_matrix, batch_node_states_, to_cuda, batch_node_name_list, id2idx_list, executor_list, scene_state_list, valid_check=valid_check, edge_priority=True)
                else:
                    batch_adjacency_matrix, executor_list, scene_state_list, touch_info, state_info, edge_info = self.interaction_grpah_helper.batch_graph_evolve(action_input, object1_node_predict, object2_node_predict, batch_adjacency_matrix, batch_node_states_, to_cuda, batch_node_name_list, id2idx_list, executor_list, scene_state_list, valid_check=valid_check, edge_priority=True)
                ###### Memory-based Feature Update
                edge_matrix = self.matrix2idx_edge(batch_adjacency_matrix, tensor_device)
                memory_batch_node_states = self.memory_unit(memory_batch_node_states, action_emb, self.node_state2_hidden, touch_info, state_info, edge_info, state_update=True, edge_update=True, action_input=action_input)
                ###### Activity-aware Feature Propagation
                graph_task_hidden = self.graph_hidden_emb_proj(torch.cat((hidden, lm_hidden), dim=1))
                temporal_batch_node_states = self.prob_env_graph_encoder(edge_matrix, memory_batch_node_states, task_emb=graph_task_hidden[:,None,:].repeat(1, N, 1))

            ### Human-centric Probability Prediction
            ###### Encode Agent History
            C = batch_node_states.shape[-1]
            character_node_state = torch.gather(temporal_batch_node_states, dim=1, index=character_index_list[:,None,None].repeat(1,1,C)).squeeze(1)
            agent_hx = self.agent_decode_gru_cell(character_node_state, agent_hx)
            character_hidden = self.agent_gru2hidden(agent_hx)
            ###### Future Link Prediction
            link_hidden = self.hidden_emb_proj(torch.cat((hidden, lm_hidden), dim=1))
            link_input_hidden = torch.cat((link_hidden, character_hidden), dim=1)[:,None,:].repeat(1, N, 1)
            link_node_hidden = torch.cat((link_input_hidden, temporal_batch_node_states), dim=2)
            link_logits = self.link_predictor(link_node_hidden.view(B*N, -1)).view(B, N)
            if not inference:
                gt_link_labels = target_topology_collector[:,i,:].to(tensor_device)
                loss_of_link = F.binary_cross_entropy_with_logits(link_logits.view(B * N), gt_link_labels.contiguous().view(-1).float(), reduction='none').view(B, N)
                loss_of_link = [sum(loss_of_link[i, :(l - 1)]) for i, l in enumerate(batch_n_nodes)]
                loss_of_link = torch.stack(loss_of_link)
                link_loss_list.append(loss_of_link)
            else:
                link_loss_list.append(torch.zeros(B).to(tensor_device))
                
            ### Instruction Generation
            ###### Graph-guided Probability Prediction for object1
            logits_object1_t, object1_node_predict = self._decode_object(hidden, batch_node_mask, temporal_batch_node_states, self.program_classifier.hidden2object1_logits, other_logits=None)
            ###### Language-guided Probability Prediction for object1
            pooling_object1_weight = torch.softmax(logits_object1_t, dim=1)
            pooled_object1_hidden = torch.sum(pooling_object1_weight[:, :, None] * temporal_batch_node_states, dim=1)
            pooled_object1_hidden = self.program_decoder.node2hidden(pooled_object1_hidden)
            object1_logits = self.program_decoder.hidden2object1(torch.cat([lm_hidden, pooled_object1_hidden], 1))
            object1_predict = torch.argmax(object1_logits, 1)
            extended_object1_logits_list = []
            for b_i in range(B):
                c_object1_logits = torch.take(object1_logits[b_i,:], index=node_name_idx[b_i,:])
                extended_object1_logits_list.append(c_object1_logits)
            extended_object1_logits = torch.stack(extended_object1_logits_list, dim=0)
            ###### Fused Probability for object1
            logits_object1_t = logits_object1_t + extended_object1_logits + link_logits
            object1_node_predict = torch.argmax(logits_object1_t, 1)
            idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + object1_node_predict
            object1_predict_emb = torch.index_select(temporal_batch_node_states.view(B * N, -1), 0, idx)
            object1_predict_index = torch.index_select(batch_object_list.view(B * N, -1), 0, idx)

            ###### Action Probability Prediction
            action_logits = self.program_classifier.hidden2action_logits(torch.cat([lm_hidden, hidden, object1_predict_emb], 1))
            action_predict = torch.argmax(action_logits, 1)
            action_predict_emb = self.program_classifier.action_embedding(action_predict)
            hidden_w_prev_action_object1 = torch.cat([hidden, action_predict_emb, object1_predict_emb], 1)

            ###### Grammarly Constraint
            if inference:
                ###### Object2 prediction with grammer_constraint
                action_predict_np = action_predict.squeeze().cpu().numpy()
                if len(action_predict_np.shape) == 0:
                    action_predict_np = [action_predict_np]
                action_grammer_num = [int(self.grammer_constraint[self.dset.action_dict.idx2word[str(action)]]) for action in action_predict_np]
                action_grammer_num = np.array(action_grammer_num)
                action_grammer_idx = (action_grammer_num == 2)
                action_grammer_idx = 1 - action_grammer_idx
                c_batch_node_mask = copy.deepcopy(batch_node_mask)
                for b_i in range(B):
                    if action_grammer_idx[b_i] == 1:
                        c_batch_node_mask[b_i, :] = - np.inf
                        none_idx = none_index_list[b_i]
                        c_batch_node_mask[b_i, none_idx] = 0
                    else:
                        none_idx = none_index_list[b_i]
                        c_batch_node_mask[b_i, none_idx] = - np.inf
            if inference:
                batch_action_mask_constraints = c_batch_node_mask
            else:
                batch_action_mask_constraints = batch_node_mask

            ###### Graph-guided Probability Prediction for object2
            logits_object2_t, object2_node_predict = self._decode_object(hidden_w_prev_action_object1, batch_action_mask_constraints, temporal_batch_node_states, self.program_classifier.hidden2object2_logits, other_logits=None)
            ###### Language-guided Probability Prediction for object2
            pooling_object2_weight = torch.softmax(logits_object2_t, dim=1)
            pooled_object2_hidden = torch.sum(pooling_object2_weight[:, :, None] * temporal_batch_node_states, dim=1)
            pooled_object2_hidden = self.program_decoder.node2hidden(pooled_object2_hidden)
            object2_logits = self.program_decoder.hidden2object2(torch.cat([self.program_decoder.action_embedding(action_predict), self.program_decoder.object_embedding(object1_predict), lm_hidden, pooled_object2_hidden], 1))
            object2_predict = torch.argmax(object2_logits, 1)
            extended_object2_logits_list = []
            for b_i in range(B):
                c_object2_logits = torch.take(object2_logits[b_i,:], index=node_name_idx[b_i,:])
                extended_object2_logits_list.append(c_object2_logits)
            extended_object2_logits = torch.stack(extended_object2_logits_list, dim=0)
            ###### Fused Probability for object2
            logits_object2_t = logits_object2_t + extended_object2_logits
            object2_node_predict = torch.argmax(logits_object2_t, 1)
            idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + object2_node_predict
            object2_predict_emb = torch.index_select(temporal_batch_node_states.view(B * N, -1), 0, idx)

            ### LM Append tensors
            action_logits_list.append(action_logits)
            object1_logits_list.append(object1_logits)
            object2_logits_list.append(object2_logits)

            action_predict_list.append(action_predict)
            object1_predict_list.append(object1_predict)
            object2_predict_list.append(object2_predict)

            if not inference:
                action_correct = (action_predict == batch_action[:, i + 1])
                object1_correct = (object1_predict == batch_word_object1[:, i + 1])
                object2_correct = (object2_predict == batch_word_object2[:, i + 1])

                action_correct_list.append(action_correct)
                object1_correct_list.append(object1_correct)
                object2_correct_list.append(object2_correct)

            ### GB Append tensors
            action_logits_t = action_logits.detach()
            action_graph_logits_list.append(action_logits_t)
            object1_graph_logits_list.append(logits_object1_t)
            object2_graph_logits_list.append(logits_object2_t)

            action_graph_predict_list.append(action_predict)
            object1_graph_predict_list.append(object1_node_predict)
            object2_graph_predict_list.append(object2_node_predict)

            if not inference:
                action_graph_correct = (action_predict == batch_action[:, i + 1])
                object1_graph_correct = (object1_node_predict == batch_object1[:, i + 1])
                object2_graph_correct = (object2_node_predict == batch_object2[:, i + 1])

                action_graph_correct_list.append(action_graph_correct)
                object1_graph_correct_list.append(object1_graph_correct)
                object2_graph_correct_list.append(object2_graph_correct)

        # B, T
        action_predict = torch.stack(action_predict_list, 1)
        object1_predict = torch.stack(object1_predict_list, 1)
        object2_predict = torch.stack(object2_predict_list, 1)

        action_graph_predict = torch.stack(action_graph_predict_list, 1)
        object1_graph_predict = torch.stack(object1_graph_predict_list, 1)
        object2_graph_predict = torch.stack(object2_graph_predict_list, 1)

        info = {}
        info.update({"action_predict": action_predict.cpu()})
        info.update({"object1_predict": object1_predict.cpu()})
        info.update({"object2_predict": object2_predict.cpu()})
        info.update({"action_graph_predict": action_graph_predict.cpu()})
        info.update({"object1_graph_predict": object1_graph_predict.cpu()})
        info.update({"object2_graph_predict": object2_graph_predict.cpu()})
        info.update({"batch_node_name_list": batch_node_name_list})

        if not inference:
            # B, T
            action_correct_list = torch.stack(action_correct_list, 1).float()
            object1_correct_list = torch.stack(object1_correct_list, 1).float()
            object2_correct_list = torch.stack(object2_correct_list, 1).float()

            self._calculate_accuracy(
                action_correct_list,
                object1_correct_list,
                object2_correct_list,
                batch_length,
                info, graph=False)

            action_graph_correct_list = torch.stack(action_graph_correct_list, 1).float()
            object1_graph_correct_list = torch.stack(object1_graph_correct_list, 1).float()
            object2_graph_correct_list = torch.stack(object2_graph_correct_list, 1).float()

            self._calculate_accuracy(
                action_graph_correct_list,
                object1_graph_correct_list,
                object2_graph_correct_list,
                batch_length,
                info, graph=True)

            # action_logits_list: B, T, A
            # object1_logits_list: B, T, N
            T = batch_object1.size(1)
            action_logits_list = torch.stack(action_logits_list, 1)
            object1_logits_list = torch.stack(object1_logits_list, 1)
            object2_logits_list = torch.stack(object2_logits_list, 1)

            object1_loss, object2_loss = self._calculate_object_loss(
                B, T - 1, object1_logits_list, object2_logits_list, batch_word_object1[:, 1:], batch_word_object2[:, 1:])
            action_loss = self._calcualte_action_loss(B, T - 1, action_logits_list, batch_action[:, 1:])

            loss_word = self._calculate_loss(
                action_loss,
                object1_loss,
                object2_loss,
                batch_length,
                info,graph=False)

            action_graph_logits_list = torch.stack(action_graph_logits_list, 1)
            object1_graph_logits_list = torch.stack(object1_graph_logits_list, 1)
            object2_graph_logits_list = torch.stack(object2_graph_logits_list, 1)

            object1_graph_loss, object2_graph_loss = self._calculate_object_loss(
                B, T - 1, object1_graph_logits_list, object2_graph_logits_list, batch_object1[:, 1:], batch_object2[:, 1:])
            action_graph_loss = self._calcualte_action_loss(B, T - 1, action_graph_logits_list, batch_action[:, 1:])

            loss_graph = self._calculate_loss(
                action_graph_loss,
                object1_graph_loss,
                object2_graph_loss,
                batch_length,
                info,graph=True)

            link_loss = torch.stack(link_loss_list, dim=1) # B, T
            link_valid_loss = [sum(link_loss[i, :(l - 1)]) for i, l in enumerate(batch_length)]
            link_valid_loss = sum(link_valid_loss) / (sum(batch_length) - 1. * len(batch_length))
            
            loss = link_valid_loss*0.01 + loss_graph + loss_word*0.1 # + 0.001*loss_of_mask
        else:
            loss = 0

        return loss, info