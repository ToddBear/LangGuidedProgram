import json
import os
import copy
import collections
import numpy as np
from termcolor import colored
from program.environment import EnvironmentGraph, EnvironmentState
from program.utils import load_name_equivalence
from program.execution import ScriptExecutor
from program.scripts import parse_script_line, Script
import torch
from torch.nn.utils.rnn import pad_sequence

class EnvGraphHelper(object):

    def __init__(self):
        edge_types = ['on', 'inside', 'between', 'close', 'facing', 'holds_rh', 'holds_lh']
        edge_type2idx = {edge_type: i for i, edge_type in enumerate(edge_types)}
        idx2edge_type = {str(i): edge_type for i, edge_type in enumerate(edge_types)}

        node_states = ['closed', 'open', 'on', 'off', 'sitting', 'dirty', 'clean', 'lying', 'plugged_in', 'plugged_out']
        node_state2idx = {node_state: i for i, node_state in enumerate(node_states)}
        idx2node_state = {str(i): node_state for i, node_state in enumerate(node_states)}

        node_categories = ['characters', 'appliances', 'placable_objects', 'ceiling', 'doors', 'lamps', 'electronics', 
                        'decor', 'windows', 'props', 'rooms', 'floors', 'walls', 'floor', 'furniture', 'special_token']
        node_category2idx = {node_category: i for i, node_category in enumerate(node_categories)}
        idx2node_category = {str(i): node_category for i, node_category in enumerate(node_categories)}

        property_types = ['surfaces', 'grabbable', 'sittable', 'lieable', 'hangable', 'drinkable', 'eatable', 'recipient', 
                        'cuttable', 'pourable', 'can_open', 'has_switch', 'readable', 'lookable', 'containers', 'clothes', 
                        'person', 'body_part', 'cover_object', 'has_plug', 'has_paper', 'movable', 'cream']
        property2idx = {property: i for i, property in enumerate(property_types)}
        idx2property = {str(i): property for i, property in enumerate(property_types)}

        touch_types = ['subject', 'object1', 'object2']
        touch_type2idx = {touch_type: i for i, touch_type in enumerate(touch_types)}
        idx2touch_type = {str(i): touch_type for i, touch_type in enumerate(touch_types)}


        self.n_edge = len(edge_types)
        self.edge_types = edge_types
        self.edge_type2idx = edge_type2idx
        self.idx2edge_type = idx2edge_type

        self.node_states = node_states
        self.n_node_state = len(node_states)
        self.node_state2idx = node_state2idx
        self.idx2node_state = idx2node_state

        self.n_node_category = len(node_categories)
        self.node_categories = node_categories
        self.node_category2idx = node_category2idx
        self.idx2node_category = idx2node_category

        self.n_property_type = len(property_types)
        self.property_type = property_types
        self.property2idx = property2idx
        self.idx2property = idx2property

        self.n_touch_type = len(touch_types)
        self.touch_types = touch_types
        self.touch_type2idx = touch_type2idx
        self.idx2touch_type = idx2touch_type

    def load_one_graph(self, graph_path, detail2abstract=None):

        def _process_init_graph(graph):

            # node name
            node_names = ['{}.{}'.format(node["class_name"], node["id"]) for node in graph["nodes"]]
            node_names.append('<eos>.<eos>')
            node_names.append('<none>.<none>')
            
            node_ids = [node["id"] for node in graph["nodes"]]
            merged_node_names = []
            for node in node_names:
                name, instance = node.split('.')
                if name in detail2abstract:
                    name = detail2abstract[name]
                merged_node_names.append('{}.{}'.format(name, instance))
            node_names = merged_node_names
            n_nodes = len(node_names)

            # node category
            node_category = [self.node_category2idx[node["category"].lower()] for node in graph["nodes"]]
            node_category.append(self.node_category2idx['special_token'])
            node_category.append(self.node_category2idx['special_token'])

            # node states
            node_states = np.zeros([n_nodes, self.n_node_state], dtype=np.int32)
            for i, node in enumerate(graph["nodes"]):
                states = node["states"]
                for state in states:
                    state = state.lower()
                    idx = self.node_states.index(state)
                    node_states[i, idx] = 1

            # adjacency matrix
            adjacency_matrix = np.zeros([self.n_edge, n_nodes, n_nodes], dtype=np.bool)
            for edge in graph["edges"]:
                edge_type = edge["relation_type"]
                src_id = edge["from_id"]
                tgt_id = edge["to_id"]

                edge_type_idx = self.edge_type2idx[edge_type.lower()]
                src_idx = node_ids.index(src_id)
                tgt_idx = node_ids.index(tgt_id)

                adjacency_matrix[edge_type_idx, src_idx, tgt_idx] = True

            return adjacency_matrix, node_names, node_category, node_states

        graph = json.load(open(graph_path, 'r'))
        init_graph = graph["init_graph"]
        adjacency_matrix, node_names, node_category, node_states = _process_init_graph(init_graph)
        final_graph = graph["final_graph"]

        return adjacency_matrix, node_names, node_category, node_states, init_graph, final_graph


class InteractionGraphHelper(object):

    def __init__(self, action_dict):

        edge_types = ['interact']
        edge_type2idx = {edge_type: i for i, edge_type in enumerate(edge_types)}
        idx2edge_type = {str(i): edge_type for i, edge_type in enumerate(edge_types)}

        touch_types = ['character', 'object', 'subject']
        touch_type2idx = {touch_type: i for i, touch_type in enumerate(touch_types)}
        idx2touch_type = {str(i): touch_type for i, touch_type in enumerate(touch_types)}
        
        release_actions = ['putobjback', 'putback', 'putoff', 'leave', 'drop', 'release']

        self.n_edge = len(edge_types)
        self.edge_types = edge_types
        self.edge_type2idx = edge_type2idx
        self.idx2edge_type = idx2edge_type

        self.n_touch = len(touch_types)
        self.touch_types = touch_types
        self.touch_type2idx = touch_type2idx
        self.idx2touch_type = idx2touch_type

        self.action_dict = action_dict
        self.release_actions = release_actions
        self.batch_node_name_list = None

    def batch_init_adjacency_matrix(self, B, N):
        return np.zeros([self.n_edge, B, N, N])

    def set_batch_node_name_list(self, batch_node_name_list):
        self.batch_node_name_list = batch_node_name_list

    def _update_adjacency_matrix(self, character_idx, action, object1, object2, adjacency_matrix):

        # add directed edge
        if 'eos' not in object1[1] and 'none' not in object1[1]:
            # character interact with object1
            adjacency_matrix[self.edge_types.index('interact'), character_idx, object1[0]] = 1

        if 'eos' not in object2[1] and 'none' not in object2[1]:
            # put object1 on object2
            adjacency_matrix[self.edge_types.index('interact'), object1[0], object2[0]] = 1

        return adjacency_matrix

    def _update_touch_idx(self, character_idx, object1, object2, touch_idx, touch_mask):

        touch_idx[character_idx, self.touch_types.index('character')] = 1
        touch_mask[character_idx] = 1
        touch_nodes = [character_idx]

        if 'eos' not in object1[1] and 'none' not in object1[1]:
            touch_idx[object1[0], self.touch_types.index('object')] = 1
            touch_mask[object1[0]] = 1
            touch_nodes.append(object1[0].item())

        if 'eos' not in object2[1] and 'none' not in object2[1]:
            touch_idx[object2[0], self.touch_types.index('subject')] = 1
            touch_mask[object2[0]] = 1
            touch_nodes.append(object2[0].item())

        return touch_idx, touch_mask, touch_nodes

    def _update_related_mask(self, character_idx, object1, object2, adjacency_matrix, touch_nodes):

        _, N, _ = adjacency_matrix.shape
        related_matrix = np.sum(adjacency_matrix, 0) + np.eye(N)
        related_mask = np.zeros(N)

        graph = {}
        srcs, tgts = np.where(related_matrix != 0)
        for src, tgt in zip(srcs, tgts):
            if src not in graph:
                graph.update({src: []})
            graph[src].append(tgt)

        visited = copy.copy(touch_nodes)
        for root in touch_nodes:
            visited += _breadth_first_search(graph, root)
        visited = list(set(visited))
        related_mask[visited] = 1
       
        return related_mask

    def _log(self, action, object1_name, object2_name, adjacency_matrix, related_mask, node_name_list):

        print("-"*30)
        print(colored("Action: [{}] <{}> <{}>".format(action, object1_name, object2_name), 'green'))

        for i, edge_type in enumerate(self.edge_types):
            i_adjacency_matrix = adjacency_matrix[i, :]
            srcs, tgts = np.where(i_adjacency_matrix == 1)
            for src, tgt in zip(srcs, tgts):
                print("{} {} {}".format(node_name_list[src], edge_type, node_name_list[tgt]))

        related_nodes = np.where(related_mask == 1)[0]
        related_nodes_name = [node_name_list[idx] for idx in related_nodes]
        print("Nodes needed to be changed: ", ', '.join(related_nodes_name))

    def batch_graph_evolve(self, batch_action, batch_object1, batch_object2, batch_adjacency_matrix, to_cuda):

        #batch_adjacency_matrix = batch_adjacency_matrix.cpu().numpy()
        _, B, N, _ = batch_adjacency_matrix.size()
        batch_adjacency_matrix = self.batch_init_adjacency_matrix(B, N)

        batch_action = self.action_dict.convert_idx2word(batch_action.cpu().numpy())
        batch_adjacency_matrix = batch_adjacency_matrix.transpose(1, 0, 2, 3)           # b, e, n, n
        batch_related_mask = []
        batch_touch_idx = []
        batch_touch_mask = []
        new_batch_adjacency_matrix = []
        N = batch_adjacency_matrix.shape[-1]

        # to cpu
        batch_object1 = batch_object1.cpu()
        batch_object2 = batch_object2.cpu()

        for j, (action, object1_idx, object2_idx, adjacency_matrix, node_name_list) in enumerate(zip(batch_action, batch_object1, batch_object2, batch_adjacency_matrix, self.batch_node_name_list)):

            #verbose = True if j == 0 else False
            verbose = False
            touch_idx = np.zeros([N, self.n_touch])
            touch_mask = np.zeros(N)

            object1_name = node_name_list[object1_idx]
            object2_name = node_name_list[object2_idx]
            object1 = (object1_idx, object1_name)
            object2 = (object2_idx, object2_name)
            character_idx = ['character' in name for name in node_name_list].index(True)

            # update
            adjacency_matrix = self._update_adjacency_matrix(character_idx, action, object1, object2, adjacency_matrix)
            touch_idx, touch_mask, touch_nodes = self._update_touch_idx(character_idx, object1, object2, touch_idx, touch_mask)
            related_mask = self._update_related_mask(character_idx, object1, object2, adjacency_matrix, touch_nodes)

            if verbose:
                self._log(action, object1_name, object2_name, adjacency_matrix, related_mask, node_name_list)

            new_batch_adjacency_matrix.append(adjacency_matrix)
            batch_touch_idx.append(touch_idx)
            batch_touch_mask.append(touch_mask)
            batch_related_mask.append(related_mask)

        new_batch_adjacency_matrix = np.array(new_batch_adjacency_matrix).astype(np.float32)
        new_batch_adjacency_matrix = torch.tensor(new_batch_adjacency_matrix)
        batch_touch_idx = np.array(batch_touch_idx).astype(np.float32)
        batch_touch_idx = torch.tensor(batch_touch_idx)
        batch_touch_mask = np.array(batch_touch_mask).astype(np.float32)
        batch_touch_mask = torch.tensor(batch_touch_mask)
        batch_related_mask = np.array(batch_related_mask).astype(np.float32)
        batch_related_mask = torch.tensor(batch_related_mask)

        if to_cuda:
            new_batch_adjacency_matrix = new_batch_adjacency_matrix.cuda()
            batch_touch_idx = batch_touch_idx.cuda()
            batch_touch_mask = batch_touch_mask.cuda()
            batch_related_mask = batch_related_mask.cuda()
            
        new_batch_adjacency_matrix = new_batch_adjacency_matrix.transpose(1, 0)
        return new_batch_adjacency_matrix, batch_touch_idx, batch_touch_mask, batch_related_mask


def _breadth_first_search(graph, root): 
    visited, queue = set(), collections.deque([root])
    while queue: 
        vertex = queue.popleft()
        for neighbour in graph[vertex]: 
            if neighbour not in visited: 
                visited.add(neighbour) 
                queue.append(neighbour) 

    return list(visited)

class InteractionGraphHelperGeo(object):

    def __init__(self, action_dict):


        edge_types = ['on', 'inside', 'between', 'close', 'facing', 'holds_rh', 'holds_lh']
        edge_priority = {0:2, 1:2, 2:0, 3:3, 4:1, 5:0, 6:0}
        edge_type2idx = {edge_type: i for i, edge_type in enumerate(edge_types)}
        idx2edge_type = {str(i): edge_type for i, edge_type in enumerate(edge_types)}

        touch_types = ['character', 'object', 'subject']
        touch_type2idx = {touch_type: i for i, touch_type in enumerate(touch_types)}
        idx2touch_type = {str(i): touch_type for i, touch_type in enumerate(touch_types)}
        
        release_actions = ['putobjback', 'putback', 'putoff', 'leave', 'drop', 'release']

        node_states = ['closed', 'open', 'on', 'off', 'sitting', 'dirty', 'clean', 'lying', 'plugged_in', 'plugged_out']
        node_state2idx = {node_state: i for i, node_state in enumerate(node_states)}
        idx2node_state = {str(i): node_state for i, node_state in enumerate(node_states)}

        self.n_edge = len(edge_types)
        self.edge_types = edge_types
        self.edge_type2idx = edge_type2idx
        self.idx2edge_type = idx2edge_type
        self.edge_priority = edge_priority

        self.n_touch = len(touch_types)
        self.touch_types = touch_types
        self.touch_type2idx = touch_type2idx
        self.idx2touch_type = idx2touch_type

        self.action_dict = action_dict
        self.release_actions = release_actions
        self.batch_node_name_list = None

        self.name_equivalence = load_name_equivalence()
        self.n_state = len(node_states)

    def batch_init_adjacency_matrix(self, B, N):
        return np.zeros([self.n_edge, B, N, N])

    def create_batch_graph(self, graph_dicts):
        executor_list = []
        scene_state_list = []
        for _, graph_dict in enumerate(graph_dicts):
            graph = EnvironmentGraph(graph_dict)
            scene_state = EnvironmentState(graph, self.name_equivalence, instance_selection=True)
            scene_state_list.append(scene_state)
            executor = ScriptExecutor(graph, self.name_equivalence)
            executor_list.append(executor)
        return executor_list, scene_state_list

    def _update_adjacency_matrix(self, character_idx, action, object1, object2, adjacency_matrix):

        # add directed edge
        if 'eos' not in object1[1] and 'none' not in object1[1]:
            # character interact with object1
            adjacency_matrix[self.edge_types.index('interact'), character_idx, object1[0]] = 1

        if 'eos' not in object2[1] and 'none' not in object2[1]:
            # put object1 on object2
            adjacency_matrix[self.edge_types.index('interact'), object1[0], object2[0]] = 1

        return adjacency_matrix

    def _update_touch_idx(self, character_idx, object1, object2, touch_idx, touch_mask):

        touch_idx[character_idx, self.touch_types.index('character')] = 1
        touch_mask[character_idx] = 1
        touch_nodes = [character_idx]

        if 'eos' not in object1[1] and 'none' not in object1[1]:
            touch_idx[object1[0], self.touch_types.index('object')] = 1
            touch_mask[object1[0]] = 1
            touch_nodes.append(object1[0].item())

        if 'eos' not in object2[1] and 'none' not in object2[1]:
            touch_idx[object2[0], self.touch_types.index('subject')] = 1
            touch_mask[object2[0]] = 1
            touch_nodes.append(object2[0].item())

        return touch_idx, touch_mask, touch_nodes

    def _update_related_mask(self, character_idx, object1, object2, adjacency_matrix, touch_nodes):

        _, N, _ = adjacency_matrix.shape
        related_matrix = np.sum(adjacency_matrix, 0) + np.eye(N)
        related_mask = np.zeros(N)

        graph = {}
        srcs, tgts = np.where(related_matrix != 0)
        for src, tgt in zip(srcs, tgts):
            if src not in graph:
                graph.update({src: []})
            graph[src].append(tgt)

        visited = copy.copy(touch_nodes)
        for root in touch_nodes:
            visited += _breadth_first_search(graph, root)
        visited = list(set(visited))
        related_mask[visited] = 1
       
        return related_mask

    def _log(self, action, object1_name, object2_name, adjacency_matrix, related_mask, node_name_list):

        print("-"*30)
        print(colored("Action: [{}] <{}> <{}>".format(action, object1_name, object2_name), 'green'))

        for i, edge_type in enumerate(self.edge_types):
            i_adjacency_matrix = adjacency_matrix[i, :]
            srcs, tgts = np.where(i_adjacency_matrix == 1)
            for src, tgt in zip(srcs, tgts):
                print("{} {} {}".format(node_name_list[src], edge_type, node_name_list[tgt]))

        related_nodes = np.where(related_mask == 1)[0]
        related_nodes_name = [node_name_list[idx] for idx in related_nodes]
        print("Nodes needed to be changed: ", ', '.join(related_nodes_name))

    def _update_and_record(self, N, prev_state, next_state, adjacency_matrix, id2idx, edge_priority=False):
        state_idx = np.zeros([N, self.n_state])
        state_mask = np.zeros(N)
        edge_idx = np.zeros(N)
        edge_type_list = np.zeros(N)
        edge_mask = np.zeros(N)

        if next_state is not None:
            # edge related record
            prev_edge_list = set()
            for key in list(prev_state._new_edges_from.keys()):
                from_id = key[0]
                relation_type = key[1].value
                to_id_list = list(prev_state._new_edges_from[key])
                for to_id in to_id_list:
                    prev_edge_list.add("{}_{}_{}".format(min(int(from_id), int(to_id)), relation_type, max(int(from_id), int(to_id))))
            new_edge_list = set()
            for key in list(next_state._new_edges_from.keys()):
                from_id = key[0]
                relation_type = key[1].value
                to_id_list = list(next_state._new_edges_from[key])
                for to_id in to_id_list:
                    edge_name = "{}_{}_{}".format(min(int(from_id), int(to_id)), relation_type, max(int(from_id), int(to_id)))
                    if edge_name not in prev_edge_list:
                        new_edge_list.add(edge_name)
            
            # state related record
            prev_state_list = set()
            for key in list(prev_state._new_nodes.keys()):
                state_list = prev_state._new_nodes[key].states
                node_id = prev_state._new_nodes[key].id
                for state in state_list:
                    state_type = state.value
                    prev_state_list.add("{}_{}".format(node_id, state_type))

            new_state_list = set()
            for key in list(next_state._new_nodes.keys()):
                state_list = next_state._new_nodes[key].states
                node_id = next_state._new_nodes[key].id
                for state in state_list:
                    state_type = state.value
                    state_name = "{}_{}".format(node_id, state_type)
                    if state_name not in prev_state_list:
                        new_state_list.add(state_name)

            node_map = next_state._graph._node_map
            # edge-related update
            ## first collect the edge update info
            ## then update the adjacency_matrix
            if edge_priority:
                new_edge_list = sorted(new_edge_list, key=lambda item:self.edge_priority[int(item.split("_")[1])-1])
            for new_edge in new_edge_list:
                from_id, edge_type, to_id = list(map(int, new_edge.split("_")))
                from_index = id2idx[from_id]
                to_index = id2idx[to_id]
                adjacency_matrix[edge_type-1, from_index, to_index] = 1
                edge_idx[from_index] = to_index
                edge_idx[to_index] = from_index
                edge_mask[from_index] = 1
                edge_mask[to_index] = 1
                edge_type_list[from_index] = edge_type-1
                edge_type_list[to_index] = edge_type-1

            # state-related update
            for new_state in new_state_list:
                node_id, state_type = list(map(int, new_state.split("_")))
                # node_name = "{}.{}".format(node_map[node_id].class_name, node_map[node_id].id)
                # node_index = node_name_list.index(node_name)
                node_index = id2idx[node_id]
                state_idx[node_index, state_type-1] = 1
                state_mask[node_index] = 1

            # delete edge record
            prev_delete_edge_list = set()
            for key in list(prev_state._removed_edges_from.keys()):
                from_id = key[0]
                relation_type = key[1].value
                to_id_list = list(prev_state._removed_edges_from[key])
                for to_id in to_id_list:
                    prev_edge_list.add("{}_{}_{}".format(from_id, relation_type, to_id))
                if len(to_id_list) == 0:
                    prev_edge_list.add("{}_{}_{}".format(from_id, relation_type, "all"))
            new_delete_edge_list = set()
            for key in list(next_state._removed_edges_from.keys()):
                from_id = key[0]
                relation_type = key[1].value
                to_id_list = list(next_state._removed_edges_from[key])
                for to_id in to_id_list:
                    edge_name = "{}_{}_{}".format(from_id, relation_type, to_id)
                    if edge_name not in prev_delete_edge_list:
                        new_delete_edge_list.add(edge_name)
                if len(to_id_list) == 0:
                    if edge_name not in prev_delete_edge_list:
                        new_delete_edge_list.add("{}_{}_{}".format(from_id, relation_type, "all"))
            for delete_edge in new_delete_edge_list:
                from_id, edge_type, to_id = delete_edge.split("_")
                from_id, edge_type = int(from_id), int(edge_type)
                # from_node_name = "{}.{}".format(node_map[from_id].class_name, node_map[from_id].id)
                # from_index = node_name_list.index(from_node_name)
                from_index = id2idx[from_id]
                if to_id == "all":
                    adjacency_matrix[edge_type-1, from_index, :] = 0
                else:
                    to_id = int(to_id)
                    to_index = id2idx[to_id]
                    adjacency_matrix[edge_type-1, from_index, to_index] = 0

        return state_idx, state_mask, edge_idx, edge_type_list, edge_mask, adjacency_matrix
    
    def batch_graph_evolve(self, batch_action, batch_object1, batch_object2, batch_adjacency_matrix, batch_node_state_, to_cuda, batch_node_name_list, id2idx_list, executor_list, scene_state_list, valid_check=True, edge_priority=False):

        #batch_adjacency_matrix = batch_adjacency_matrix.cpu().numpy()
        _, B, N, _ = batch_adjacency_matrix.size()
        batch_action = self.action_dict.convert_idx2word(batch_action.cpu().numpy())
        batch_adjacency_matrix = batch_adjacency_matrix.transpose(1, 0)           # b, e, n, n
        batch_touch_idx = [] # B, N, n_touch
        batch_touch_mask = [] # B, N
        batch_state_idx = [] # B, N
        batch_state_mask = [] # B, N
        batch_state_legal = [] # B, N
        batch_edge_idx = [] # B, N
        batch_edge_type = [] # B, N
        batch_edge_mask = [] # B, N
        new_batch_adjacency_matrix = []
        N = batch_adjacency_matrix.shape[-1]

        # to cpu
        batch_object1 = batch_object1.cpu()
        batch_object2 = batch_object2.cpu()

        for j, (action, object1_idx, object2_idx, adjacency_matrix, node_name_list, id2idx) in enumerate(zip(batch_action, batch_object1, batch_object2, batch_adjacency_matrix, batch_node_name_list, id2idx_list)):

            #verbose = True if j == 0 else False
            verbose = False
            touch_idx = np.zeros([N, self.n_touch])
            touch_mask = np.zeros(N)
            state_idx = np.zeros([N, self.n_state])
            state_legal = np.zeros(N)
            state_mask = np.zeros(N)
            edge_idx = np.zeros(N)
            edge_type = np.zeros(N)
            edge_mask = np.zeros(N)

            object1_name = node_name_list[object1_idx]
            object2_name = node_name_list[object2_idx]
            object1 = (object1_idx, object1_name)
            object2 = (object2_idx, object2_name)
            character_idx = ['character' in name for name in node_name_list].index(True)

            instruction_string = "[{}] <{}> ({}) <{}> ({})".format(action, object1_name.split(".")[0], object1_name.split(".")[1], object2_name.split(".")[0], object2_name.split(".")[1])
            instruction_string = instruction_string.replace("<<none>> (<none>)", "").replace("<<eos>> (<eos>)", "").replace("[<eos>]", "")
            instruction_string = instruction_string.strip()

            prev_state = scene_state_list[j]
            next_state = None
            if len(instruction_string) != 0:
                instruction_line = parse_script_line(instruction_string, 1)
                if instruction_line != None:
                    script = Script([instruction_line])
                    try:
                        next_state = executor_list[j].execute_one_step(script, prev_state, valid_check=valid_check)
                    except AttributeError:
                        next_state = None

            if next_state is not None:
                scene_state_list[j] = next_state

            state_idx, state_mask, edge_idx, edge_type, edge_mask, adjacency_matrix = self._update_and_record(N, prev_state, next_state, adjacency_matrix, id2idx, edge_priority)
            touch_idx, touch_mask, touch_nodes = self._update_touch_idx(character_idx, object1, object2, touch_idx, touch_mask)

            if verbose:
                self._log(action, object1_name, object2_name, adjacency_matrix, node_name_list)

            new_batch_adjacency_matrix.append(adjacency_matrix)
            batch_touch_idx.append(touch_idx)
            batch_touch_mask.append(touch_mask)
            batch_state_idx.append(state_idx)
            batch_state_mask.append(state_mask)
            batch_state_legal.append(state_legal)
            batch_edge_idx.append(edge_idx)
            batch_edge_type.append(edge_type)
            batch_edge_mask.append(edge_mask)


        new_batch_adjacency_matrix = torch.stack(new_batch_adjacency_matrix, dim=0)
        batch_touch_idx = np.array(batch_touch_idx).astype(np.float32)
        batch_touch_idx = torch.tensor(batch_touch_idx)
        batch_touch_mask = np.array(batch_touch_mask).astype(np.float32)
        batch_touch_mask = torch.tensor(batch_touch_mask)

        batch_state_idx = np.array(batch_state_idx).astype(np.float32)
        batch_state_idx = torch.tensor(batch_state_idx)    
        batch_state_mask = np.array(batch_state_mask).astype(np.float32)
        batch_state_mask = torch.tensor(batch_state_mask)
        batch_state_legal = np.array(batch_state_legal).astype(np.float32)
        batch_state_legal = torch.tensor(batch_state_legal)

        batch_edge_idx = np.array(batch_edge_idx).astype(np.float32)
        batch_edge_idx = torch.tensor(batch_edge_idx)
        batch_edge_type = np.array(batch_edge_type).astype(np.float32)
        batch_edge_type = torch.tensor(batch_edge_type)
        batch_edge_mask = np.array(batch_edge_mask).astype(np.float32)
        batch_edge_mask = torch.tensor(batch_edge_mask)

        if to_cuda:
            new_batch_adjacency_matrix = new_batch_adjacency_matrix.cuda()
            batch_touch_idx = batch_touch_idx.cuda()
            batch_touch_mask = batch_touch_mask.cuda()
            batch_state_idx = batch_state_idx.cuda()
            batch_state_mask = batch_state_mask.cuda()
            batch_state_legal = batch_state_legal.cuda()
            batch_edge_idx = batch_edge_idx.cuda()
            batch_edge_type = batch_edge_type.cuda()
            batch_edge_mask = batch_edge_mask.cuda()

        new_batch_adjacency_matrix = new_batch_adjacency_matrix.transpose(1, 0)

        return new_batch_adjacency_matrix, executor_list, scene_state_list, (batch_touch_idx, batch_touch_mask), (batch_state_idx, batch_state_legal, batch_state_mask), (batch_edge_idx, batch_edge_type, batch_edge_mask)