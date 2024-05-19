import os
import random
import json
import re
import numpy as np
from termcolor import colored

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import re
from torch.nn.utils.rnn import pad_sequence

# General
from dataset_utils import load_programs, get_program_dictionary, expand_dictionary_with_resources, convert2idx_program, dictionary, \
                          get_desc_tokens, get_title_tokens, get_words, \
                          convert2idx_title, convert2idx_desc

# Program
from dataset_utils import load_detail2abstract, convert_to_graph_path, add_ground_truth
from program.graph_utils import EnvGraphHelper


cur_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(cur_dir, '../../dataset/VirtualHome-Env/')
resource_dir = os.path.join(cur_dir, '../../dataset/VirtualHome-Env/resources/')

PARENT_PATH = ""
program_path = PARENT_PATH + "/Data/original_programs/executable_programs/*/*/*txt"
augment_program_path = PARENT_PATH + "/Data/augment_programs/*/executable_programs/*/*/*/*txt"
sketch_path = PARENT_PATH + "/LangGuidedProgram/dataset/VirtualHome-Env/sketch_annotation.json"
glove_embedding_path = resource_dir + 'glove_embedding.npz'

kb_file = resource_dir + 'knowledge_base.npz'
object_merged = resource_dir + 'object_merged.json'
train_patt = dataset_dir + 'split/train_progs_path.txt'
test_patt = dataset_dir + 'split/test_progs_path.txt'


# General function
def add_end(tensor, value):
    return tensor + [value]


def add_front(tensor, value):
    return [value] + tensor


################################
# Program
################################

def get_compatibility_matrix(action_dict, object_dict, object_merged, kb_file):

    kb_content = np.load(kb_file)
    kb_actions = kb_content['action_names']
    kb_objects = kb_content['object_names']
    kb_action_dict = {
        x.lower().replace(' ', '').replace('_', ''): i \
        for i, x in enumerate(kb_actions)}
    kb_object_dict = {
        x.lower().replace(' ', '').replace('_', ''): i \
        for i, x in enumerate(kb_objects)}
    num_actions = len(action_dict.words)
    num_objects = len(object_dict.words)
    results = []

    with open(object_merged, 'r') as f:
        obj_merged = json.load(f)

    for name in ['actions_obj1', 'actions_obj2']:

        comp_kb = kb_content[name]
        comp_total = []

        # Build matrix #actions_kb, #objects_dataset
        for object_name in object_dict.words:
            # find all the equivalent object_names
            object_names = [object_name]
            if object_name in obj_merged.keys():
                object_names += obj_merged[object_name]
            object_names = [x.lower().replace(' ', '').replace('_', '') for x in object_names]
            ids_kb = [kb_object_dict[obj_name]
                      for obj_name in object_names if obj_name in kb_object_dict.keys()]
            if len(ids_kb) == 0:
                comp_total.append(np.zeros((comp_kb.shape[0], 1)))
            else:
                comp_total.append((np.sum(comp_kb[:, ids_kb], 1) > 0)[:, None])

        comp_kb_objects_dataset = np.concatenate(comp_total, 1)
        # matrix actions_dataset, objects_dataset
        cmatrix = np.zeros((num_actions, num_objects))

        for id_action_str, action_name in action_dict.idx2word.items():
            id_action = int(id_action_str)

            # <eos><sos><none>
            if int(id_action) < 3:
                cmatrix[id_action, id_action] = 1.
            else:
                if action_name == 'release':
                    id_kb = kb_action_dict['putobjback']
                else:
                    id_kb = kb_action_dict[action_name]
                cmatrix[id_action, :] = comp_kb_objects_dataset[id_kb, :]
                if cmatrix[id_action, :].sum() == 0:
                    cmatrix[id_action,
                            int(action_dict.word2idx['<none>'])] = 1.
        results.append(cmatrix)

    return results


def get_prog_dataset(args, train):

    programs = load_programs([augment_program_path,
                             program_path],
                             object_merged,
                             check_graph=True)

    print("Total available of programs:", len(programs))
    programs = sorted(programs, key=lambda k: k['title'])

    if train:
        action_dict, object_dict = get_program_dictionary(
            programs, object_merged, verbose=args.verbose)
        expand_dictionary_with_resources(
            object_dict,
            object_merged,
            resource_dir=resource_dir)
        word_dict = None
    else:
        # use the dictionary saved during training
        _dict_path = os.path.dirname(args.checkpoint)
        _dict_path = os.path.join(_dict_path, 'dict.json')
        fp = open(_dict_path, 'r')
        _dict = json.load(fp)
        action_dict = dictionary(init_dict=_dict['action_dict'])
        object_dict = dictionary(init_dict=_dict['object_dict'])
        word_dict = dictionary(init_dict=_dict['word_dict'])
        fp.close()

    # Obtain the compatibility matrix, telling which actions go with which
    # objects
    compatibility_matrices = get_compatibility_matrix(
        action_dict, object_dict, object_merged, kb_file)

    convert2idx_program(programs, object_merged, action_dict, object_dict)
    get_desc_tokens(programs)
    get_title_tokens(programs)

    train_dset = program_dset(
        programs,
        action_dict,
        object_dict,
        word_dict,
        is_train=True,
        max_length=args.max_words,
        compatibility_matrix=compatibility_matrices,
        sketch_path=args.sketch_path)

    # unseen activities + unseen graph or seen graph
    test_dset = program_dset(
        programs,
        action_dict,
        object_dict,
        word_dict,
        is_train=False,
        max_length=args.max_words,
        compatibility_matrix=compatibility_matrices,
        sketch_path=args.sketch_path)

    # print("------------{}---------------".format(len(test_dset)))

    return train_dset, test_dset

def pad_to_tensor(data_tuple):
    index, length = data_tuple[0], data_tuple[1]
    list_data = list(data_tuple)[2:]

    return_list = []

    index_tensor = torch.tensor(index)
    length_tensor = torch.tensor(length)

    return_list.append(index_tensor)
    return_list.append(length_tensor)

    for list_d in list_data:
        pad_tensor = pad_sequence(list_d, batch_first=True)
        return_list.append(pad_tensor)

    return return_list

def remove_padding(data_tuple):
    index, length = data_tuple[0], data_tuple[1]
    list_data = list(data_tuple)[2:]

    index = list(index.cpu().numpy())
    length = list(length.cpu().numpy())

    return_list = []

    return_list.append(index)
    return_list.append(length)

    for list_d in list_data:
        un_pad_list_d = [list_d[i][:length[i]] for i in range(len(length))]
        return_list.append(un_pad_list_d)

    return return_list

def prog_collate_fn(data_list):

    program_data = [data[0] for data in data_list]
    sketch_data = [data[1] for data in data_list]
    graph_data = [data[2] for data in data_list]
    path_data = [data[3] for data in data_list]
    desc_data = [data[4] for data in data_list]
    title_data = [data[5] for data in data_list]

    batch_sketch_length = [data[0] for data in sketch_data]
    batch_sketch_action = [data[1] for data in sketch_data]
    batch_sketch_object1 = [data[2] for data in sketch_data]
    batch_sketch_object2 = [data[3] for data in sketch_data]
    batch_sketch_index1 = [data[4] for data in sketch_data]
    batch_sketch_index2 = [data[5] for data in sketch_data]
    batch_sketch_index = np.arange(len(batch_sketch_length))

    batch_program_length = [data[0] for data in program_data]
    batch_program_action = [data[1] for data in program_data]
    batch_program_object1 = [data[2] for data in program_data]
    batch_program_object2 = [data[3] for data in program_data]
    batch_program_index = np.arange(len(batch_program_length))

    # the number of nodes might be different

    batch_node_names = [data[1] for data in graph_data]
    batch_init_graph = [data[4] for data in graph_data]
    batch_final_graph = [data[5] for data in graph_data]

    batch_file_path = [data[0] for data in path_data]
    batch_graph_path = [data[1] for data in path_data]

    batch_n_nodes = [len(node_name) for node_name in batch_node_names]
    N = max(batch_n_nodes)
    batch_adjacency_matrix = torch.stack(
        [F.pad(data[0], (0, N - n, 0, N - n)) for data, n in zip(graph_data, batch_n_nodes)])
    batch_node_category = torch.stack(
        [F.pad(data[2], (0, N - n)) for data, n in zip(graph_data, batch_n_nodes)])
    batch_node_states = torch.stack(
        [F.pad(data[3], (0, 0, 0, N - n)) for data, n in zip(graph_data, batch_n_nodes)])

    batch_desc_length = [data[0] for data in desc_data]
    batch_desc_words = [data[1] for data in desc_data]
    batch_desc_index = np.arange(len(batch_desc_length))

    batch_title_length = [data[0] for data in title_data]
    batch_title_words = [data[1] for data in title_data]
    batch_title_index = np.arange(len(batch_title_length))

    batch_sketch_data = (
        batch_sketch_index,
        batch_sketch_length,
        batch_sketch_action,
        batch_sketch_object1,
        batch_sketch_object2,
        batch_sketch_index1,
        batch_sketch_index2)
    batch_program_data = (
        batch_program_index,
        batch_program_length,
        batch_program_action,
        batch_program_object1,
        batch_program_object2)
    batch_graph_data = (
        batch_adjacency_matrix,
        batch_node_names,
        batch_node_category,
        batch_node_states,
        batch_init_graph,
        batch_final_graph)

    batch_path_data = (batch_file_path, batch_graph_path)
    batch_title_data = (batch_title_index, batch_title_length, batch_title_words)
    batch_desc_data = (batch_desc_index, batch_desc_length, batch_desc_words)

    batch_title_data = pad_to_tensor(batch_title_data)
    batch_desc_data = pad_to_tensor(batch_desc_data)
    batch_sketch_data = pad_to_tensor(batch_sketch_data)
    batch_program_data = pad_to_tensor(batch_program_data)

    batch_wise_data_idx = torch.arange(len(batch_node_names))

    return [
        batch_program_data,
        batch_sketch_data,
        batch_graph_data,
        batch_path_data,
        batch_desc_data,
        batch_title_data,
        batch_wise_data_idx]


def prog_to_cuda_fn(data):
    batch_program_data, batch_sketch_data, batch_graph_data, batch_path_data, batch_desc_data, batch_title_data, batch_wise_data_idx = data

    # recover to original length
    batch_desc_data = remove_padding(batch_desc_data)
    batch_sketch_data = remove_padding(batch_sketch_data)
    batch_title_data = remove_padding(batch_title_data)
    batch_program_data = remove_padding(batch_program_data)
    # batch_graph_data special dealing
    batch_wise_data_idx = list(batch_wise_data_idx.cpu().numpy())

    # batch_program_data, batch_sketch_data, batch_graph_data, batch_path_data, batch_desc_data, batch_title_data = data
    batch_sketch_index, batch_sketch_length, batch_sketch_action, batch_sketch_object1, batch_sketch_object2, batch_sketch_index1, batch_sketch_index2 = batch_sketch_data
    batch_program_index, batch_program_length, batch_program_action, batch_program_object1, batch_program_object2 = batch_program_data
    batch_adjacency_matrix, batch_node_names, batch_node_category, batch_node_states, batch_init_graph, batch_final_graph = batch_graph_data
    batch_desc_index, batch_desc_length, batch_desc_words = batch_desc_data
    batch_title_index, batch_title_length, batch_title_words = batch_title_data
    batch_file_path, batch_graph_path = batch_path_data

    batch_node_names = [batch_node_names[idx] for idx in range(len(batch_node_names)) if idx in batch_wise_data_idx]
    batch_init_graph = [batch_init_graph[idx] for idx in range(len(batch_init_graph)) if idx in batch_wise_data_idx]
    batch_final_graph = [batch_final_graph[idx] for idx in range(len(batch_final_graph)) if idx in batch_wise_data_idx]

    batch_file_path = [batch_file_path[idx] for idx in range(len(batch_file_path)) if idx in batch_wise_data_idx]
    batch_graph_path = [batch_graph_path[idx] for idx in range(len(batch_graph_path)) if idx in batch_wise_data_idx]

    batch_n_nodes = [len(node_name) for node_name in batch_node_names]
    N = max(batch_n_nodes)
    batch_adjacency_matrix = batch_adjacency_matrix[:, :, :N, :N]
    batch_node_category = batch_node_category[:, :N]
    batch_node_states = batch_node_states[:, :N, :]

    # to cuda
    batch_sketch_action = [i.cuda() for i in batch_sketch_action]
    batch_sketch_object1 = [i.cuda() for i in batch_sketch_object1]
    batch_sketch_object2 = [i.cuda() for i in batch_sketch_object2]
    batch_sketch_index1 = [i.cuda() for i in batch_sketch_index1]
    batch_sketch_index2 = [i.cuda() for i in batch_sketch_index2]
    batch_program_action = [i.cuda() for i in batch_program_action]
    batch_program_object1 = [i.cuda() for i in batch_program_object1]
    batch_program_object2 = [i.cuda() for i in batch_program_object2]
    batch_desc_words = [i.cuda() for i in batch_desc_words]
    batch_title_words = [i.cuda() for i in batch_title_words]
    batch_adjacency_matrix = batch_adjacency_matrix.cuda()
    batch_node_category = batch_node_category.cuda()
    batch_node_states = batch_node_states.cuda()

    batch_sketch_data = (
        batch_sketch_index,
        batch_sketch_length,
        batch_sketch_action,
        batch_sketch_object1,
        batch_sketch_object2,
        batch_sketch_index1,
        batch_sketch_index2)
    batch_program_data = (
        batch_program_index,
        batch_program_length,
        batch_program_action,
        batch_program_object1,
        batch_program_object2)
    batch_graph_data = (
        batch_adjacency_matrix,
        batch_node_names,
        batch_node_category,
        batch_node_states,
        batch_init_graph,
        batch_final_graph)
    batch_title_data = (batch_title_index, batch_title_length, batch_title_words)
    batch_desc_data = (batch_desc_index, batch_desc_length, batch_desc_words)
    batch_path_data = (batch_file_path, batch_graph_path)

    return [
        batch_program_data,
        batch_sketch_data,
        batch_graph_data,
        batch_path_data,
        batch_desc_data,
        batch_title_data]


class program_dset(Dataset):

    sketch_keys = [
        'sketch_action_list',
        'sketch_object1_list',
        'sketch_object2_list',
        'sketch_instance1_str_list',
        'sketch_instance2_str_list']

    def __init__(
            self,
            programs,
            action_dict,
            object_dict,
            word_dict,
            is_train,
            min_length=4,
            max_length=30,
            compatibility_matrix=None,
            sketch_path=None):
        """
            `sketch_path` only used when testing
        """

        programs = self._trim_programs(programs, min_length, max_length)

        # print("---------------{}-------------".format(len(programs)))

        self.action_dict = action_dict
        self.object_dict = object_dict
        self.max_indexes = 10
        self.index_dict = self._get_index_dict()
        self.num_indexes = self.index_dict.n_words
        self.compatibility_matrix = compatibility_matrix

        # if the sketch doesn't cover all the dataset,
        # use random sample for training set, and manually collected sketch for
        # testing set

        if sketch_path is not None:
            print(colored("Using predicted sketch", "red"))
            programs_w_sketch = self._add_sketch_from_predicted_sketch(programs, sketch_path)
        else:
            programs_w_sketch = self._add_sketch_from_real_sketch(programs)

        train_programs, test_programs = self._split_set(programs_w_sketch)

        word_dict = self._get_word_dict_and_embeddings(train_programs, word_dict)

        print("-" * 30)
        print("Train set size: {}, Test set size: {}".format(
            len(train_programs), len(test_programs)))
        # if the sketch cover all the dataset
        #train_programs, test_programs

        self.env_graph_helper = EnvGraphHelper()
        self.detail2abstract = load_detail2abstract(object_merged)
        self.programs = train_programs if is_train else test_programs

        convert2idx_desc(self.programs, word_dict)
        convert2idx_title(self.programs, word_dict)

        self.is_train = is_train
        self.random_index = is_train
        self.num_actions = action_dict.n_words
        self.num_objects = object_dict.n_words
        self.num_words = word_dict.n_words
        self.word_dict = word_dict
        self.initial_program = (int(action_dict.word2idx['<sos>']),
                                int(object_dict.word2idx['<sos>']),
                                int(object_dict.word2idx['<sos>']))
        
    def __getitem__(self, index):
        program = self.programs[index]
        data = self._preprocess_one_program(program)
        return data

    def __len__(self):
        return len(self.programs)

    def _get_index_dict(self):
        index_dict = dictionary()
        for i in range(self.max_indexes):
            index_dict.add_items(str(i))
        return index_dict

    def visualize_dict(self):
        self.action_dict.visualize()
        self.object_dict.visualize()

    def _trim_programs(self, programs, min_length, max_length):
        valid_programs = []
        for program in programs:
            if len(
                    program['action_list']) >= min_length and len(
                    program['action_list']) <= max_length:
                valid_programs.append(program)
        print("Trim out {:.2f}%  of program".format(
            100. * (len(programs) - len(valid_programs)) / len(programs)))
        return valid_programs

    def _split_set(self, programs):

        # read the split
        f = open(train_patt)
        train_split_patt = f.read()
        train_split_patt = [
            i for i in filter(
                lambda v: v != '',
                train_split_patt.split("\n"))]
        f.close()

        f = open(test_patt)
        test_split_patt = f.read()
        test_split_patt = [
            i for i in filter(
                lambda v: v != '',
                test_split_patt.split("\n"))]
        f.close()

        # split by unique programs
        unique_programs = []
        for program in programs:
            path_split = program["file_path"].split('/')
            temp_path = path_split[-1].split("\\")
            path_split.pop(-1)
            path_split += temp_path
            j = path_split.index('executable_programs')
            postfix = '/'.join(path_split[(j + 2):] \
                    if len(path_split[(j + 2):]) == 2 else path_split[(j + 2):-1])
            postfix = postfix.replace('.txt', '')
            unique_programs.append(postfix)

        unique_programs = list(set(unique_programs))

        # split by scenes and the unique programs
        train_programs, test_programs = [], []
        for program in programs:
            path_split = program["file_path"].split('/')
            temp_path = path_split[-1].split("\\")
            path_split.pop(-1)
            path_split += temp_path
            j = path_split.index('executable_programs')
            postfix = '/'.join(path_split[(j + 2):] \
                    if len(path_split[(j + 2):]) == 2 else path_split[(j + 2):-1])
            postfix = postfix.replace('.txt', '')

            if postfix in train_split_patt and "TrimmedTestScene7" not in program["file_path"]:
                train_programs.append(program)
            elif postfix in test_split_patt:
                if "TrimmedTestScene7" in program["file_path"]:
                    test_programs.append(program)

        return train_programs, test_programs

    def _add_sketch_from_real_sketch(self, programs):

        sketch_annotation_data = json.load(open(sketch_path))

        program_with_collected_sketch = {}
        for answer, input in zip(
                sketch_annotation_data['answer'], sketch_annotation_data['input']):

            prefix = input["prefix"]
            valid = sum(answer) != 0
            if valid:
                program_with_collected_sketch[prefix] = answer

        # create dictionary with key: results_intentions_march-13-18/file463_2 (parent program name)
        #                      value: [?, ?], [?, ?], [?, ?], [?, ?], [?, ?]
        sketch_annotation = {}

        num_counter = {}
        for program in programs:
            if 'original_programs' in program["file_path"]:
                path_split = program["file_path"].split('/')
                prefix = '/'.join(path_split[-2:])
                prefix = prefix.replace('.txt', '')
                if prefix in program_with_collected_sketch and prefix not in sketch_annotation:
                    answer = program_with_collected_sketch[prefix]
                    answer_idx = np.where(answer)[0]
                    assert len(answer) == len(program['action_list']), print(
                        "The sketch is not aligned with the program")

                    sketch_annotation[prefix] = []
                    for key in self.sketch_keys:
                        value_list = [program[key.replace(
                            'sketch_', '')][idx] for idx in answer_idx]
                        sketch_annotation[prefix].append(value_list)

                    # 统计各个title的sketch数量
                    title = program['title']
                    if title not in num_counter:
                        num_counter[title] = []
                    sketch_programs = np.array(program['program_list'])[list(answer_idx)]
                    num_counter[title].append(sketch_programs)

        # with open("sketch_counter.pkl", "wb") as f:
        #     pkl.dump(num_counter, f)

        programs_w_sketch = []
        programs_wo_sketch = 0
        prefix_list = {}
        for program in programs:
            path_split = program["file_path"].split('/')
            temp_rest = path_split[-1].split("\\")
            path_split.pop(-1)
            path_split += temp_rest

            j = path_split.index('executable_programs')
            prefix = '/'.join(path_split[(j + 2):] \
                                  if len(path_split[(j + 2):]) == 2 else path_split[(j + 2):-1])
            prefix = prefix.replace('.txt', '')

            if prefix in sketch_annotation:
                if prefix not in prefix_list.keys():
                    prefix_list[prefix] = []
                prefix_list[prefix].append(program["file_path"])
                # with sketh annotated
                for n, key in enumerate(self.sketch_keys):
                    program[key] = sketch_annotation[prefix][n]
                programs_w_sketch.append(program)
            else:
                programs_wo_sketch += 1

        print("-" * 30)
        print("Total programs with sketch annotation: {}".format(
            len(programs_w_sketch)))

        return programs_w_sketch

    def _add_sketch_from_predicted_sketch(self, programs, sketch_path):
        
        patt_action = r'\[(.+?)\]'
        patt_object1= r'\] \<(.+?)\>'
        patt_object2 = r'\> \<(.+?)\>'
        sketch_annotation_data = json.load(open(sketch_path))
        sketch_annotation_data = sketch_annotation_data['programs']

        sketch_annotation = {}
        for sketch_annotation_data_i in sketch_annotation_data:

            prefix = sketch_annotation_data_i['path']
            # [watch] <broom> <shaver>
            prediction = sketch_annotation_data_i['pred']

            # action
            prediction = prediction.replace('[<sos>]', '[NONE]')
            prediction = prediction.replace('[<eos>]', '[NONE]')
            prediction = prediction.replace('[<none>]', '[NONE]')

            # object1
            prediction = prediction.replace('] <<sos>> <', '] <NONE> <')
            prediction = prediction.replace('] <<eos>> <', '] <NONE> <')
            prediction = prediction.replace('] <<none>> <', '] <NONE> <')

            # object2
            prediction = prediction.replace('> <<sos>>', '> <NONE>')
            prediction = prediction.replace('> <<eos>>', '> <NONE>')
            prediction = prediction.replace('> <<none>>', '> <NONE>')

            
            # extract actions
            sketch_action_str = []
            action_match = re.search(patt_action, prediction)
            while action_match:
                sketch_action_str.append(action_match.group(1))
                action_match = re.search(
                    patt_action, action_match.string[action_match.end(1):])

            sketch_object1_str = []
            object1_match = re.search(patt_object1, prediction)
            while object1_match:
                sketch_object1_str.append(object1_match.group(1))
                object1_match = re.search(
                    patt_object1, object1_match.string[object1_match.end(1):])

            sketch_object2_str = []
            object2_match = re.search(patt_object2, prediction)
            while object2_match:
                sketch_object2_str.append(object2_match.group(1))
                object2_match = re.search(
                    patt_object2, object2_match.string[object2_match.end(1):])

            assert len(sketch_action_str) == len(sketch_object1_str)
            assert len(sketch_action_str) == len(sketch_object2_str)

            sketch_action_str = [
                elem if elem != 'NONE' else '<none>' for elem in sketch_action_str]
            sketch_object1_str = [
                elem if elem != 'NONE' else '<none>' for elem in sketch_object1_str]
            sketch_object2_str = [
                elem if elem != 'NONE' else '<none>' for elem in sketch_object2_str]

            sektch_action = self.action_dict.convert_word2idx(
                sketch_action_str)
            sketch_object1 = self.object_dict.convert_word2idx(
                sketch_object1_str)
            sketch_object2 = self.object_dict.convert_word2idx(
                sketch_object2_str)

            sketch_instance1_str_list = [
                '1' if elem != 'NONE' else '<none>' for elem in sketch_object1_str]
            sketch_instance2_str_list = [
                '1' if elem != 'NONE' else '<none>' for elem in sketch_object2_str]
            sketch_annotation[prefix] = [
                sektch_action,
                sketch_object1,
                sketch_object2,
                sketch_instance1_str_list,
                sketch_instance2_str_list]

        programs_w_sketch = []
        programs_wo_sketch = 0
        for program in programs:
            path_split = program["file_path"].split('/')
            temp_path = path_split[-1].split("\\")
            path_split.pop(-1)
            path_split += temp_path
            j = path_split.index('executable_programs')
            prefix = '/'.join(path_split[(j + 2):] \
                    if len(path_split[(j + 2):]) == 2 else path_split[(j + 2):-1])
            prefix = prefix.replace('.txt', '')

            if prefix in sketch_annotation:
                # with sketh annotated
                for n, key in enumerate(program_dset.sketch_keys):
                    program[key] = sketch_annotation[prefix][n]
                programs_w_sketch.append(program)
            else:
                programs_wo_sketch += 1

        print("-" * 30)
        print("{:.2f}% of programs do not have sketch label".format(
            programs_wo_sketch / len(programs) * 100.))
        print(
            "Total programs with sketch annotation: {}".format(
                len(programs_w_sketch)))
        return programs_w_sketch

    def interpret_desc(self, batch_desc_words,  with_sos=True):

        start_idx = 1 if with_sos else 0
        descs_str = []
        for desc_words in batch_desc_words:
            desc_words = self.word_dict.convert_idx2word(np.array(desc_words))

            desc_str = []
            for word in desc_words[start_idx:]:
                if word == "<eos>":
                    break
                desc_str.append(word)
            
            desc_str = ' '.join(desc_str)
            descs_str.append(desc_str)

        return descs_str

    def interpret(
            self,
            batch_action,
            batch_object1,
            batch_object2,
            batch_node_name_list,
            with_sos=True):

        start_idx = 1 if with_sos else 0
        programs_str = []
        for program_action, program_object1, program_object2, program_node_name_list in zip(
                batch_action, batch_object1, batch_object2, batch_node_name_list):
            program_action = self.action_dict.convert_idx2word(
                np.array(program_action))
            program_object1 = [program_node_name_list[i] for i in program_object1]
            program_object2 = [program_node_name_list[i] for i in program_object2]

            program_str = []
            for action, object1, object2 in zip(
                    program_action[start_idx:], program_object1[start_idx:], program_object2[start_idx:]):
                object1, number1 = object1.split('.')
                object2, number2 = object2.split('.')

                program = '[{}] <{}> ({}) <{}> ({})'.format(
                    action, object1, number1, object2, number2)

                if self._is_terminate(
                        action,
                        object1,
                        number1,
                        object2,
                        number2):
                    break
                    
                program_str.append(program)

            program_str = ', '.join(program_str)
            programs_str.append(program_str)

        return programs_str

    def interpret_sketch(
            self,
            batch_sketch_action,
            atch_sketch_object1,
            batch_sketch_object2,
            batch_sketch_index1,
            batch_sketch_index2,
            with_sos=True,
            end_with_eos = True):

        start_idx = 1 if with_sos else 0
        programs_str = []
        for sketch_action, sketch_object1, sketch_object2, sketch_index1, sketch_index2 in zip(
                batch_sketch_action, atch_sketch_object1, batch_sketch_object2, 
                batch_sketch_index1, batch_sketch_index2):

            sketch_action = self.action_dict.convert_idx2word(
                np.array(sketch_action))
            sketch_object1 = self.object_dict.convert_idx2word(
                np.array(sketch_object1))
            sketch_object2 = self.object_dict.convert_idx2word(
                np.array(sketch_object2))
            sketch_index1 = self.index_dict.convert_idx2word(
                np.array(sketch_index1))
            sketch_index2 = self.index_dict.convert_idx2word(
                np.array(sketch_index2))

            program_str = []
            for action, object1, object2, index1, index2 in zip(
                    sketch_action[start_idx:], sketch_object1[start_idx:], sketch_object2[start_idx:], 
                    sketch_index1[start_idx:], sketch_index2[start_idx:]):

                program = '[{}] <{}> ({}) <{}> ({})'.format(
                    action, object1, index1, object2, index2)

                if self._is_terminate(action, object1, index1, object2, index2) and end_with_eos:
                    break
                program_str.append(program)

            program_str = ', '.join(program_str)
            programs_str.append(program_str)

        return programs_str

    def _is_terminate_woid(self, action, object1, object2):
        # terminate when action is eos
        if action == '<eos>':
            return True
        else:
            return False

    def interpret_sketch_woid(
            self,
            batch_action,
            batch_object1,
            batch_object2,
            with_sos=True,
            end_with_eos=True):

        start_idx = 1 if with_sos else 0
        sketchs_str = []
        for sketch_action, sketch_object1, sketch_object2 in zip(
                batch_action, batch_object1, batch_object2):
            sketch_action = self.action_dict.convert_idx2word(
                np.array(sketch_action))
            sketch_object1 = self.object_dict.convert_idx2word(
                np.array(sketch_object1))
            sketch_object2 = self.object_dict.convert_idx2word(
                np.array(sketch_object2))

            sketch_str = []
            for action, object1, object2 in zip(
                    sketch_action[start_idx:], sketch_object1[start_idx:], sketch_object2[start_idx:]):

                sketch = '[{}] <{}> <{}>'.format(action, object1, object2)

                if self._is_terminate_woid(action, object1, object2) and end_with_eos:
                    break
                sketch_str.append(sketch)

            sketch_str = ', '.join(sketch_str)
            sketchs_str.append(sketch_str)

        return sketchs_str

    def interpreter(self, query_idx_list, action=False):
        if action == True:
            output = self.action_dict.convert_idx2word(np.array(query_idx_list))
        else:
            output = self.object_dict.convert_idx2word(np.array(query_idx_list))

        return output

    def _is_terminate(self, action, object1, number1, object2, number2):
        # terminate when action is eos
        if object1 == '<eos>' or action == '<eos>':
            return True
        else:
            return False

    def preprocess_one_graph(self, graph):
        return self.env_graph_helper.process_one_graph(graph, detail2abstract=self.detail2abstract)

    def _preprocess_one_program(self, program):

        # load graphs
        graph_path = convert_to_graph_path(program['file_path'])  # init_and_final_graphs

        # adjacent_matrix : 邻接矩阵，大小为 7 × Node_num × Node_num，7表示7中边的类型
        adjacency_matrix, node_names, node_category, node_states, init_graph, final_graph = \
            self.env_graph_helper.load_one_graph(
                graph_path, detail2abstract=self.detail2abstract)
        program_object1, program_object2 = add_ground_truth(
            program, node_names)

        # program_data
        program_action = program['action_list']

        # sample sketch
        sketch_length, sketch_action, sketch_fake_node1_name, sketch_fake_node2_name = \
            self._sample_one_program_w_annotation(program)

        eos_idx = int(self.action_dict.word2idx["<eos>"])
        sos_idx = int(self.action_dict.word2idx["<sos>"])
        program_action = add_end(add_front(program_action, sos_idx), eos_idx)
        program_length = len(program_action)
        sketch_action = add_end(add_front(sketch_action, sos_idx), eos_idx)
        sketch_length = len(sketch_action)

        for j, node_name in enumerate(node_names):
            if 'character' in node_name:
                character_idx = j

        for j, node_name in enumerate(node_names):
            if 'eos' in node_name:
                end_token_idx = j

        program_object1 = add_end(
            add_front(program_object1, character_idx), end_token_idx)
        program_object2 = add_end(
            add_front(program_object2, character_idx), end_token_idx)

        # get the sketch object and index within the program
        sketch_instance1, sketch_object1 = self._convert_fake_node_idx_to_node_name(
            sketch_fake_node1_name)
        sketch_index1 = self._get_index_within_program(sketch_instance1, sketch_object1, random=self.random_index)
        sketch_instance2, sketch_object2 = self._convert_fake_node_idx_to_node_name(sketch_fake_node2_name)
        sketch_index2 = self._get_index_within_program(sketch_instance2, sketch_object2, random=self.random_index)

        sketch_object1 = add_end(
            add_front(sketch_object1, int(self.object_dict.word2idx['<sos>'])), 
            int(self.object_dict.word2idx['<eos>']))
        sketch_object2 = add_end(
            add_front(sketch_object2, int(self.object_dict.word2idx['<sos>'])), 
            int(self.object_dict.word2idx['<eos>']))
        sketch_index1 = add_end(
            add_front(sketch_index1, int(self.index_dict.word2idx['<sos>'])), 
            int(self.index_dict.word2idx['<eos>']))
        sketch_index2 = add_end(
            add_front(sketch_index2, int(self.index_dict.word2idx['<sos>'])), 
            int(self.index_dict.word2idx['<eos>']))

        desc_words = program["desc_words_idx"]
        desc_words = add_end(add_front(desc_words, int(self.word_dict.word2idx['<sos>'])), int(self.word_dict.word2idx['<eos>']))
        desc_length = len(desc_words)

        title_words = program["title_words_idx"]
        title_words = add_end(add_front(title_words, int(self.word_dict.word2idx['<sos>'])), int(self.word_dict.word2idx['<eos>']))
        title_length = len(title_words)

        # to tensor
        program_action = torch.tensor(program_action, dtype=torch.int64)
        program_object1 = torch.tensor(program_object1, dtype=torch.int64)
        program_object2 = torch.tensor(program_object2, dtype=torch.int64)
        sketch_action = torch.tensor(sketch_action, dtype=torch.int64)
        sketch_object1 = torch.tensor(sketch_object1, dtype=torch.int64)
        sketch_object2 = torch.tensor(sketch_object2, dtype=torch.int64)
        sketch_index1 = torch.tensor(sketch_index1, dtype=torch.int64)
        sketch_index2 = torch.tensor(sketch_index2, dtype=torch.int64)
        adjacency_matrix = torch.tensor(adjacency_matrix.astype(np.int32))
        node_states = torch.tensor(node_states.astype(np.int32))
        node_category = torch.tensor(np.array(node_category).astype(np.int32))
        desc_words = torch.tensor(desc_words, dtype=torch.int64)
        title_words = torch.tensor(title_words, dtype=torch.int64)

        program_data = (
            program_length,
            program_action,
            program_object1,
            program_object2)
        sketch_data = (
            sketch_length,
            sketch_action,
            sketch_object1,
            sketch_object2,
            sketch_index1,
            sketch_index2)
        graph_data = (
            adjacency_matrix,
            node_names,
            node_category,
            node_states,
            init_graph,
            final_graph)
        desc_data = (desc_length, desc_words)
        title_data = (title_length, title_words)
        path_data = (program['file_path'], graph_path)

        data = [program_data, sketch_data, graph_data, path_data, desc_data, title_data]
        return data

    def _convert_fake_node_idx_to_node_name(self, fake_node_names):

        object_list = []
        instance_list = []
        for node_name in fake_node_names:
            #obj_instance = node_names[obj]
            # instance_list.append(obj_instance)
            name, id = node_name.split('.')
            name = int(self.object_dict.word2idx[name])
            object_list.append(name)
            instance_list.append(node_name)

        return instance_list, object_list

    def _get_index_within_program(self, instance_list, object_list, random=True):

        objs_id2index = {i: {"available_index": np.arange(
            self.max_indexes)} for i in object_list}
        index_list = []

        for obj_instance in instance_list:
            name, id = obj_instance.split('.')
            if '<none>' in name:
                index_list.append(int(self.index_dict.word2idx["<none>"]))
            else:
                name = int(self.object_dict.word2idx[name])
                id2index = objs_id2index[name]

                if id not in id2index:
                    if random:
                        index = np.random.choice(id2index["available_index"])
                    else:
                        index = np.min(id2index["available_index"])

                    i_to_removed = np.where(
                        id2index["available_index"] == index)[0][0]
                    id2index["available_index"] = np.delete(
                        id2index["available_index"], i_to_removed)
                    id2index[id] = index
                else:
                    index = id2index[id]

                index = int(self.index_dict.word2idx[str(index)])
                index_list.append(index)
                
        return index_list

    def _sample_one_program_w_annotation(self, program):

        sample_action = program['sketch_action_list']
        sample_object1_str = [self.object_dict.idx2word[str(
            i)] for i in program['sketch_object1_list']]
        sample_object2_str = [self.object_dict.idx2word[str(
            i)] for i in program['sketch_object2_list']]
        sample_instance1_str = program['sketch_instance1_str_list']
        sample_instance2_str = program['sketch_instance2_str_list']

        sample_fake_node_name1 = [
            '{}.{}'.format(
                object, instance) for object, instance in zip(
                sample_object1_str, sample_instance1_str)]
        sample_fake_node_name2 = [
            '{}.{}'.format(
                object, instance) for object, instance in zip(
                sample_object2_str, sample_instance2_str)]
        sample_length = len(sample_action)

        return sample_length, sample_action, sample_fake_node_name1, sample_fake_node_name2

    def _get_word_dict_and_embeddings(self, train_programs, word_dict):

        glove_embedding = np.load(glove_embedding_path)
        glove_embedding_words = [i for i in glove_embedding["words"]]
        if word_dict is None:
            word_dict = dictionary(no_special_words=True)

            words = get_words(train_programs)
            # re-order
            words = [w for w in filter(lambda w: w in glove_embedding_words, words)] + \
                [w for w in filter(lambda w: w not in glove_embedding_words, words)]

            for w in words:
                word_dict.add_items(w)
            
            word_dict.add_items('<unk>')
            word_dict.add_items('<none>')
            word_dict.add_items('<sos>')
            word_dict.add_items('<eos>')

        idx = []
        for w in word_dict.words:
            if w in glove_embedding_words:
                idx.append(glove_embedding_words.index(w))

        self.glove_embedding_vector = np.array([glove_embedding['vectors'][i].astype(np.float32) for i in idx])
        return word_dict

    def _trim_programs_with_too_many_words(self, programs, max_length):

        valid_program = []
        for program in programs:
            if len(program['desc_words']) >= 3 and \
                len(program['desc_words']) <= max_length and \
                len(program['title_words']) >= 0 and \
                len(program['title_words']) <= 10:
                valid_program.append(program)

        return valid_program
