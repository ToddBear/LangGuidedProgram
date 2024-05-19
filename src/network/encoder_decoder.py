import copy
import numpy as np
from termcolor import colored
import torch
import torch.nn as nn

def _align_tensor_index(reference_index, tensor_index):

    where_in_tensor = []
    for i in reference_index:
        where = np.where(i == tensor_index)[0][0]
        where_in_tensor.append(where)
    return np.array(where_in_tensor)

def _sort_by_length(list_of_tensor, batch_length, return_idx=False):
    idx = np.argsort(np.array(copy.copy(batch_length)))[::-1]
    for i, tensor in enumerate(list_of_tensor):
        list_of_tensor[i] = [tensor[j] for j in idx]
    if return_idx:
        return list_of_tensor, idx
    else:
        return list_of_tensor

def _sort_by_index(list_of_tensor, idx):
    for i, tensor in enumerate(list_of_tensor):
        list_of_tensor[i] = [tensor[j] for j in idx]
    return list_of_tensor


class Desc2ProgramGeo(nn.Module):

    summary_keys = ['action_loss', 'object1_loss', 'object2_loss', 'total_loss',
                    'action_accuracy', 'object1_accuracy', 'object2_accuracy',
                    'attribute_precision', 'relation_precision', 'total_precision',
                    'attribute_recall', 'relation_recall', 'total_recall',
                    'attribute_f1', 'relation_f1', 'total_f1',
                    'lcs_score', 'parsibility', 'executability', 'schedule_sampling_p']

    def __init__(self, dset, **kwargs):

        from network.module import ProgramGraphClassifierGeo
        from network.graph_module import VanillaGraphEncoderGeo
        from program.graph_utils import EnvGraphHelper
        from network.module import WordEncoder
        from helper import fc_block 
        super(Desc2ProgramGeo, self).__init__()

        embedding_dim = kwargs["embedding_dim"]
        sketch_hidden = kwargs["sketch_hidden"]
        program_hidden = kwargs["program_hidden"]
        max_words = kwargs["max_words"]
        graph_hidden = kwargs["graph_hidden"]

        # initial the graph encoding as well as graph update netowrk
        env_graph_helper = EnvGraphHelper()
        env_graph_encoder = VanillaGraphEncoderGeo(
            2,
            env_graph_helper.n_edge,
            graph_hidden,
            embedding_dim,
            residual=False,
            heads=[4,4],
            negative_slope=0.05,
            alpha=0.05,
            feat_drop=0.0,
            attn_drop=0.2,
            task_hidden=sketch_hidden)

        prob_env_graph_helper = EnvGraphHelper()
        prob_env_graph_encoder = VanillaGraphEncoderGeo(
            1,
            env_graph_helper.n_edge,
            graph_hidden,
            embedding_dim,
            residual=False,
            heads=[4,4],
            negative_slope=0.05,
            alpha=0.05,
            feat_drop=0.0,
            attn_drop=0.2,
            task_hidden=sketch_hidden)

        # init the submodules
        from helper import CombinedEmbedding
        num_words = dset.num_words
        num_pretrained_words = dset.glove_embedding_vector.shape[0]
        pretrained_word_embedding = nn.Embedding.from_pretrained(torch.tensor(dset.glove_embedding_vector))
        word_embedding = nn.Embedding(num_words - num_pretrained_words, embedding_dim)
        word_embedding = CombinedEmbedding(pretrained_word_embedding, word_embedding)
        desc_encoder = WordEncoder(dset, embedding_dim, sketch_hidden, word_embedding)

        # set up the graph update module
        kwargs = {}
        from program.graph_utils import InteractionGraphHelperGeo
        from network.graph_module import ResidualActionGraphEncoder as ActionGraphEncoder
        interaction_grpah_helper = InteractionGraphHelperGeo(dset.action_dict)
        interaction_graph_encoder = ActionGraphEncoder(interaction_grpah_helper.n_edge, interaction_grpah_helper.n_touch, graph_hidden, embedding_dim, sketch_hidden)
        kwargs.update({"interaction_grpah_helper": interaction_grpah_helper})
        kwargs.update({"interaction_graph_encoder": interaction_graph_encoder})

        # set up the network
        large_model_name = "gpt2"
        kwargs.update({"lora_dropout": 0.10})
        program_decoder = ProgramGraphClassifierGeo(large_model_name, dset, prob_env_graph_encoder, env_graph_encoder, env_graph_helper, sketch_hidden, embedding_dim, program_hidden, max_words, **kwargs)

        # for quick save and load
        all_modules = nn.Sequential()
        all_modules.add_module('desc_encoder', desc_encoder)
        all_modules.add_module('program_decoder', program_decoder)
        self.initial_program = dset.initial_program
        self.desc_encoder = desc_encoder
        self.program_decoder = program_decoder
        self.all_modules = all_modules
        self.to_cuda_fn = None
        self.dset = dset
        self.gt_link_collector = None

    def set_to_cuda_fn(self, to_cuda_fn):
        self.to_cuda_fn = to_cuda_fn

    def set_gt_link(self, gt_link_collector):
        self.gt_link_collector = gt_link_collector

    def forward(self, data, inference, **kwargs):
        if self.to_cuda_fn:
            data = self.to_cuda_fn(data)

        batch_sketch_length = data[1][1]

        # sort according to the length of sketch
        program = _sort_by_length(list(data[0]), batch_sketch_length)
        graph = _sort_by_length(list(data[2]), batch_sketch_length)
        desc = _sort_by_length(list(data[4]), batch_sketch_length)
        path_data = _sort_by_length(list(data[3]), batch_sketch_length)

        batch_path_data = path_data
        file_paths = batch_path_data[0]
        graph_paths = batch_path_data[1]
        data_idxes = [file_path.replace("\\", "/").split('/').index("Data")+2 for file_path in file_paths]
        trunc_graph_paths = ["/".join(graph_path.replace("\\", "/").split('/')[data_idxes[g_i]:]) for g_i, graph_path in enumerate(graph_paths)]

        # desc encoder
        batch_data = desc[1:]
        word_data = [d.cpu().numpy() for d in batch_data[1]]
        desc_NL = self.dset.interpret_desc(word_data)
        desc_NL = ["Description:" + desc_NL[i] for i in range(len(desc_NL))]

        batch_data = desc[1:]
        desc_emb, _ = self.desc_encoder(batch_data)

        # program decoder
        batch_program_length = program[1]
        batch_data, sort_idx = _sort_by_length(program, batch_program_length, return_idx=True)
        desc_emb = torch.stack(_sort_by_index([desc_emb], sort_idx)[0])
        desc_NL = _sort_by_index([desc_NL], sort_idx)[0]
        trunc_graph_paths = _sort_by_index([trunc_graph_paths], sort_idx)[0]
        batch_program_index = np.array(batch_data[0])
        batch_data = batch_data[1:]

        if not inference:
            character_index_list = torch.stack([self.gt_link_collector[path]["character_id"] for path in trunc_graph_paths], dim=0).to(desc_emb).long()
            target_topology_collector = [self.gt_link_collector[path]["gt_link"] for path in trunc_graph_paths]
        else:
            character_index_list, target_topology_collector = None, None

        graph = _sort_by_index(graph, sort_idx)
        kwargs.update({"initial_program": self.initial_program})
        kwargs.update({"graph": graph})
        kwargs.update({"desc_emb": desc_emb})
        kwargs.update({"input_sent": desc_NL})
        kwargs.update({"data": batch_data})
        kwargs.update({"character_index_list": character_index_list})
        kwargs.update({"target_topology_collector": target_topology_collector})

        loss, info = self.program_decoder(inference=inference, **kwargs)

        # align the output with the input `data`
        where_in_tensor = _align_tensor_index(data[0][0], batch_program_index)
        for k in info.keys():
            if k in ['batch_node_name_list', 'action_predict', 'object1_predict', 'object2_predict',
                     'action_loss_per_program', 'object1_loss_per_program', 'object2_loss_per_program',
                     'batch_node_state', 'action_graph_predict', 'object1_graph_predict', 'object2_graph_predict']:
                info[k] = [info[k][i] for i in where_in_tensor]

        program_pred = [info['action_graph_predict'], info['object1_graph_predict'],
                        info['object2_graph_predict'], info['batch_node_name_list']]

        if inference:
            return loss, program_pred, info
        else:
            return loss

    def write_summary(self, writer, info, postfix):

        model_name = 'Desc2Program-{}/'.format(postfix)
        for k in self.summary_keys:
            if k in info.keys():
                writer.scalar_summary(model_name + k, info[k])

    def save(self, path, verbose=False):

        if verbose:
            print(colored('[*] Save model at {}'.format(path), 'magenta'))
        torch.save(self.all_modules.state_dict(), path)

    def load(self, path, verbose=False):

        if verbose:
            print(colored('[*] Load model at {}'.format(path), 'magenta'))
        self.all_modules.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))