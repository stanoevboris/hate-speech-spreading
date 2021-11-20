from abc import ABC
import dgl
import torch as th
from torch.utils.data import Dataset, DataLoader
from scripts.embeddings import tokenize

DATASET_PATH = 'graphs/heterograph.bin'


class BaseDataset(ABC):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.meta_paths = None
        self.meta_path_dict = None


class NodeClassificationDataset(BaseDataset):
    r"""
    Description
    -----------
    The class *NodeClassificationDataset* is a base class for datasets which can be used in task *node classification*.
    """

    def __init__(self):
        super(NodeClassificationDataset, self).__init__()
        self.g = None
        self.category = None
        self.num_classes = None
        # self.has_feature = False
        self.multi_label = False
        self.in_dim = None

    def get_labels(self):
        raise NotImplemented

    def get_idx(self, ):
        raise NotImplemented


class HinNodeClassification(NodeClassificationDataset):
    def __init__(self, dataset):
        super(HinNodeClassification, self).__init__()
        self.g, self.category, self.num_classes, self.in_dim = self.load_graph(dataset)

    @staticmethod
    def load_graph(dataset):
        graph, _ = dgl.load_graphs(dataset)
        # Cast the graph to one with idtype int64
        graph = graph[0].long()
        category = 'user'
        num_classes = 2
        in_dim = graph.ndata['h']['user'].shape[1]

        return graph, category, num_classes, in_dim

    def get_idx(self, validation=True):
        if 'train_mask' not in self.g.nodes[self.category].data:
            num_nodes = self.g.number_of_nodes(self.category)
            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test

            train, test = th.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = th.tensor(train.indices)
            test_idx = th.tensor(test.indices)
            if validation:
                random_int = th.randperm(len(train_idx))
                val_idx = train_idx[random_int[:len(train_idx) // 10]]
                train_idx = train_idx[random_int[len(train_idx) // 10:]]
            else:
                val_idx = train_idx
                train_idx = train_idx
        else:
            train_mask = self.g.nodes[self.category].data.pop('train_mask')
            test_mask = self.g.nodes[self.category].data.pop('test_mask')
            train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            if validation:
                if 'val_mask' in self.g.nodes[self.category].data:
                    val_mask = self.g.nodes[self.category].data.pop('val_mask')
                    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
                    pass
                else:
                    random_int = th.randperm(len(train_idx))
                    val_idx = train_idx[random_int[:len(train_idx) // 10]]
                    train_idx = train_idx[random_int[len(train_idx) // 10:]]
            else:
                val_idx = train_idx
                train_idx = train_idx
        return train_idx, val_idx, test_idx

    def get_labels(self, validation=True):
        if 'labels' in self.g.nodes[self.category].data:
            labels = self.g.nodes[self.category].data.pop('labels').long()
        else:
            raise ValueError('labels in not in the g.nodes[category].data')
        return labels


class HSDataset(Dataset):
    def __init__(self, tweets, targets, tokenizer, max_len):
        self.tweets = tweets
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        user_tweets = self.tweets[item]
        target = self.targets[item]

        input_ids, attention_mask = tokenize(self.tokenizer, user_tweets, self.max_len)

        return {
            'user_tweets': str(user_tweets),
            'input_ids': input_ids.flatten(),
            'attention_mask': attention_mask.flatten(),
            'targets': th.tensor(target, dtype=th.float)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = HSDataset(
        tweets=df['RAW_TWEET'].to_numpy(),
        targets=df['LABEL'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len)

    return DataLoader(
        ds,
        batch_size=batch_size
    )
