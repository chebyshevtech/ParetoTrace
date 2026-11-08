"""Candidate verified-raw QM9 loader; ignores all processed caches.

Frozen staging loader remains qm9_data.py. Reads only the hash-verified official
raw tensor dictionaries, reproducing PyG's no-RDKit conversion without a cache.
"""
import hashlib
import json
from pathlib import Path
import zipfile
import torch
from types import SimpleNamespace
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import Distance

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(__file__).parent / 'qm9_upstream'
COMMIT = '4336804847eaa5e0b924b743d76beec7ac3fdc97'
SPLIT_SHA256 = 'cccf1a47c31eae0d262fd2ddd9c009ed2cd070053376ef521c522d9b2d69a7a7'
ARCHIVE_SHA256 = '21feb47bd6bd8fb4afb8644a354a983f5c4ed25a0cdc79781da2f82c7026bc07'
RAW_SHA256 = '90052e9288b669cc41ecf4899b28ff99e1082e47f2c05eccfb1899572524d721'
TARGETS = [0, 1, 2, 3, 5, 6, 12, 13, 14, 15, 11]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def complete(data):
    n = data.num_nodes
    row = torch.arange(n).repeat_interleave(n)
    col = torch.arange(n).repeat(n)
    attrs = data.edge_attr.new_zeros(n*n, data.edge_attr.shape[1])
    attrs[data.edge_index[0]*n + data.edge_index[1]] = data.edge_attr
    keep = row != col
    data.edge_index = torch.stack([row[keep], col[keep]])
    data.edge_attr = attrs[keep]
    return Distance(norm=False)(data)


class VerifiedRawQM9:
    """Read-only source dictionaries; callers receive independent graph objects."""
    def __init__(self, raw_path):
        self._records = torch.load(raw_path, map_location='cpu', weights_only=True)
        if not isinstance(self._records, list) or len(self._records) != 130831:
            raise ValueError('unexpected pinned raw QM9 record layout')
        self._data = SimpleNamespace(y=torch.cat([record['y'] for record in self._records], dim=0))
        if self._data.y.shape != (130831, 19):
            raise ValueError('unexpected pinned raw QM9 label shape')

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        # Match PyG's Data(**data_dict) conversion. No transform/filter/reorder.
        return Data(**self._records[int(index)]).clone()


def load_real(root=ROOT / 'data/qm9', *, prepare=False):
    root = Path(root)
    raw = root / 'raw/qm9_v3.pt'
    archive = root / 'qm9_v3.zip'
    if not raw.exists():
        if not prepare or not archive.exists():
            raise FileNotFoundError(f'Real QM9 required: download https://data.pyg.org/datasets/qm9_v3.zip '
                                    f'to {archive}, then use --prepare. No random fallback.')
        if sha256(archive) != ARCHIVE_SHA256:
            raise ValueError('official QM9 archive SHA256 mismatch')
        with zipfile.ZipFile(archive) as zf:
            if 'qm9_v3.pt' not in zf.namelist():
                raise ValueError('unexpected official archive layout')
            zf.extract('qm9_v3.pt', root / 'raw')
    raw_digest = sha256(raw)
    if raw_digest != RAW_SHA256:
        raise ValueError('official QM9 raw tensor SHA256 mismatch')
    split_path = UPSTREAM / 'random_split.t'
    if sha256(split_path) != SPLIT_SHA256:
        raise ValueError('official split SHA256 mismatch')
    split = torch.as_tensor(torch.load(split_path, map_location='cpu', weights_only=True)).long()
    dataset = VerifiedRawQM9(raw) # Never opens or trusts processed/ files.
    if len(dataset) != 130831 or split.shape != (130831,) or not torch.equal(split.sort().values, torch.arange(len(dataset))):
        raise ValueError('official split/dataset length or permutation mismatch')
    labels = dataset._data.y
    mean, std = labels.mean(0, keepdim=True), labels.std(0, keepdim=True)
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('invalid official full-dataset normalization')
    info = dict(upstream_commit=COMMIT, split_sha256=SPLIT_SHA256,
                archive_sha256=sha256(archive) if archive.exists() else None,
                raw_sha256=raw_digest, graphs=len(dataset),
                split_sizes=dict(train=len(dataset)-20000, validation=10000, test=10000),
                target_order=TARGETS, normalization='official full-dataset mean and unbiased std, before split',
                consumed_data='hash-verified raw tensor dictionaries; processed caches ignored',
                loader_sha256=sha256(Path(__file__)),
                runtime=dict(torch=str(torch.__version__), torch_geometric=str(torch_geometric.__version__)),
                mean=mean.tolist(), std=std.tolist())
    return dataset, split, mean, std, info


def real_batch(dataset, indices, mean, std, *, dtype=torch.float32, device='cpu'):
    graphs = []
    for index in indices:
        graph = dataset[int(index)].clone()
        graph.y = ((graph.y - mean) / std)[:, TARGETS]
        graphs.append(complete(graph))
    batch = Batch.from_data_list(graphs).to(device=device)
    for key, value in batch:
        if torch.is_tensor(value) and value.is_floating_point():
            batch[key] = value.to(dtype=dtype)
    return batch


def provenance():
    return dict(commit=COMMIT, base_url=f'https://github.com/median-research-group/LibMTL/tree/{COMMIT}',
                license='MIT; retained in qm9_upstream/LICENSE',
                files={p.name: sha256(p) for p in UPSTREAM.iterdir() if p.is_file()})


if __name__ == '__main__':
    _, _, _, _, info = load_real(prepare=True)
    info['upstream'] = provenance()
    (ROOT / 'results/v52/qm9_verified_data_provenance.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps({k: info[k] for k in ['graphs', 'split_sizes', 'archive_sha256']}))
