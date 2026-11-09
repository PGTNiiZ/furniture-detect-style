import torch
import pytest
from torch.utils.data import DataLoader

from furniture.train import run_epoch


def test_accumulation_handles_final_partial_batch():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2, bias=False)
        def forward(self, images):
            return {'type':self.backbone(images)}
    class Rows(torch.utils.data.Dataset):
        def __init__(self):
            self.x = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[2.,1.],[1.,2.]])
            self.y = torch.tensor([0,1,0,1,0])
        def __len__(self): return len(self.y)
        def __getitem__(self, index): return self.x[index], {'type':self.y[index]}
    torch.manual_seed(5)
    first, second = Toy(), Toy()
    second.load_state_dict(first.state_dict())
    mapping = {'type':{'a':0,'b':1}}
    optim1 = torch.optim.SGD(first.parameters(), lr=.1)
    optim2 = torch.optim.SGD(second.parameters(), lr=.1)
    run_epoch(first, DataLoader(Rows(),batch_size=2), mapping, torch.device('cpu'), optim1, accumulation_steps=2)
    run_epoch(second, DataLoader(Rows(),batch_size=4), mapping, torch.device('cpu'), optim2)
    assert torch.allclose(first.backbone.weight, second.backbone.weight, atol=1e-7)


def test_atomic_checkpoint_supports_deep_paths(tmp_path):
    from furniture.train import atomic_torch_save
    parent = tmp_path
    while len(str(parent)) < 222:
        parent = parent / 'deepsegment'
    parent.mkdir(parents=True)
    checkpoint = parent / 'best.pt'
    atomic_torch_save({'value':torch.tensor([1,2])},checkpoint)
    assert torch.equal(torch.load(checkpoint,weights_only=True)['value'],torch.tensor([1,2]))
    assert list(parent.glob('*.tmp')) == []


@pytest.mark.parametrize('task', ['type','style'])
def test_each_task_loss_reaches_shared_backbone(task):
    from furniture.model import AttributeModel
    torch.set_num_threads(1)
    model = AttributeModel({'type':{'a':0,'b':1}, 'style':{'x':0,'y':1}}, pretrained=False)
    logits = model(torch.randn(2,3,64,64))[task]
    torch.nn.functional.cross_entropy(logits,torch.tensor([0,1])).backward()
    assert model.backbone.conv1.weight.grad.abs().sum() > 0


def test_frozen_backbone_batchnorm_and_unfreeze_update():
    from furniture.model import AttributeModel
    torch.set_num_threads(1)
    model=AttributeModel({'type':{'a':0,'b':1}}, pretrained=False)
    class Rows(torch.utils.data.Dataset):
        def __len__(self): return 2
        def __getitem__(self, i): return torch.randn(3,64,64), {'type':torch.tensor(i)}
    loader=DataLoader(Rows(),batch_size=2)
    optimizer=torch.optim.SGD(model.parameters(), lr=.01)
    initial=model.backbone.conv1.weight.detach().clone()
    stats=model.backbone.bn1.running_mean.detach().clone()
    model.freeze_backbone(True)
    run_epoch(model,loader,{'type':{'a':0,'b':1}},torch.device('cpu'),optimizer,frozen=True)
    assert torch.equal(stats,model.backbone.bn1.running_mean)
    assert torch.equal(initial,model.backbone.conv1.weight)
    model.freeze_backbone(False)
    run_epoch(model,loader,{'type':{'a':0,'b':1}},torch.device('cpu'),optimizer)
    assert not torch.equal(initial,model.backbone.conv1.weight)
