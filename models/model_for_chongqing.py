import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.param = param

        # Backbone
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        self.backbone.proj_out = nn.Identity()

        # Load pretrained weights if requested
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            self.backbone.load_state_dict(state_dict, strict=False)  # <- allow missing/unexpected keys


        self.num_classes = 1
        self.dropout = param.dropout
        self.classifier_type = param.classifier
        self.classifier = None  # will initialize dynamically if needed

    def forward(self, x):
        # Ensure 4D input: (B, C, S, D)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # add channel dimension

        feats = self.backbone(x)  # output shape: (B, C, S, D)

        # Dynamic classifier for 'all_patch_reps'
        if self.classifier_type == 'all_patch_reps' and self.classifier is None:
            B, C, S, D = feats.shape
            in_feat = C * S * D
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(in_feat, 128),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(128, 1)
            ).to(feats.device)

        # Avg-pooling classifier (optional)
        elif self.classifier_type == 'avgpooling_patch_reps' and self.classifier is None:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                #nn.Linear(feats.shape[1], self.num_classes)
                nn.Linear(feats.shape[1], 1)
            ).to(feats.device)

        out = self.classifier(feats)
        out = out.squeeze(-1)   # (B, 1) → (B,)
        return out
