# third party
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class NBMEModel(nn.Module):
    """
    Class that creates a model for the NBME data
    """

    def __init__(self, config):  # noqa
        super().__init__()

        # Get the config and model
        self.config = config
        self.model_config = AutoConfig.from_pretrained(self.config.model)
        self.model = AutoModel.from_pretrained(self.config.model)

        # Dropout
        self.fc_dropout = nn.Dropout(self.config.fc_dropout)

        # Linear layers
        self.linear = nn.Linear(self.model_config.hidden_size, 1)
        self._init_weights(self.linear)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):  # noqa
        transformer_features = self.model(**inputs)

        # Get the last hidden state
        # outputs size is (1, 512, 1024)
        last_hidden_states = transformer_features[0]

        # Pass through the linear layer
        outputs = self.fc_dropout(last_hidden_states)
        outputs = self.linear(outputs)

        return outputs
