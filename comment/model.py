import torch
import torch.nn as nn
from copy import deepcopy
from transformers import AutoModelForSequenceClassification


class BertModel(nn.Module):
    def __init__(self, model_name, n_class=2, p=0.1, freeze=False, select=False):
        super().__init__()
        if 'beep' in model_name:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                    output_attentions=False, output_hidden_states=True)
            self.model.classifier.out_proj = nn.Linear(in_features=768, out_features=n_class)

        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                        num_labels=n_class, output_attentions=False, output_hidden_states=True)
        self.n_class = n_class
        self.select = select

        if select:
            self.head = nn.Linear(3072, n_class)

        if 'KcELECTRA' in model_name:
            self.model.classifier.dropout = nn.Dropout(p)
        elif 'kcbert' in model_name:
            self.model.dropout = nn.Dropout(p)
            
        if freeze:
            self._freeze_param(self.model)
        self._trainable_param(self.model)

    def forward(self, input_ids, attention_mask, segment_ids):
        x = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        
        if self.select:
            hidden_states = x.hidden_states
            pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4,-3,-2,-1]]), dim=-1)
            pooled_output = pooled_output[:, 0, :]
            x = self.head(pooled_output)
        else:
            x = x.logits
        return x # B, n_class

    def _freeze_param(self, model):
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                print(f'freezed >> {name}')
                param.requires_grad = False
        return

    def _trainable_param(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(f'training >> {name}')
        return

def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.model.electra.encoder.layer
    newModuleList = nn.ModuleList()
 
    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])
 
    # create a copy of the model, modify it with the new list, and return
    copyOfModel = deepcopy(model)
    copyOfModel.model.electra.encoder.layer = newModuleList
 
    return copyOfModel