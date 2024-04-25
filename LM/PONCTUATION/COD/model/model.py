from typing import Tuple
import torch
import torch.nn as nn
from loss.aggreagator_loss import AggregatorLoss
from model.token_classifier import TokenClassifier
from loss.cross_entropy_loss import CrossEntropyLoss
from transformers import BertForTokenClassification
import os

from collections import namedtuple
# Define the named tuple outside of your model class
Output = namedtuple('Output', ['punct_logits', 'capit_logits', 'loss'])

__all__ = ['PunctuationCapitalizationModel']


class PunctuationCapitalizationModel(nn.Module):
    def __init__(self,  
                 num_punct: int,
                 num_capit: int = 2,
                 pretrained_bert_model_name: str = 'bert-base-multilingual-cased',
                 capit_hidden_size: int = 512,
                 punct_hidden_size: int = 512,
                 activation: str = "relu",
                 dropout:float = 0.1,
                 punct_num_layers:int = 1,
                 capit_num_layers:int = 11) -> None:
        
        super(PunctuationCapitalizationModel, self).__init__()
        self.capit_hidden_size = capit_hidden_size
        self.punct_hidden_size = punct_hidden_size
        """Initializes BERT Punctuation and Capitalization model."""
        self.pretrained_bert_model_name = pretrained_bert_model_name
        self.num_punct = num_punct
        self.num_capit = num_capit
        self.activation = activation
        self.dropout = dropout
        self.punct_num_layers = punct_num_layers
        self.capit_num_layers = capit_num_layers

        self.agg_loss = AggregatorLoss(num_losses = 2)
        self.loss = CrossEntropyLoss(ignore_index=-100)
        self.bert_model = BertForTokenClassification.from_pretrained(self.pretrained_bert_model_name)
        self.input_token_class = self.bert_model.config.hidden_size
        self.punct_classifier = TokenClassifier(
            self.input_token_class,
            self.punct_hidden_size,
            self.num_punct,
            activation = self.activation,
            # log_softmax = False,
            dropout = self.dropout,
            num_layers = self.punct_num_layers
        )

        self.capit_classifier = TokenClassifier(
            self.input_token_class,
            self.capit_hidden_size,
            self.num_capit,
            activation = self.activation,
            # log_softmax = False,
            dropout = self.dropout,
            num_layers = self.capit_num_layers
        )

    def forward(self, input_ids, subtokens_mask, punct_labels, capit_labels, segment_ids, input_mask, loss_mask) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes a forward pass through the model. For more details see ``forward`` method of HuggingFace BERT-like
        (models which accept ``input_ids``, ``attention_mask``, ``token_type_ids`` arguments) models.

        Args:
            input_ids (:obj:`torch.Tensor`): an integer torch tensor of shape ``[Batch, Time]``. Contains encoded
                source tokens.
            attention_mask (:obj:`torch.Tensor`): a boolean torch tensor of shape ``[Batch, Time]``. Contains an
                attention mask for excluding paddings.
            token_type_ids (:obj:`torch.Tensor`): an integer torch Tensor of shape ``[Batch, Time]``. Contains an index
                of segment to which a token belongs. If ``token_type_ids`` is not ``None``, then it should be a zeros
                tensor.

        Returns:
            :obj:`Tuple[torch.Tensor, torch.Tensor]`: a tuple containing

                - ``punct_logits`` (:obj:`torch.Tensor`): a float torch tensor of shape
                  ``[Batch, Time, NumPunctuationLabels]`` containing punctuation logits
                - ``capit_logits`` (:obj:`torch.Tensor`): a float torch tensor of shape
                  ``[Batch, Time, NumCapitalizationLabels]`` containing capitalization logits
        """

        outputs = self.bert_model(
                            input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, output_hidden_states=True
                        )
        hidden_states = outputs.hidden_states
        last_layer_hidden_states = hidden_states[-1]

        punct_logits = self.punct_classifier(last_layer_hidden_states)
        capit_logits = self.capit_classifier(last_layer_hidden_states)
        
        punct_loss = self.loss(logits=punct_logits, labels=punct_labels, loss_mask=loss_mask)
        capit_loss = self.loss(logits=capit_logits, labels=capit_labels, loss_mask=loss_mask)
        loss = self.agg_loss(punct_loss, capit_loss)

        return Output(punct_logits=punct_logits.float(), capit_logits=capit_logits.float(), loss=loss)

    def save_pretrained(self, save_directory: str) -> None:
        """
        Saves the model to the specified directory.

        Args:
            save_directory (str): Directory where the model should be saved.
        """
        os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
        save_path = os.path.join(save_directory, "model.safetensors")
        torch.save({
            'bert_model_state_dict': self.bert_model.state_dict(),
            'punct_classifier_state_dict': self.punct_classifier.state_dict(),
            'capit_classifier_state_dict': self.capit_classifier.state_dict(),
        }, save_path)

    def parameters(self) -> Tuple:
        """
        Returns the parameters of the model.

        Returns:
            :obj:`Tuple`: A tuple containing the parameters of the model.
        """
        return (self.bert_model.parameters(), 
                self.punct_classifier.parameters(), 
                self.capit_classifier.parameters())
    
    def state_dict(self) -> dict:
        """
        Returns the state dictionary of the model.

        Returns:
            :obj:`dict`: A dictionary containing the state of the model.
        """
        return {
            'bert_model_state_dict': self.bert_model.state_dict(),
            'punct_classifier_state_dict': self.punct_classifier.state_dict(),
            'capit_classifier_state_dict': self.capit_classifier.state_dict(),
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads the state dictionary into the model.

        Args:
            state_dict (dict): A dictionary containing the state of the model.
        """
        self.bert_model.load_state_dict(state_dict['bert_model_state_dict'])
        self.punct_classifier.load_state_dict(state_dict['punct_classifier_state_dict'])
        self.capit_classifier.load_state_dict(state_dict['capit_classifier_state_dict'])
        
    












