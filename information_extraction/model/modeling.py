from paddlenlp.transformers import (
    ErniePretrainedModel,
    ErnieConfig,
    ErnieModel,
    XLNetPretrainedModel,
    XLNetModel,
    XLNetConfig,
    RoFormerConfig,
    RoFormerModel,
    RoFormerPretrainedModel,
    BertModel,
)
from paddle.static import InputSpec

from paddlenlp.utils.log import logger
from paddle import nn, Tensor, squeeze
from typing import Optional


# main model
class UIE(ErniePretrainedModel):
    """
    Ernie Model with two linear layer on top of the hidden-states
    output to compute `start_prob` and `end_prob`,
    designed for Universal Information Extraction.
    Args:
        config (:class:`ErnieConfig`):
            An instance of ErnieConfig used to construct UIE
    """

    def __init__(self, config: ErnieConfig):
        super(UIE, self).__init__(config)
        self.ernie = ErnieModel(config)
        self.linear_start = nn.Linear(config.hidden_size, 1)
        self.linear_end = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
        ]

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieModel`.
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import UIE, ErnieTokenizer
                tokenizer = ErnieTokenizer.from_pretrained('uie-base')
                model = UIE.from_pretrained('uie-base')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                start_prob, end_prob = model(**inputs)
        """
        '''TODO softprompt
        Goal: 訓練一個 softprompt，將固定的 promp 變成 softprompt 後，concat 原本的 embedding
        需實作：自己繼承一個 Word embedding layer，將 prompt 和 content 各自餵給不同的 embedding
            接著將所有 layer 的 gradient fix 起來，只訓練 prompt embedding，如此一來就有固定的
            prompt embedding，最後將 prompt embedding fix 住，訓練其他layer，看成效如何。
        
        Note: 只要 input_ids = None 就可以讓模型只吃到 inputs_embeds
        
        實作:
        1. 

        '''
        
        sequence_output, _ = self.ernie(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        start_logits = self.linear_start(sequence_output)
        start_logits = squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob


class IE_Ernie(ErniePretrainedModel):
    """
    Ernie Model with two linear layer on top of the hidden-states
    output to compute `start_prob` and `end_prob`,
    designed for Universal Information Extraction.
    Args:
        config (:class:`ErnieConfig`):
            An instance of ErnieConfig used to construct UIE
    """

    def __init__(self, config: ErnieConfig):
        super(IE_Ernie, self).__init__(config)
        self.ernie = ErnieModel(config)
        self.linear_start = nn.Linear(config.hidden_size, 1)
        self.linear_end = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
        ]

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ):

        return self.ernie(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )


class IE_XLNet(XLNetPretrainedModel):
    def __init__(self, config: XLNetConfig):
        super(IE_XLNet, self).__init__(config)
        self.xlnet = XLNetModel(config)
        self.initializer_range = config.initializer_range
        self.linear_start = nn.Linear(config.d_model, 1)
        self.linear_end = nn.Linear(config.d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
        ]

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieModel`.
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import UIE, ErnieTokenizer
                tokenizer = ErnieTokenizer.from_pretrained('uie-base')
                model = UIE.from_pretrained('uie-base')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                start_prob, end_prob = model(**inputs)
        """
        sequence_output, _ = self.xlnet(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        start_logits = self.linear_start(sequence_output)
        start_logits = squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob


class IE_Roformer(RoFormerPretrainedModel):
    def __init__(self, config: RoFormerConfig):
        super(IE_Roformer, self).__init__(config)
        self.roformer = RoFormerModel(config)
        self.linear_start = nn.Linear(config.hidden_size, 1)
        self.linear_end = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
        ]

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieModel`.
        Example:
            .. code-block::
                import paddle
                from paddlenlp.transformers import UIE, ErnieTokenizer
                tokenizer = ErnieTokenizer.from_pretrained('uie-base')
                model = UIE.from_pretrained('uie-base')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                start_prob, end_prob = model(**inputs)
        """

        sequence_output, _ = self.roformer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        start_logits = self.linear_start(sequence_output)
        start_logits = squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob
