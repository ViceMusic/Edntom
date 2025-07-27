import torch, math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from torcheval.metrics import BinaryAUPRC, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall

nc_dic ={'A':0, 'T':1, 'G':2, 'C':3, 'N':4}
device = "cuda:0"

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # old implementation
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


###################################
#  Embedding
###################################
def positional_norm_enc(att_center, len, att_std):
    position = torch.arange(0, len).float().unsqueeze(1).cuda()
    dist = torch.distributions.Normal(att_center, att_std)
    d = torch.exp_(dist.log_prob(position))
    return d


def positional_beta_enc(att_center, len, att_std):
    position = torch.arange(1, len + 1).float().unsqueeze(1).cuda()
    position = position / (len + 1)
    dist = torch.distributions.beta.Beta(att_center, att_std)
    d = torch.exp_(dist.log_prob(position))
    return d


#################################################################
# refinement-1: to make the positional embedding to be learnable.
#################################################################
class PositionalEmbedding_plus(nn.Module):
    def __init__(self, d_model, max_len=13):
        super(PositionalEmbedding_plus, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # self.pe = torch.nn.parameter.Parameter(pe.float().to("cuda: 2"), requires_grad=True)
        self.pe = torch.nn.parameter.Parameter(pe.float().to(device), requires_grad=True)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


########################################################################
# refinement-2: consider the relative distance in the attention module.
########################################################################
class Relative_PositionalEmbedding(nn.Module):
    def __init__(self, num_units=13, max_relative_position=3):
        super(Relative_PositionalEmbedding, self).__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position

        self.embedding_table = torch.nn.parameter.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units).to(device), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding_table)

    def forward(self, seq_len):
        with torch.no_grad():
            range_vec_q = torch.arange(seq_len)
            range_vec_k = torch.arange(seq_len)
            dist_mat = range_vec_k[None, :] - range_vec_q[:, None]
            dist_mat_clipped = torch.clamp(dist_mat, -self.max_relative_position, self.max_relative_position)
            final_mat = dist_mat_clipped + self.max_relative_position
            final_mat = torch.LongTensor(final_mat)
            embeddings = self.embedding_table[final_mat]

        return embeddings


"""
class SegmentEmbedding(nn.Embedding):
	def __init__(self, embed_size=32):
		super().__init__(3, embed_size, padding_idx=0)

class TokenEmbedding(nn.Embedding):
	def __init__(self, vocab_size=4, embed_size=32):
		super().__init__(vocab_size, embed_size, padding_idx=0)
"""


# Ensembl of the embedding information
class BERTEmbedding_plus(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1, inLineEmbed=True):
        super(BERTEmbedding_plus, self).__init__()

        self.position = PositionalEmbedding_plus(d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)

        self.featEmbed = nn.Linear(vocab_size, embed_size)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.featEmbed(sequence) + self.position(sequence)
        return self.dropout(x)


############ attention ############
class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class Attention_relative(nn.Module):

    def forward(self, query, key, value, r_k, r_v, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        # additional transform for the relative information
        batch_size, heads, length, _ = query.size()
        r_q = query.permute(2, 0, 1, 3).contiguous()
        r_q = r_q.reshape([length, heads * batch_size, -1])
        rel_score = torch.matmul(r_q, r_k.transpose(-2, -1))
        rel_score = rel_score.contiguous().reshape([length, batch_size, heads, -1]).permute([1, 2, 0, 3])
        scores = scores + rel_score / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        # additional transform for the relative information
        r_attn = p_attn.permute(2, 0, 1, 3).contiguous().reshape([length, heads * batch_size, -1])
        rel_v = torch.matmul(r_attn, r_v)
        rel_v = rel_v.contiguous().reshape([length, batch_size, heads, -1]).permute([1, 2, 0, 3])

        return torch.matmul(p_attn, value) + rel_v, p_attn


class MultiHeadedAttention_relative(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention_relative, self).__init__()

        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention_relative()
        self.dropout = nn.Dropout(p=dropout)

        # fixed relative embedding here, total_window_length=21, relative context_window_length=3
        self.Relative_PositionalEmbedding1 = Relative_PositionalEmbedding(self.d_k, 3)
        self.Relative_PositionalEmbedding2 = Relative_PositionalEmbedding(self.d_k, 3)
        self.r_v = self.Relative_PositionalEmbedding1(13)
        self.r_k = self.Relative_PositionalEmbedding2(13)

        self.attn_output = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, self.r_k, self.r_v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        self.attn_output = attn

        return self.output_linear(x)


############ Transform block build ############

class TransformerBlock_relative(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super(TransformerBlock_relative, self).__init__()

        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.attention = MultiHeadedAttention_relative(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


######################################
## BERT: bi-directional model build ##
######################################
class BERT_plus_encoder(pl.LightningModule):
    def __init__(self, vocab_size=7, hidden=100, n_layers=3, attn_heads=4, dropout=0, motif_shift=0, motif_len=2,
                 inLineEmbed=True):
        super(BERT_plus_encoder, self).__init__()

        # note the network structure is defined in the inital function
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # feed forward hidden
        self.feed_forward_hidden = hidden * 4

        # embedding module
        self.embedding = BERTEmbedding_plus(vocab_size, hidden, dropout, inLineEmbed)

        # stacked transformers
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock_relative(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        # change to the relative context window length accordingly.
        self.linear = nn.Linear(hidden * 7, 2)
        self.tanh = nn.Tanh()

        self.flag=0

    def forward(self, x, nano_data):
        # print("~~~~~~~~~~~~~~", self.transformer_blocks[0].attention.Relative_PositionalEmbedding1.embedding_table)
        seqs = list(x)
        seq_encode = [[nc_dic[c] for c in f] for f in seqs]
        seq_encode = torch.Tensor(seq_encode).to(torch.int64)
        input_ids = F.one_hot(seq_encode)
        input_ids = input_ids.to(device)
        # print(input_ids.size())
        # print(nano_data.size())
        x = torch.cat((input_ids, nano_data), -1)
        x = self.embedding(x)
        mask = None

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        mid_idx = int(x.size(1) / 2)
        rep = torch.cat((x[:, mid_idx - 3, :], x[:, mid_idx - 2, :], x[:, mid_idx - 1, :], x[:, mid_idx, :],
                                     x[:, mid_idx + 1, :], x[:, mid_idx + 2, :], x[:, mid_idx + 3, :]), -1)
        out = self.linear(rep)

        out = self.tanh(out)
        return out, rep

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # with torch.autograd.detect_anomaly():
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.CrossEntropyLoss()#weight=torch.FloatTensor([1, 1])).to(device)
        x, nano_data, y = batch

        output, _ = self.forward(x, nano_data)

        # calculate the accuracy
        acc_sum = (output.argmax(dim=1) == y).float().sum().item()
        n = y.shape[0]
        acc = acc_sum / n

        loss = criterion(output, y)
        # loss.backward(retain_graph=True)

        predictions = F.softmax(output, dim=1)
        predictions = predictions[:, 1]
        predictions = predictions.cpu()
        y = y.cpu()
        # print(predictions)
        # print(y)
        # print("11111prediction: ", predictions.size())
        # print("11111y: ", y.size())


        metric1 = BinaryAUPRC()
        metric1.update(predictions, y)
        AUPRC = metric1.compute()

        metric2 = BinaryAUROC()
        metric2.update(predictions, y)
        AUROC = metric2.compute()

        metric3 = BinaryPrecision()
        metric3.update(predictions, y)
        Precision = metric3.compute()

        metric4 = BinaryRecall()
        metric4.update(predictions, y)
        Recall = metric4.compute()

        metric5 = BinaryF1Score()
        metric5.update(predictions, y)
        F1Score = metric5.compute()

        # Logging to TensorBoard by default
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ACC", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_AUPRC", AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_AUROC", AUROC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_Precision", Precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_Recall", Recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_F1Score", F1Score, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) #1e-3
        return optimizer

    def validation_step(self, val_batch, batch_idx):
        # self.eval()
        criterion = nn.CrossEntropyLoss()
        x, nano_data, y = val_batch

        output, _ = self.forward(x, nano_data)

        # calculate the accuracy
        acc_sum = (output.argmax(dim=1) == y).float().sum().item()
        n = y.shape[0]
        acc = acc_sum / n

        loss = criterion(output, y)

        predictions = F.softmax(output, dim=1)
        predictions = predictions[:, 1]
        predictions = predictions.cpu()
        y = y.cpu()

        # print("11111prediction: ", predictions.size())
        # print("11111y: ", y.size())

        metric1 = BinaryAUPRC()
        metric1.update(predictions, y)
        AUPRC = metric1.compute()

        metric2 = BinaryAUROC()
        metric2.update(predictions, y)
        AUROC = metric2.compute()

        metric3 = BinaryPrecision()
        metric3.update(predictions, y)
        Precision = metric3.compute()

        metric4 = BinaryRecall()
        metric4.update(predictions, y)
        Recall = metric4.compute()

        metric5 = BinaryF1Score()
        metric5.update(predictions, y)
        F1Score = metric5.compute()

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ACC", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_AUPRC", AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_AUROC", AUROC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_Precision", Precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_Recall", Recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_F1Score", F1Score, on_step=False, on_epoch=True, prog_bar=True)
        acc = torch.tensor(acc, dtype=float).cuda()

        # 返回损失值字典
        # print('val_loss:', loss)
        return {'val_loss': loss, 'val_ACC': acc, 'val_AUPRC': AUPRC, 'val_AUROC': AUROC, 'val_Precision': Precision, 'val_Recall': Recall, 'val_F1Score': F1Score,}

    def validation_epoch_end(self, outputs):
        # 计算并输出平均损失
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        avg_ACC = torch.stack([x['val_ACC'] for x in outputs]).mean()
        self.log('avg_val_ACC', avg_ACC, on_step=False, on_epoch=True, prog_bar=True)
        avg_AUPRC = torch.stack([x['val_AUPRC'] for x in outputs]).mean()
        self.log('avg_val_AUPRC', avg_AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        avg_AUROC = torch.stack([x['val_AUROC'] for x in outputs]).mean()
        self.log('avg_val_AUROC', avg_AUROC, on_step=False, on_epoch=True, prog_bar=True)
        avg_Precision = torch.stack([x['val_Precision'] for x in outputs]).mean()
        self.log('avg_val_Precision', avg_Precision, on_step=False, on_epoch=True, prog_bar=True)
        avg_Recall = torch.stack([x['val_Recall'] for x in outputs]).mean()
        self.log('avg_val_Recall', avg_Recall, on_step=False, on_epoch=True, prog_bar=True)
        avg_F1Score = torch.stack([x['val_F1Score'] for x in outputs]).mean()
        self.log('avg_val_F1Score', avg_F1Score, on_step=False, on_epoch=True, prog_bar=True)

    # def on_validation_end(self):
    #     # 强制将验证集损失值写入日志系统
    #     self.log('val_loss', self.trainer.callback_metrics['val_loss'], on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        criterion = nn.CrossEntropyLoss()
        x, nano_data, y = test_batch

        output, rep = self.forward(x, nano_data)

        # calculate the accuracy
        acc_sum = (output.argmax(dim=1) == y).float().sum().item()
        n = y.shape[0]
        acc = acc_sum / n

        loss = criterion(output, y)

        predictions = F.softmax(output, dim=1)
        predictions = predictions[:, 1]
        predictions = predictions.cpu()
        y = y.cpu()

        metric1 = BinaryAUPRC()
        metric1.update(predictions, y)
        AUPRC = metric1.compute()

        metric2 = BinaryAUROC()
        metric2.update(predictions, y)
        AUROC = metric2.compute()

        metric3 = BinaryPrecision()
        metric3.update(predictions, y)
        Precision = metric3.compute()

        metric4 = BinaryRecall()
        metric4.update(predictions, y)
        Recall = metric4.compute()

        metric5 = BinaryF1Score()
        metric5.update(predictions, y)
        F1Score = metric5.compute()

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_ACC", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_AUPRC", AUPRC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_AUROC", AUROC, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_Precision", Precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_Recall", Recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_F1Score", F1Score, on_step=False, on_epoch=True, prog_bar=True)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        # 计算并输出平均损失
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        print('avg_test_loss:', avg_loss)
