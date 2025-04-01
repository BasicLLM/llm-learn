import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, dim_emb: int, max_len: int = 5000):
        """
        正余弦位置编码模块

        :param dim_emb: 嵌入维度
        :param max_len: 预设最大序列长度 (default: 5000)
        """
        super().__init__()

        # 初始化位置编码矩阵 [max_len, dim_emb]
        pe = torch.zeros(max_len, dim_emb)

        # 生成位置序列 [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算频率调节因子 [dim_emb/2]
        div_term = torch.exp(
            torch.arange(0, dim_emb, 2).float() * (-math.log(10000.0) / dim_emb)
        )

        # 交替填充正弦余弦值
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        # 注册为缓冲区 (不参与梯度计算)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, dim_emb]

    def forward(self, x, start, end) -> torch.Tensor:
        # 动态截取序列长度
        return x + Variable(self.pe[:, start: end], requires_grad=False)

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float=1e-5):
        """
        Root Mean Squared Error Normalized

        :param dim: 维度
        :param eps: 公式中的 ε，很小的数，防止除零
        """
        super().__init__()
        self.eps = eps
        # weight 即缩放矩阵的权重
        self.weight = nn.Parameter(torch.ones(dim))

    def _rms(self, x):
        # rsqrt 是 sqrt 的倒数，即 rsqrt(x) = 1/sqrt(x)
        # 这里 mean() 中,dim = -1 ，指对最后一维求平均数，而 keepdim 指不改变该张量的维度
        return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def _norm(self, x):
        return x * self._rms(x)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, dim_hidden, bias=False)
        self.w2 = nn.Linear(dim_hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, dim_hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_emb:int, dim_head:int, dim_out:int, head_num:int):
        """
        Multi-Head Attention

        :param dim_emb: 数据的嵌入维度
        :param dim_head: 每个 Attention 块对应的头维度
        :param dim_out: 数据的输出维度
        :param head_num: 注意力头数
        """
        super().__init__()

        self.dim_emb = dim_emb
        self.dim_head = dim_head
        self.head_num = head_num

        # 计算多个 Attention 头维度合并后的维度
        dim_mid = dim_head * head_num

        # 创建 Attention 权重
        self.Q_net = nn.Linear(dim_emb, dim_mid)
        self.K_net = nn.Linear(dim_emb, dim_mid)
        self.V_net = nn.Linear(dim_emb, dim_mid)

        # 创建权重
        self.W_net = nn.Linear(dim_mid, dim_out)

    def forward(self, x_q, x_k, x_v, mask_mat = None, use_cache = False, kv_cache = None):

        # 获取各个维度的值
        batch, q_seq_len, _ = x_q.size()
        _, kv_seq_len, _ = x_k.size()

        # 求 q、k、v 的值，并将其形状变为 (batch, head_num, seq_len, dim_head)
        # 注意这里不能直接进行 view 操作，会导致张量中的元素分配错误，具体详情请参考 pytorch 官方文档
        q = self.Q_net(x_q).view(batch, q_seq_len, self.head_num, self.dim_head).transpose(1, 2)
        k = self.K_net(x_k).view(batch, kv_seq_len, self.head_num, self.dim_head).transpose(1, 2)
        v = self.V_net(x_v).view(batch, kv_seq_len, self.head_num, self.dim_head).transpose(1, 2)

        # 如果使用 KV Cache，拼接历史 K 和 V
        if use_cache and kv_cache is not None:
            k_prev, v_prev = kv_cache
            # 在序列长度维度拼接
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        # 记录新的 KV Cache（供下一步使用）
        new_kv_cache = (k, v) if use_cache else None

        # 将 K 的最后两个维度进行转置，转置后的维度为 (batch, head_num, dim_head, kv_seq_len)
        k_t = k.transpose(-1, -2)

        # 计算 qk^T / sqrt(d_k)，此时 s0 的维度为 (batch, head_num, q_seq_len, kv_seq_len)
        s0 = torch.matmul(q, k_t) / math.sqrt(self.dim_head)

        # 进行掩码遮掩操作
        if mask_mat is not None:
            # 进行遮掩
            s0 = torch.masked_fill(
                s0,
                mask_mat,
                float('-inf')
            )

        # 计算 softmax(s)*v ，此时 s1 的维度为 (batch, head_num, q_seq_len, dim_head)
        s1 = torch.matmul(F.softmax(s0, dim=-1), v)

        # 我们需要使用 s1*W ，这里就要将矩阵变换为可以跟 W 矩阵进行矩阵乘法的维度，即：
        # s1 变换维度为：(batch, q_seq_len, dim_head * head_num)
        s1 = s1.transpose(1, 2).contiguous() # 这里需要让内存连续（ 使用 reshape 则不用）
        s1 = s1.view(batch, q_seq_len, self.head_num * self.dim_head)

        # 输出的最终维度为：(batch, q_seq_len, dim_out)
        output = self.W_net(s1)

        return output, new_kv_cache


class Encoder(nn.Module):
    def __init__(self, dim_emb:int, dim_head:int, head_num:int):
        """
        Transformer Encoder

        :param dim_emb: 数据的嵌入维度
        :param dim_head: 单个注意力模块的头维度
        :param head_num: 多头注意力的头数
        """
        super().__init__()

        self.dim_emb = dim_emb
        self.dim_head = dim_head
        self.head_num = head_num

        # 构建编码器需要的子模块
        self.multi_attention = MultiHeadAttention(dim_emb, dim_head, dim_emb, head_num)
        self.rms_norm1 = RMSNorm(dim_emb)
        self.feed_forward = FeedForward(dim_emb, 4 * dim_emb)
        self.rms_norm2 = RMSNorm(dim_emb)

    def forward(self, x):

        d0, _ = self.multi_attention(x, x, x)
        d1 = self.rms_norm1(x + d0)
        d2 = self.feed_forward(d1)
        d3 = self.rms_norm2(d1 + d2)

        return d3

class Decoder(nn.Module):
    def __init__(self, dim_emb:int, dim_head:int, head_num:int):
        """
        Transformer Decoder

        :param dim_emb: 数据的嵌入维度
        :param dim_head: 单个注意力模块的头维度
        :param head_num: 多头注意力的头数
        """
        super().__init__()

        self.dim_emb = dim_emb
        self.dim_head = dim_head
        self.head_num = head_num

        # 构建解码器需要的子模块
        self.masked_multi_head_attention = MultiHeadAttention(dim_emb, dim_head, dim_emb, head_num)
        self.rms_norm1 = RMSNorm(dim_emb)
        self.multi_head_attention = MultiHeadAttention(dim_emb, dim_head, dim_emb, head_num)
        self.rms_norm2 = RMSNorm(dim_emb)
        self.feed_forward = FeedForward(dim_emb, 4 * dim_emb)
        self.rms_norm3 = RMSNorm(dim_emb)

    def forward(self, x ,enc_k = None, enc_v = None, dec_mask_mat = None, enc_mask_mat = None, use_cache = False, kv_caches = None):

        d0, kv_cache0 = self.masked_multi_head_attention(
            x, x, x, dec_mask_mat,
            use_cache = use_cache,
            kv_cache = kv_caches[0] if kv_caches is not None else None
        )
        d1 = self.rms_norm1(x + d0)

        _enc_k = d1 if enc_k is None else enc_k
        _enc_v = d1 if enc_v is None else enc_v

        d2, kv_cache1 = self.multi_head_attention(
            d1, _enc_k, _enc_v, enc_mask_mat,
            use_cache = use_cache,
            kv_cache = kv_caches[1] if kv_caches is not None else None
        )
        d3 = self.rms_norm2(d1 + d2)
        d4 = self.feed_forward(d3)
        d5 = self.rms_norm3(d3 + d4)

        new_kv_caches = (kv_cache0, kv_cache1) if use_cache else None

        return d5, new_kv_caches

class Transformer(nn.Module):
    def __init__(
            self,
            encoder_num:int,
            decoder_num:int,
            vocab_size:int,
            dim_emb:int,
            dim_head:int,
            head_num:int,
            max_seq_len:int = 50000,
            training:bool = False,
    ):
        """
        Transformer

        :param encoder_num: 编码器数量
        :param decoder_num: 解码器数量
        :param vocab_size: 词表大小
        :param dim_emb: 嵌入维度
        :param dim_head: 单个注意力模块的头维度
        :param head_num: 头数
        :param max_seq_len: 最大序列长度
        :param training: 是否开启训练模式
        """
        super().__init__()

        self.encoder_num = encoder_num
        self.decoder_num = decoder_num
        self.vocab_size = vocab_size
        self.dim_emb = dim_emb
        self.dim_head = dim_head
        self.head_num = head_num
        self.training = training

        # 位置编码模块
        self.position_embedding = PositionalEmbedding(dim_emb, max_len = max_seq_len)

        # 词向量嵌入模块
        self.input_embedding = nn.Embedding(vocab_size, dim_emb)
        self.output_embedding = nn.Embedding(vocab_size, dim_emb)

        # 创建解码器和编码器
        self.encoder_blocks = nn.ModuleList()
        for i in range(encoder_num):
            self.encoder_blocks.append(Encoder(dim_emb, dim_head, head_num))

        self.decoder_blocks = nn.ModuleList()
        for i in range(decoder_num):
            self.decoder_blocks.append(Decoder(dim_emb, dim_head, head_num))

        # 创建线性层，线性层的作用是将嵌入维度转换为词表维度
        self.linear = nn.Linear(dim_emb, vocab_size)

    def forward(
            self,
            input_seq,
            output_seq,
            input_mask = None,
            output_mask = None,
            enc_output_cache = None,
            dec_kv_caches = None
    ):

        training = self.training
        batch, output_seq_len = output_seq.size()

        # ============
        #   编码过程
        # ============

        if training or enc_output_cache is None:
            # 输入词向量嵌入
            embedded_input_seq = self.input_embedding(input_seq)

            # 对 input_seq 进行位置标记
            _,input_seq_len = input_seq.size()
            pe_input_seq = self.position_embedding(embedded_input_seq, 0, input_seq_len)

            # 计算编码器的计算结果，其最终输出维度为 (batch, input_seq_len, dim_emb)
            for encoder in self.encoder_blocks:
                pe_input_seq = encoder(pe_input_seq)

            enc_output = pe_input_seq

        else:
            enc_output = enc_output_cache

        # ============
        #   解码过程
        # ============

        # 输出词向量嵌入
        embedded_output_seq = self.output_embedding(output_seq)

        # 计算解码器掩码矩阵
        if output_seq_len > 1:
            triu_mask_mat = torch.triu(
                torch.ones(output_seq_len, output_seq_len, dtype=torch.bool, device=output_seq.device),
                diagonal=1
            )
            if output_mask is not None:
                dec_mask_mat = torch.stack([triu_mask_mat | (~ output_mask[i].bool()) for i in range (batch)]).type(dtype=torch.bool)
                dec_mask_mat = dec_mask_mat.view(batch, 1, output_seq_len, output_seq_len).requires_grad_(False)
            else:
                dec_mask_mat = triu_mask_mat.requires_grad_(False)
        else:
            dec_mask_mat = None

        # 计算编码器掩码矩阵
        enc_mask_mat = None
        if input_mask is not None:
            input_seq_len = input_mask.size(1)
            zeros_mat = torch.zeros(output_seq_len, input_seq_len, dtype=torch.bool, device=output_seq.device)
            enc_mask_mat = (torch.stack([zeros_mat | (~ input_mask[i].bool()) for i in range (batch)])).type(dtype=torch.bool)
            enc_mask_mat = enc_mask_mat.view(batch, 1, output_seq_len, input_seq_len).requires_grad_(False)

        # 检查是否需要缓存
        new_dec_kv_caches = [] if not training else None
        if training:

            # 位置编码
            pe_output_seq = self.position_embedding(embedded_output_seq, 0, output_seq_len)

            # 计算解码器的计算结果，其最终输出维度为 (batch, output_seq_len, dim_emb)
            for decoder in self.decoder_blocks:
                pe_output_seq, _ = decoder(
                    pe_output_seq, enc_output, enc_output,
                    enc_mask_mat = enc_mask_mat,
                    dec_mask_mat = dec_mask_mat
                )

        else:

            # 位置编码
            if dec_kv_caches is not None:
                # dec_kv_caches 对应的索引： [编码器层数][编码器中的多头注意力的层数][K Cache/V Cache]
                # K Cache / V Cache 对应的张量维度： (批数, 头数, 缓存的 K / V 序列长度, 头维度)
                cached_seq_len = dec_kv_caches[0][0][0].size(2)
                pe_output_seq = self.position_embedding(embedded_output_seq, cached_seq_len, cached_seq_len + 1)
            else:
                pe_output_seq = self.position_embedding(embedded_output_seq, 0, output_seq_len)

            # 计算解码器的计算结果，并缓存 KV 的值
            for dec_idx, decoder in enumerate(self.decoder_blocks):
                if dec_kv_caches is not None:
                    kv_caches = dec_kv_caches[dec_idx]
                else:
                    kv_caches = None

                pe_output_seq, new_kv_caches = decoder(
                    pe_output_seq, enc_output, enc_output,
                    enc_mask_mat = enc_mask_mat,
                    dec_mask_mat = dec_mask_mat,
                    use_cache = True,
                    kv_caches = kv_caches
                )
                new_dec_kv_caches.append(new_kv_caches)

        # 最终输出维度为 (batch, output_seq_len, vocab_size)
        logits = self.linear(pe_output_seq)
        if not training:
            output = F.softmax(logits, dim=-1)
            return output, enc_output, new_dec_kv_caches
        else:
            # 需要使用交叉熵计算损失，而 pytorch 提供的交叉熵算法内部已经包含了 softmax
            # 所以这里结果不要进行 softmax ，否则会导致损失计算不稳定
            return logits