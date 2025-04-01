# LLM Learning Pipeline 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub stars](https://img.shields.io/github/stars/BasicLLM/llm-learn?style=social)

一个模块化的大模型学习工程化实践项目，从零实现 transformer 核心组件到复现经典大模型架构。

## 项目结构 📂

1. Transformer : 实现一个简单的 transformer 网络，此次实现仅注重原理，不适合实际使用
2. Transformer-OPT : 实现一个优化的 transformer 网络，并进行简单的训练
3. Tokenizer : 实现一个分词器
4. Minimind-REC : 记录解读大语言模型 [MiniMind](https://github.com/jingyaogong/minimind)
5. Pop-LLM-Analysis : 当前流行的大语言模型分析

## 核心特性 ✨

- 学习路线平缓，从 Transformer 逐步讲解注意力、MoE 到大模型实现
- 所有关键算法均通过 Jupyter Notebook 实现
- 每个代码单元附带详细注释说明（📌标注关键实现细节）

## 快速开始 🚀

由于有时 Github 中的 ipynb 文件展示时存在无法正确加载图片的情况，这里推荐读者通过 clone 到本地的方式进行阅读实践：

```bash
git clone https://github.com/BasicLLM/llm-learn.git
```

## 许可协议 📜

[MIT License](https://opensource.org/licenses/MIT)

---

🦉 特别鸣谢 [MiniMind](https://github.com/jingyaogong/minimind/tree/master) 提供的架构设计参考