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

【温馨提示】为确保最佳学习体验，建议您通过本地环境运行代码笔记：

克隆代码仓库（推荐使用国内镜像源加速）：

```bash
git clone https://github.com/BasicLLM/llm-learn.git
```

启动Jupyter Notebook：

```bash
jupyter notebook
```

若遇GitHub在线预览的图片加载问题，这是因平台对*.ipynb文件的动态渲染限制所致。本地运行还可获得以下优势：
- 实时执行和修改代码
- 完整保留可视化输出
- 支持自定义调试

## 许可协议 📜

[MIT License](https://opensource.org/licenses/MIT)

---

🦉 特别鸣谢 [MiniMind](https://github.com/jingyaogong/minimind/tree/master) 提供的架构设计参考