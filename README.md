# 🏥 AI 健康导诊助手

一个基于 **LangGraph** 和 **Gradio** 构建的智能健康导诊与心理支持系统。通过多 Agent 协作，帮助用户快速了解症状对应的挂号科室，并提供简短的心理状态自测与陪伴服务。

## ✨ 功能特点

- **智能导诊**：通过多轮对话收集症状信息，推荐合适的挂号科室
- **紧急预警**：自动识别心梗、卒中等高危症状，提示立即就医
- **心理支持**：提供简短的心理状态自测（8 题版）和温暖的陪伴对话
- **多模态扩展**：支持文字输入，预留图片分析接口
- **安全审核**：自动过滤诊断结论和用药建议，强制添加免责声明

## 🧠 系统架构

项目采用 **Multi-Agent** 设计，由 4 个核心 Agent 协作完成：

| Agent | 职责 |
| :--- | :--- |
| **Agent 1** | 症状提取官：多轮追问，收集结构化症状信息 |
| **Agent 2** | 紧急急救官：纯函数硬编码，拦截 RED 级危险信号 |
| **Agent 3** | 科室匹配员：根据症状推荐挂号科室 |
| **Agent 4** | 安全审核员：过滤诊断结论，强制添加免责声明 |
| **Agent 5** | 心理支持助理：心理自测与陪伴对话 |

工作流基于 **LangGraph** 的 StateGraph 实现条件路由，支持 RED 级急诊短路。

## 🛠️ 技术栈

- **后端框架**：LangGraph + LangChain
- **大模型**：DeepSeek-V3（通过 SiliconFlow API 调用）
- **前端界面**：Gradio
- **开发语言**：Python 3.9+

## 🚀 快速开始

### 环境要求

- Python 3.9 或以上
- SiliconFlow API Key

### 安装与运行

```bash
# 1. 克隆项目
git clone https://github.com/你的用户名/AI_health_chatbox.git
cd AI_health_chatbox

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
# 在终端输入
# export SILICON_API_KEY="你的API密钥"

# 4. 运行应用
python app.py
