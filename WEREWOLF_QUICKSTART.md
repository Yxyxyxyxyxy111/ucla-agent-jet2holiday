# 大模型狼人杀 agent

这个版本使用现成模型推理，不训练或微调模型。专门适配本项目的七人狼人杀：2 狼、1 预言家、4 村民。

## 配置和启动

在项目根目录操作，使用 Python 3.11+：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

在现有 `.env` 中配置 `OPENAI_API_KEY` 和 `OPENAI_MODEL`。模型必须支持 Chat Completions 的严格 JSON Schema 输出。可选 `OPENAI_BASE_URL` 用于兼容接口；兼容服务也必须支持该输出格式。平台连接沿用已有 `ALTRUAGENT_CONTROL_URL`、`ALTRUAGENT_OFFICIAL_AGENT_KEY`，不要覆盖现有密钥。

```bash
python -m agent --check-tournament --agent agent.werewolf_agent
python -m agent --match --agent agent.werewolf_agent
```

在平台 Testing 页面创建狼人杀比赛，将自己的席位设为 Mine (self-hosted)。上述进程等待分配并自动接入；每个席位有独立实例。

正式比赛使用：

```bash
python -m agent --tournament --agent agent.werewolf_agent
```

## 行为与验证

- 根据自己的 observation、公开死亡身份、历史投票和发言决策；身份策略在 `agent/werewolf_agent.py` 的 SYSTEM 中。
- 记忆在单场内保留；重启后发言记忆和私有笔记不保留。
- 只广播简短英文发言，每天最多 3 次决策请求，每次最多 2 次模型调用。消息上限 40 词、240 字符。
- 动作每次最多 2 次模型调用，整体预算 35 秒；讨论依据剩余时间缩短预算。网络超时受底层 HTTP 客户端约束，无法保证严格墙钟截止。
- 无效输出或 API 失败自动重试，再退回合法的非弃权动作或结束讨论。兜底不代表有策略优势。
- 私有笔记仅用于模型上下文，不作为公开 reasoning_summary 或日志发送；公开发言仍由模型生成。
- 不会读取其他席位的私有状态；每个己方席位独立运行。

```bash
python -m pytest tests/test_werewolf_agent.py
```

模拟测试只验证接口行为，不证明胜率。先在 Testing 对比原占位 agent 和通用 LLM agent，分别观察狼人和好人胜率。每次改变策略保持对手阵容与模型配置一致，积累多场再判断。真实对局会产生模型 API 费用。
