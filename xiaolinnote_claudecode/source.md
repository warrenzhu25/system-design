# Source

> Archived from https://xiaolinnote.com/claudecode/ (source). Personal study copy.


## Claude Code 源码拆解：51 万行泄漏代码里的架构设计

> Source: https://xiaolinnote.com/claudecode/source/cc_source.html

大家好，我是小林。

Claude Code 源码泄漏这个瓜，大家都吃了吧？

堂堂业界最强编程 Agent，就因为 npm 打包时配置手抖，不小心把 `.map` 文件一起传了上去，结果 51 万行核心代码直接全网裸奔。

<figure>

<figcaption>image.png</figcaption>
</figure>

面对这场史诗级乌龙，大家都忙着吃瓜，而懂行的第一反应都是：**赶紧下载源码**！毕竟这可是目前地表最强 Agent 的底牌。

<figure>

</figure>

先说清楚，这些泄漏的是 Claude Code 客户端的源码，并不是 Claude Opus 大模型的源码。

可能就有同学疑惑了，客户端源码有啥好看的？

说到这，不知道大家最近有没有注意到，AI 圈最近又造了一个新词，叫 **Harness Engineering（线束工程）**，大家最近都刷到了吧？

<figure>

</figure>

听起来是不是特别唬人？每次 AI 圈一出这种高大上的新词，其实大半都是在「重新包装常识」。

说白了，这词的意思就是大家终于认清现实了：**与其天天「做法」祈求大模型变聪明、别产生幻觉，不如老老实实给这匹野马套上缰绳（Harness）**。用系统去约束它的行为，给它划好赛道，它才能稳稳当当出活。

这也是目前 AI 圈最真实的风向转变：从「拼模型智商」变成了「拼系统求稳」。

而这 51 万行的 Claude Code 源码，简直就是 Harness Engineering 的最佳教科书。

啃完你会发现，人家 80% 的代码根本不是在搞什么黑科技让 AI 更聪明，而是在死磕「可靠性」。

今天这篇文章，我就带大家一层一层把 Claude Code 的核心架构给扒明白。

搞懂这些实打实的工程实践，未来不管是你自己手搓 Agent，还是去面试被问到相关架构设计，那绝对是妥妥的降维打击。

### 一、Claude Code 是什么

在拆源码之前，先简单介绍一下 Claude Code 到底是什么东西。

Claude Code 是 Anthropic 官方推出的编程 Agent 工具。你可以把它理解成一个能直接在你的终端里干活的 AI 程序员，它不是一个聊天窗口，而是真正能读你的代码、改你的文件、跑你的命令、帮你管理 Git 的那种。

<figure>

</figure>

说到底，Claude Code 的本质就是一个 **AI Agent**。但 Agent 这个词现在被用得太泛了，很多人会把它和聊天机器人搞混，所以我得先把这个概念说清楚。

很多人会把 AI Agent 和聊天机器人搞混。其实它们的区别很大。ChatBot 就是你问一句它答一句，一次性的问答。Copilot 呢，是你写代码时给你补全建议，本质上也是一次性预测。

**Agent 的核心是一个「感知-决策-行动」的自主循环**。你给它一个目标（比如「帮我修复这个 bug」），它会自己决定先读哪个文件、再跑什么命令、然后改哪行代码，整个过程可能循环几十轮，直到任务完成。

<figure>

</figure>

Claude Code 的 Agent 循环大概是这样的：

<figure>

</figure>

注意看这个循环的关键：**大模型自己决定下一步做什么**。

它不是按照预定义的流程图走的，而是每次看到当前的对话上下文后，自主判断「我现在应该读个文件」还是「我应该执行一条命令」还是「我可以回复用户了」

好，理解了 Claude Code 是什么、它的核心循环长什么样，接下来我们就可以开始看源码了。先从它的整体架构看起。

### 二、架构设计

一个能自主编程的 Agent 要处理的事情非常多：调大模型 API、执行 40 多种工具、管理权限、压缩上下文、维护记忆、支持多 Agent 协作……如果这些东西全部塞在一个文件里，代码会立刻变成一团乱麻。

那 Claude Code 是怎么组织这些子系统的？它采用了一个**四层分层架构**：

<figure>

</figure>

我们从底层往上，一层一层来理解。

- **引擎层**是 Agent 的「大脑」，负责思考和调度。它的关键设计原则是**不包含任何业务逻辑**，它不知道怎么读文件、怎么改代码、怎么搜索，这些全是工具层的事。引擎层只做三件事：第一，协调，把用户输入、系统指令、历史对话拼在一起，发给大模型；第二，分发，大模型说「我要用某个工具」时，找到对应的工具并执行；第三，决策，根据大模型的返回决定是继续循环还是结束对话。这种设计的好处是：**新增能力只需要新增一个工具，引擎层完全不用改**。
- **工具层**是 Agent 的全部「能力」，40 多个工具都在这一层。每个工具就是 Agent 的一个能力：执行 Shell 命令、读写文件、搜索代码、生成子 Agent……这些工具不是随便写的，它们遵循一个统一的规范。这个规范不仅定义了「工具能做什么」，还强制定义了三个安全属性：这个工具是只读的还是会改东西的？它是否具有破坏性需要额外确认？它能不能和其他工具同时执行？这三个属性不是「建议加上」的，而是类型系统强制要求的，漏了任何一个，代码就编译不过。这意味着**每一把刀都有刀鞘，从出厂就配好了安全机制**。
- **服务层**是所有层共享的「基础设施」。这一层包括三样东西：调大模型 API（不管是谁要调，主循环也好、子 Agent 也好，都走这一层）、上下文压缩（后面会详细讲的五步压缩策略）、MCP 协议（和外部工具服务器通信的标准接口）。你可以把它类比成大楼的水电煤，所有楼层都需要，但谁也不会自己去铺设管道。
- **安全与治理层**有点特殊，它不像其他三层那样各管一块，而是**像一张安全网罩在所有层上面**。权限系统决定哪些操作需要用户确认、哪些可以自动执行；Hook 系统允许在工具执行前后插入自定义行为（比如「每次 git push 前自动跑 lint」）；Bash 安全模块会对 Shell 命令做语法级别的分析，检测命令注入、路径逃逸等危险模式，而不是简单地用正则匹配关键词。

### 三、Agent 工作模式

搞清楚了四层架构的宏观布局之后，一个自然的问题来了：引擎层那个主循环里，到底发生了什么？Agent 是怎么「思考」和「行动」的？它用的是什么 Agent 框架？是大家常说的 ReAct 模式吗？

这个问题值得深入聊聊，因为 Agent 的工作模式决定了整个系统的架构走向。

Claude Code 的答案可能出乎你的意料，它没有用 ReAct，而是用了一个更简洁、更高效的模式。

#### 什么是 ReAct

如果你接触过 Agent 开发，大概率听说过 ReAct（Reasoning + Acting）。它是 2022 年提出的一种 Agent 范式，核心思路是把 Agent 的每一步拆成三个阶段：

<figure>

</figure>

具体来说，模型在每一轮都会先输出一段「思考」（Thought），比如「我需要先读取 config.ts 文件来了解数据库连接配置」；然后选择一个工具调用（Action）；最后拿到工具结果（Observation）。这三步不断循环，直到模型认为任务完成。

这个模式在 2022 年非常流行，因为当时的大模型（GPT-3.5 时代）推理能力有限，需要用显式的「Thought」步骤来引导模型一步步思考。

<figure>

</figure>

但 ReAct 有几个问题：

- **第一个问题：Token 浪费。** 每一轮都要输出一段 Thought 文本，这些文本要作为上下文的一部分发给 API，占用了宝贵的 Token 预算。对于编程 Agent 来说，一次任务可能循环 50 轮，每轮都写一段「我打算先读取……然后分析……」的思考过程，加起来就是好几万 Token 的浪费。
- **第二个问题：应用层代码太复杂。** 你需要解析模型的输出，区分「哪部分是 Thought、哪部分是 Action」，然后提取 Action 调用工具，再把 Observation 拼回去。这个解析过程写起来很麻烦，而且很容易出 bug，因为模型输出的格式不一定标准，一崩就全崩了。
- **第三个问题：ReAct 是为「弱模型」设计的。** 当大模型的推理能力不够强时，用显式的 Thought 来「强迫」它一步步思考是有意义的。但 Claude Opus 这种级别的模型，推理能力已经足够强了，它完全可以在内部完成推理，不需要在输出里显式写出每一步的思考过程。

#### Tool-Use Loop

Claude Code 没有采用 ReAct 的 Thought-Action-Observation 三步循环，而是用了一个更简洁的模式，我把它叫做 **Tool-Use Loop**。

核心思路非常简单，就一个 `while(true)` 循环：

<figure>

</figure>

看到区别了吗？**没有 Thought 步骤**。

模型在内部完成推理（通过 Extended Thinking，这是 Claude Opus 的一个能力，模型在生成回复前会在内部进行一段不可见的深度推理，不占用上下文空间），然后直接返回两种结果之一：

- `**tool_use**`：「我要用某个工具」，应用层执行工具，把结果拼入消息列表，继续循环
- `**end_turn**`：「我说完了」，跳出循环，把最终结果返回给用户

<figure>

</figure>

这个设计的核心哲学是：**信任模型的推理能力，保持应用层框架尽可能简单**。

来看 `query.ts` 中的核心循环，它的实际代码长这样（这是一段 TypeScript 代码，其中 `yield` 的作用是流式输出，你可以理解为一边接收 API 的响应，一边把每个 token 实时传给 UI 显示）：

```
async function* queryLoop(
  params: QueryParams,
  consumedCommandUuids: string[],
): AsyncGenerator<StreamEvent | Message, Terminal> {
  let state: State = { messages, toolUseContext, turnCount: 1, ... }

  while (true) {
    // 步骤 1：压缩上下文（五步从轻到重）
    // 步骤 2：调用大模型 API，流式接收
    for await (const event of streamAPI(params)) {
      yield event  // 流式输出每个 token
    }
    // 步骤 3：分析模型返回
    if (response.stopReason === 'end_turn') break  // 完成了，跳出循环

    // 步骤 4：执行工具调用（并发/串行编排）
    const toolResults = await executeToolCalls(toolUseMessages)

    // 步骤 5：更新 state，继续循环
    state = { ...state, messages: updatedMessages, turnCount: turnCount + 1 }
    continue
  }
}
```

注意 `break` 和 `continue`，模型说 `end_turn` 就 `break` 跳出循环，说 `tool_use` 就 `continue` 回到循环开头。整个决策逻辑就这么简单。

#### 为什么比 ReAct 更好

你可能会问：不就是把 Thought 去掉了吗，有什么了不起的？区别其实很大，我列了三个关键原因：

**第一，Extended Thinking 让推理在「模型内部」完成。** Claude Opus 支持 Extended Thinking，模型在生成最终回复之前，会在内部进行一段不可见的深度推理。这段推理发生在模型内部，不占用应用的上下文空间，也不需要应用层去解析。所以 ReAct 的 Thought 步骤在 Claude 的架构里是多余的，模型已经在内部「想好了」，不需要在外部输出中再写一遍。

**第二，API 原生支持 tool_use。** Claude 的 API 原生支持工具调用，模型可以直接返回 `tool_use` 类型的响应，不需要用正则表达式从文本中提取「Action」。这消除了 ReAct 的格式解析问题，应用层代码变得极其简洁。

**第三，end_turn 作为天然的终止信号。** ReAct 需要一套额外的规则来判断「Agent 是否完成了」，比如检测输出中是否包含「Final Answer」。而 Tool-Use Loop 用模型的 `end_turn` 信号作为终止条件，这是 API 层面的原语，语义清晰，不需要任何解析。

用一个表格来总结两者的区别：

| **维度**   | **ReAct**                     | **Tool-Use Loop**          |
|------------|-------------------------------|----------------------------|
| 推理方式   | 显式 Thought 文本             | 模型内部 Extended Thinking |
| 工具调用   | 解析文本提取 Action           | API 原生 tool_use          |
| 终止判断   | 检测 「Final Answer」         | API 原生 end_turn          |
| Token 开销 | 每轮要输出 Thought            | 无额外开销                 |
| 编排复杂度 | 高（需要解析 Thought/Action） | 低（只需要 if/else）       |
| 适合场景   | 弱模型 + 简单工具             | 强模型 + 复杂工具集        |

Claude Code 的 Agent 工作模式可以总结为一句话：**信任模型的推理能力，把应用层框架做得尽可能简单**。

ReAct 的设计哲学是「帮模型思考」，用显式的 Thought 步骤引导模型一步步推理。这在弱模型时代是必要的。

但 Claude Code 面对的是 Opus 级别的强模型，它的推理能力完全可以在内部完成，不需要应用层去「教」它怎么想。

所以 Claude Code 的 Tool-Use Loop 只做最简单的事：调 API、执行工具、再调 API。推理交给模型，执行交给工具，编排交给最简单的 `while(true)` 循环。

这种「大道至简」的设计，反而是最高效的。

#### Plan Mode

Claude Code 不仅有 Tool-Use Loop 这种「边想边做」模式，还有 **Plan Mode**，一个更精细的两阶段工作流：先规划、再执行。

<figure>

</figure>

Plan Mode 的核心思想是：**复杂任务应该先规划再执行，避免方向跑偏、浪费精力**。它并不是一个独立的框架，而是在同一个 Tool-Use Loop 中通过 `EnterPlanMode` 和 `ExitPlanMode` 两个工具实现的：

<figure>

</figure>

整个流程分三步：

- **第一步：模型自主进入或用户手动触发。** 当模型判断「这是一个复杂任务」时，它会调用 `EnterPlanMode` 工具。对于简单任务（修 typo、加 console.log），则明确不进入。用户也可以通过 Shift+Tab 手动切换。
- **第二步：只读探索 + 设计方案。** 进入 Plan Mode 后，权限降为只读，模型只能用 Read、Grep、Glob 这些工具去探索代码库，不能写文件、不能改代码、不能跑命令。探索完后，把计划写入 `.claude/plans/` 目录。每 5 轮对话，系统会偷偷给模型塞一张「小纸条」，提醒它「你现在还在 Plan Mode，别手痒改代码」，防止模型在长对话中「走神」。
- **第三步：用户审批后实施。** 模型调用 `ExitPlanMode`，此时需要用户确认。用户批准后，权限恢复为之前的模式，模型开始自由执行读写操作，按计划实施。

Plan Mode 最值得学习的设计是\*\*「工具即能力」\*\*。

对模型来说，Plan Mode 不是一种特殊的「模式切换」，而只是调用了 `EnterPlanMode` 和 `ExitPlanMode` 这两个工具。

就像调用 Read 工具读文件一样自然。整个过程不需要引擎层做任何特殊处理，`query()` 仍然只是一个简单的 `while(true)` 循环。

### 四、System Prompt 的构造

System Prompt 就是 Claude Code 的灵魂，它定义了 Agent 的身份、行为规范、可用工具、安全约束……一切。

但 Claude Code 的 System Prompt 不是一个静态的文本文件。它是**动态组装**的，由十几个 Section 拼接而成，而且在组装过程中做了非常精巧的**缓存优化**。

<figure>

</figure>

我们先来看一下，Claude Code 的 System Prompt 到底长什么样，它是怎么「调教」大模型变成一个靠谱的编程 Agent 的。

*注：Claude Code 源码中所有 Prompt 原文均为英文。为了让大家更好地理解设计思路，下面展示的 Prompt 内容我翻译成了中文，并保留了关键术语的英文原文。*

#### 角色定义与安全红线

每个 Agent 的 System Prompt 都要回答一个根本问题：你是谁？Claude Code 的开场是这样的：

```
你是一个交互式代理（interactive agent），帮助用户完成软件工程任务。
请使用下面的指令和可用的工具来协助用户。

重要：你绝对不能为用户生成或猜测 URL，除非你确信这些 URL
是为了帮助用户完成编程任务。你可以使用用户在消息或本地文件中
提供的 URL。
```

注意两个关键点。第一，它把自己定位为「interactive agent」，而不是「assistant」或「chatbot」，这从一开始就暗示了模型应该主动采取行动，而不是被动回答。第二，立刻划了安全红线：不能乱编 URL。这

看起来是个小事，但对编程 Agent 非常重要，如果模型瞎编一个 npm 包的 URL，用户执行了就可能中招。

紧接着是一段安全约束指令，这段话非常值得每个 Agent 开发者抄作业：

```
重要：允许协助已授权的安全测试、防御性安全研究、CTF 挑战赛
和教育场景。拒绝涉及破坏性技术、DoS 攻击、大规模目标扫描、
供应链攻击或用于恶意目的的检测规避请求。
```

这段 Prompt 没有用「绝对不能做 X」的口吻，而是先说「可以做什么」（授权的安全测试、CTF 挑战），再划定「不能做什么」（DoS、供应链攻击）。

这种「先肯定再约束」的写法，比纯禁止清单效果好得多，它给了模型清晰的判断依据，而不是一堆模糊的红线。

#### 行为准则

接下来是一大段关于「怎么做事」的行为指南，这部分是 Claude Code System Prompt 的精华。我挑几条最值得学习的：

**关于修改代码前先阅读：**

```
一般来说，不要对你没有阅读过的代码提出修改建议。如果用户
要求你查看或修改某个文件，先读一遍它。在提出修改建议之前，
先理解现有代码。
```

这条看起来简单，但解决了 Agent 的一个常见问题：很多 Agent 会根据用户描述直接生成代码，而不先看看现有代码是什么样的，结果经常和项目风格不一致或者引入重复实现。

**关于代码风格：「少即是多」：**

```
不要在用户要求之外添加功能、重构代码或进行"改进"。修一个 bug
不需要顺手清理周围的代码。一个简单功能不需要额外的可配置性。

不要为一次性操作创建辅助函数、工具类或抽象层。
三行相似的代码比一个过早的抽象更好。
```

这个设计思路太重要了。

如果你用过 Agent 写代码，你一定遇到过这种情况：你让它修一个 bug，它顺手把整个文件重构了，加了一堆你没要求的类型标注和错误处理。Claude Code 在 Prompt 里明确禁止了这种行为。

**关于失败处理：「先诊断再换方案」：**

```
如果某个方案失败了，先诊断原因再决定是否换方案——读报错信息、
检查你的假设、尝试有针对性的修复。不要盲目重试完全相同的操作，
但也不要因为一次失败就放弃一个可行的方案。
```

这条解决了 Agent 的另一个常见问题，「摆烂式重试」或「草率放弃」。Claude Code 要求模型先搞清楚为什么失败了，再决定是修复还是换方案，而不是两个极端。

#### 操作安全

Claude Code 对「什么操作需要用户确认」做了非常详细的规定。我建议每个 Agent 开发者都研读这段 Prompt：

```
仔细考虑操作的可逆性（reversibility）和影响范围（blast radius）。
一般来说，你可以自由执行本地的、可逆的操作，比如编辑文件或
运行测试。但对于难以撤销、影响共享系统或有风险的操作，
请先和用户确认后再执行。

需要用户确认的高风险操作示例：
- 破坏性操作：删除文件/分支、删表、rm -rf
- 难以逆转的操作：force-push、git reset --hard、修改已发布的 commit
- 对他人可见的操作：推送代码、创建/关闭 PR、发送消息
- 上传到第三方工具：内容可能被缓存或索引，即使删除也无法撤回
```

这段的核心思想是用**可逆性**和**影响范围**两个维度来判断风险。读文件、改本地代码是低风险的（可逆、只影响本地），直接放行。`git push`、发 Slack 消息是高风险的（不可逆、影响他人），必须确认。

然后还有一句非常精妙的补充：

```
用户批准了某个操作（比如 git push）一次，并不意味着他在所有
场景下都批准这个操作。授权仅对指定的范围有效，不能超出范围。
```

这解决了「权限蔓延」的问题，用户同意了一次 push 不代表以后都自动 push，**授权是一次性的、有范围的**。这个原则在 Agent 权限设计中非常重要。

#### 工具使用指南

```
当有专用工具可用时，不要用 Bash 来执行命令。使用专用工具可以
让用户更好地理解和审查你的工作。这一点至关重要：

- 读取文件用 Read 工具，而不是 cat、head、tail 或 sed
- 编辑文件用 Edit 工具，而不是 sed 或 awk
- 创建文件用 Write 工具，而不是 echo 重定向
- 搜索文件用 Glob 工具，而不是 find 或 ls
- 搜索内容用 Grep 工具，而不是 grep 或 rg
```

这条规则的设计动机值得深思。为什么不让模型直接用 `cat` 读文件、用 `sed` 改代码？技术上完全可以。

原因是**可审查性**。当模型调用 `Read` 工具读文件时，UI 会清晰地展示「Agent 正在读取 src/index.ts」。但如果模型执行 `cat src/index.ts`，用户看到的只是一条 Bash 命令和一大坨输出，完全不知道 Agent 在干什么。

而且，专用工具有专用的权限检查，`Read` 工具会检查文件路径是否在允许范围内，而 `cat` 命令就没有这层保护了。所以「用专用工具而不是 Bash」不仅是体验问题，更是安全问题。

#### Git 安全协议

Claude Code 对 Git 操作有一套非常严格的安全协议，这段 Prompt 写得极其细致：

```
Git 安全协议：
- 绝不修改 git config
- 绝不执行破坏性 git 命令（push --force、reset --hard、
  checkout .、clean -f），除非用户明确要求
- 绝不跳过 hooks（--no-verify），除非用户明确要求
- 绝不 force push 到 main/master 分支，如果用户要求则发出警告

关键：始终创建新的 commit（NEW commit），而不是用 --amend 修改。
当 pre-commit hook 失败时，commit 实际上并没有发生——所以
--amend 会修改上一个（不相关的）commit，可能导致代码丢失。
正确做法是：修复问题后创建一个新的 commit。
```

最后一条关于 `--amend` 的警告特别值得注意。

很多人（包括一些 Agent 实现）在 commit 失败后会习惯性地 `git commit --amend`。但如果失败原因是 pre-commit hook 拒绝了，那么 commit 实际上没发生！

这时候 `--amend` 会修改上一个（不相关的）commit，可能导致代码丢失。这种微妙的 bug 很难被发现，Claude Code 直接在 Prompt 里防住了。

#### 输出风格约束

Claude Code 对模型的输出风格也有严格规定：

```
# 输出效率
直奔重点。先尝试最简单的方案。要极度简洁。
工具调用之间的文字不超过 25 个词。最终回复不超过 100 个词。

先给出答案或行动，而不是推理过程。跳过填充词、开场白和
不必要的过渡句。不要复述用户说过的话——直接做就行。
```

25 个词的限制非常苛刻，这意味着模型在两次工具调用之间，基本只能说一句话。这个设计的目的是避免 Agent 「话痨」，没人想看 Agent 在每次读文件前先写一段「让我来看看这个文件的内容……」的废话。

#### 环境信息注入

每次对话开始时，Claude Code 会把当前环境信息注入 System Prompt：

```
# 环境信息
- 主工作目录：/Users/you/my-project
- 是否为 Git 仓库：是
- 操作系统平台：darwin (macOS)
- Shell 类型：zsh
- 当前模型：Claude Opus 4.6 (1M context)
- 知识截止日期：2025 年 5 月
```

这些信息让模型知道自己「在哪里」，是什么操作系统、什么 Shell、什么项目目录。没有这些信息，模型可能会在 macOS 上执行 `apt-get install`，或者在 zsh 环境里用 bash 语法。

#### 分割线与三级缓存

了解了各个 Section 的内容，我们回到一个很实际的问题：**这些 Section 是怎么组装到一起的？为什么组装方式会影响费用？**

先看一段组装后的 System Prompt 长什么样（简化版）：

```
┌─────────────────────────────────────────────────┐
│  [角色定义] 你是一个交互式代理，帮助用户完成...    │  ← 所有用户完全一样
│  [安全红线] 重要：允许协助已授权的安全测试...       │  ← 所有用户完全一样
│  [行为准则] 一般来说，不要对你没有阅读过的代码...   │  ← 所有用户完全一样
│  [操作安全] 仔细考虑操作的可逆性...               │  ← 所有用户完全一样
│  [工具使用] 当有专用工具可用时...                  │  ← 所有用户完全一样
│  [Git 安全] 绝不修改 git config...               │  ← 所有用户完全一样
│  [输出风格] 直奔重点，要极度简洁...               │  ← 所有用户完全一样
├────── __SYSTEM_PROMPT_DYNAMIC_BOUNDARY__ ────────┤
│  [环境信息] 主工作目录: /Users/you/my-project    │  ← 每个用户不一样
│  [CLAUDE.md] 本项目使用 TypeScript + Jest...      │  ← 每个项目不一样
│  [记忆指令] 你有一个持久记忆系统...               │  ← 每次对话可能不一样
│  [MCP 指令] 你已连接 GitHub MCP server...         │  ← 每个用户不一样
└─────────────────────────────────────────────────┘
```

<figure>

</figure>

看到中间那条粗线了吗？那就是 Claude Code 在 System Prompt 中插入的分割标记 `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__`。

**分割线之上的内容，对所有用户都完全一样。** 不管你是北京的 Java 工程师还是纽约的 Python 开发者，你看到的「角色定义」「行为准则」「Git 安全协议」这些内容是一模一样的。

**分割线之下的内容，每个用户都不同。** 你的工作目录、你的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>、你的记忆文件、你连接的 MCP 服务，这些是因人而异的。

<figure>

</figure>

为什么要这么分？因为 Claude API 有一个 **Prompt Cache** 机制：如果两次请求的 Prompt 前缀完全相同，API 会复用上次的计算结果，**费用可以降低 90%**。对于几万 Token 的 System Prompt 来说，缓存命中与否意味着每次请求几美分和几美元的差距。

分割线之上的内容对所有用户都一样，所以可以**全球所有用户共享同一份缓存**——你用的和东京的开发者用的是同一份。而分割线之下的内容因人而异，没法共享，只能实时生成。

这就是 Claude Code 的三级缓存体系：**全局缓存**（分割线之上，跨组织跨用户共享）→ **组织缓存**（同一组织内跨会话共享）→ **会话缓存**（同一个 Section 在一次会话内只计算一次）。每一级都在帮 API 省钱。

#### 小结

回过头来看 Claude Code 的 System Prompt，你会发现它其实在做一件事：**用最小的 Token 成本，给模型划出最清晰的行为边界**。

怎么划的呢？我总结了三个最值得抄作业的设计。

第一个是「先给范围再画红线」。比如安全约束那段，它不是一上来就说「不准做这不准做那」，而是先说「安全测试、CTF 挑战这些可以做」，然后再说「DoS、供应链攻击这些不能做」。这比你写十句「不准 XX」管用得多，因为模型拿到了判断标准，而不是一堆模糊的禁令。

<figure>

</figure>

第二个是「用两个维度把风险分出层次」。Claude Code 判断一个操作安不安全，不看它「看起来危不危险」，而是看两件事：这操作能撤回吗？会影响别人吗？改本地代码当然能撤回、只影响自己，直接放行。`git push` 撤不回来、别人能看到，那就得确认。这个思路比笼统的「危险/安全」二分法精细太多了。

<figure>

</figure>

第三个是「静态内容和动态内容用分割线隔开」。那条分割线不是随便画的，它把所有用户都一样的部分和因人而异的部分切开了。这样做的好处是，分割线之上的内容可以被全球所有用户共享缓存，每次 API 调用能省 90% 的费用。一个看似简单的排版调整，背后是实打实的成本优化。

<figure>

</figure>

### 五、记忆系统

每次启动 Claude Code 都是一个全新的会话，模型不记得上次对话的任何内容。但用户的偏好、项目背景、行为反馈，这些信息需要跨会话保持。

这个问题看起来简单，做起来却非常难。业界常见的方案是用向量数据库，把记忆存成 embedding，每次对话时做相似度检索。

<figure>

</figure>

但 Claude Code 没有这么做。为什么？

因为 Agent 需要记住的大部分不是「相似的文档片段」，而是「用户说过'不要 mock 数据库'」这种**结构化的行为指令**。

用向量相似度去检索「不要 mock 数据库」这句话，效果其实很差，它可能匹配到一堆包含「数据库」关键词的无关内容，真正重要的行为反馈却被淹没了。

Claude Code 设计了一套完全不同的记忆系统，我们来一层一层拆解。

#### 记什么：四类型分类

Claude Code 把记忆分成了**四种明确的类型**：

```
export const MEMORY_TYPES = [
  'user',      // 用户画像：角色、偏好、知识水平
  'feedback',  // 行为反馈：该做什么、不该做什么
  'project',   // 项目动态：在做什么、截止日期、协作信息
  'reference', // 外部指针：哪里能找到什么信息
] as const
```

注意，只有这四种，**不能随便加新的**。

<figure>

</figure>

为什么不搞一个通用的「any」类型什么都能存？因为无约束的记忆会迅速膨胀成垃圾堆。

限定四种类型，就是在逼 Agent 做分类决策。每存一条记忆，它必须想清楚「这到底属于哪一类」，而不是一股脑往里塞。

我逐个解释一下这四种类型的设计意图。

**User（用户画像）**是最个人化的一类，记住用户是谁、擅长什么、知识水平如何。比如用户说「我是一个写了十年 Go 的后端工程师，第一次接触 React」，Agent 就应该在解释前端概念时用后端的类比，而不是从零讲起。这类记忆让 Agent 的回答**因人而异**，而不是千篇一律。

<figure>

</figure>

\*\*Feedback（行为反馈）\*\*是最重要的一类，记住用户说过「不要做什么」和「做得好继续保持」。这类记忆的关键在于，它不仅记规则本身，还要求记录 **Why（为什么）** 和 **How to apply（怎么应用）**：

```
规则本身：集成测试必须使用真实数据库，不能用 mock
Why：上季度 mock 测试全部通过但生产环境迁移失败了
How to apply：在这个模块写测试时，始终连接真实数据库
```

为什么一定要记 Why？因为光记住「不要 mock 数据库」是不够的。如果遇到一个边缘情况，比如一个纯单元测试不涉及数据库迁移，Agent 需要根据 Why 来判断「这条规则在这个场景下是否适用」。没有 Why，Agent 只能盲目遵守，可能在不该用真实数据库的地方也强行连接。

<figure>

</figure>

\*\*Project（项目动态）\*\*记的是「正在发生什么」，谁在做什么、截止日期是什么、有什么重要决策。这类记忆有一个特殊要求：**必须把相对日期转成绝对日期**。用户说「周四之前冻结合并」，Agent 要存成「2026-03-05 之前冻结合并」，因为「周四」过几天就没意义了，但「2026-03-05」永远准确。

<figure>

</figure>

**Reference（外部指针）**记的是「去哪找什么信息」，Bug 在 Linear 的哪个项目里追踪、Grafana 看板的地址是什么、Slack 的哪个频道能问到相关的人。这类记忆的价值在于，Agent 不需要知道外部系统的具体内容，只需要知道**去哪里找**。

<figure>

</figure>

#### 不记什么：排除清单

Claude Code 明确规定了**什么不应该存到记忆里**，这个设计和「记什么」同样重要。

<figure>

</figure>

首先是代码模式、项目架构和文件结构这些信息，通过 `grep`、`git`、`CLAUDE.md` 就能获取，存在记忆里反而会导致记忆和代码实际状态不一致。

然后是 Git 历史和最近的改动，`git log` 和 `git blame` 才是权威来源，不需要记忆系统再来存一遍。调试方案和修复方法也不存，因为修复已经在代码里了，commit 消息已经记录了上下文。

<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里已经写了的内容也不存，避免重复。最后是临时任务状态和当前对话上下文，这些是会话级的信息，不需要跨会话保持。

这个排除清单背后的核心原则是：**可以从当前代码推导出来的信息，一律不存**。

因为代码是「活的」，它随时在变，但记忆是「死的」，它存下来就定格了。如果记忆说「`AuthService 在 src/auth.ts 第 42 行`」，但代码已经重构了，那这条记忆就变成了一个「权威的错误」，比没有记忆还糟糕。

#### 怎么存：索引 + 独立文件

搞清楚了「记什么」和「不记什么」，接下来看「怎么存」。

每条记忆存为一个独立的 `.md` 文件，文件开头有一段 YAML 格式的元信息（你可以理解为这条记忆的「身份证」）：

```
---
name: no-mock-database
description: 集成测试必须使用真实数据库，不能用 mock
type: feedback
---

集成测试必须使用真实数据库，不能用 mock。

**Why:** 上季度 mock 测试全部通过但生产环境迁移失败了。
**How to apply:** 在这个模块写测试时，始终连接真实数据库。
```

文件开头那段 YAML 格式的元信息里，三个字段各有用途：`name` 是人类可读的标识；`description` 是一句话摘要，**专门用于检索时的相关性匹配**（后面会讲到）；`type` 标记四类型之一。

然后有一个 `MEMORY.md` 文件作为索引，它是一个不超过 200 行（25KB）的轻量目录：

```
- [No Mock Database](feedback_no_mock_db.md) — tests must use real DB
- [User Preferences](user_preferences.md) — prefers terse responses
- [Auth Rewrite](project_auth_rewrite.md) — driven by compliance, not tech debt
```

注意这个 200 行的硬性上限。为什么要限制？来看源码里的截断逻辑：

```
export const MAX_ENTRYPOINT_LINES = 200
export const MAX_ENTRYPOINT_BYTES = 25_000  // 25KB

export function truncateEntrypointContent(raw: string): EntrypointTruncation {
  // 同时检查行数和字节数上限
  const wasLineTruncated = lineCount > MAX_ENTRYPOINT_LINES
  const wasByteTruncated = byteCount > MAX_ENTRYPOINT_BYTES

  if (wasLineTruncated || wasByteTruncated) {
    // 截断并附加警告
    return {
      content: truncated + '\n\n> WARNING: MEMORY.md 太大了...',
      // ...
    }
  }
}
```

它同时检查**行数**和**字节数**两个维度。为什么要两个？因为有人可能写了 199 行，每行 500 字，行数没超但字节数爆了。双重检查堵住了这个漏洞。

现在来看整个存储架构的关键设计：**<a href="http://MEMORY.md" target="_blank" rel="noopener noreferrer">MEMORY.md</a> 索引始终被加载到 System Prompt 里**，但独立记忆文件**按需加载**。

<figure>

</figure>

这解决了一个经典矛盾，如果把所有记忆都塞进 System Prompt，50 条记忆就可能占满上下文；如果完全不塞，Agent 又不知道有哪些记忆可用。

索引文件两全其美：Agent 看到索引就知道有哪些记忆，但只加载真正相关的那几条。

#### 怎么召回：Sonnet 当秘书

存好了记忆，关键问题来了：每次对话时，怎么从几十条记忆里挑出最相关的那几条加载进来？

Claude Code 的做法非常巧妙，用一个廉价的小模型（Sonnet）来做记忆检索。

<figure>

</figure>

整个召回流程分为三步：

第一步：扫描所有记忆文件的「头部信息」

```
export async function scanMemoryFiles(
  memoryDir: string,
  signal: AbortSignal,
): Promise<MemoryHeader[]> {
  const entries = await readdir(memoryDir, { recursive: true })
  const mdFiles = entries.filter(
    f => f.endsWith('.md') && basename(f) !== 'MEMORY.md',
  )
  // 只读每个文件的前 30 行（frontmatter 区域），不读全文
  const headers = await Promise.allSettled(
    mdFiles.map(async (relativePath) => {
      const { content, mtimeMs } = await readFileInRange(
        filePath, 0, 30,  // 只读前 30 行！
      )
      const { frontmatter } = parseFrontmatter(content)
      return {
        filename: relativePath,
        description: frontmatter.description || null,
        type: parseMemoryType(frontmatter.type),
        mtimeMs,  // 文件修改时间，用于后续的新旧度判断
      }
    }),
  )
  // 按修改时间倒序，最多 200 个
  return headers.sort((a, b) => b.mtimeMs - a.mtimeMs)
    .slice(0, 200)
}
```

注意它**只读每个文件的前 30 行**，足够提取文件开头那段元信息里的 name、description、type，但不会读取记忆的完整内容。这样即使有 200 个记忆文件，扫描开销也很小。

第二步：拼成清单，发给 Sonnet 做选择。

扫描完之后，所有记忆的「头部信息」被拼成一个文本清单：

```
- [feedback] feedback_no_mock.md (2026-03-28): 集成测试必须使用真实数据库
- [user] user_preferences.md (2026-03-25): 用户是后端工程师，偏好简洁回复
- [project] project_auth.md (2026-03-20): 认证模块重写由合规需求驱动
```

然后把这个清单连同用户当前的输入一起发给 Sonnet：

```
const result = await sideQuery({
  model: getDefaultSonnetModel(),
  system: '你是一个记忆选择器，从列表中选出最多 5 条与用户问题最相关的记忆...',
  messages: [{
    role: 'user',
    content: `用户问题: ${query}\n\n可用的记忆:\n${manifest}`,
  }],
  max_tokens: 256,  // 只需要返回文件名列表，非常短
})
```

Sonnet 返回的只是一个文件名列表（比如 `["feedback_no_mock.md", "project_auth.md"]`），不是记忆内容本身。

第三步：加载选中记忆的完整内容，注入上下文。

拿到文件名列表后，系统才去读取这几条记忆的完整内容，作为 `<system-reminder>` 注入当前对话。

这里还有一个非常讲究的细节，**记忆陈旧度检测**。对于超过 1 天的记忆，系统会自动附加一段警告：

```
export function memoryFreshnessText(mtimeMs: number): string {
  const d = memoryAgeDays(mtimeMs)
  if (d <= 1) return ''  // 今天或昨天的记忆不加警告
  return (
    `这条记忆已经有 ${d} 天了。` +
    `记忆是某个时间点的观察，不是实时状态——` +
    `其中关于代码行为或 file:line 引用的断言可能已经过时。` +
    `在当作事实引用之前，请先对照当前代码验证。`
  )
}
```

为什么需要这个？因为用户可能 30 天前存了一条记忆说「`AuthService 在 src/auth.ts 第 42 行使用了 JWT`」，但代码早就改了。如果模型盲目相信这条记忆，就会给出错误的建议。陈旧度警告提醒模型「这个信息可能过时了，先验证再引用」。

#### 性能优化：并行预取

最后一个值得学习的设计：记忆召回的**执行时机**。

Sonnet 侧查询不是在主模型需要时才触发的，而是**在用户提交消息后立刻就开始了**，和主模型的 API 调用并行执行：

```
// query.ts 中的调用——在进入主循环之前就启动记忆预取
using pendingMemoryPrefetch = startRelevantMemoryPrefetch(
  state.messages,
  state.toolUseContext,
)
```

时序大概是这样的：

<figure>

</figure>

Sonnet 比 Opus 快得多（延迟通常只有几百毫秒），所以等主模型的响应回来时，记忆选择早就完成了。整个记忆召回过程**几乎不增加任何额外延迟**。

还有一个小优化：如果用户当前正在使用某些工具（比如正在调用某个 MCP 工具），Sonnet 选择器会自动过滤掉该工具的使用文档类记忆，因为模型已经在用这个工具了，它的用法文档此刻是噪声，不是信号。

但「该工具的已知 bug 和注意事项」类记忆仍然会被选中，**正在用的时候，恰恰是最需要知道坑在哪里的时候**。

#### 小结

回顾一下 Claude Code 的记忆系统，它的核心设计哲学可以用三句话概括。

<figure>

</figure>

- 第一句是「记该记的，不记能推导的」。通过四类型封闭集合加上排除清单，把记忆控制在有价值的范围内，防止它膨胀成一个什么都往里塞的垃圾堆。
- 第二句是「存索引，按需加载详情」。<a href="http://MEMORY.md" target="_blank" rel="noopener noreferrer">MEMORY.md</a> 作为轻量索引始终常驻在 System Prompt 里，但每条记忆的具体内容是独立文件，用到的时候才加载。这样既让 Agent 知道有哪些记忆可用，又不会撑爆上下文。
- 第三句是「用小模型做秘书，大模型做决策」。Sonnet 负责并行预取和选择记忆，Opus 只管做决策，加上陈旧度检测机制，实现了零延迟、低成本、高可靠。

### 六、上下文窗口管理

这可能是整个 Claude Code 里最复杂也最精妙的部分。

大模型有上下文窗口限制。即使是 200K Token 的窗口，一次复杂的编程任务（读了几十个文件、执行了几十条命令）很容易就塞满了。

<figure>

</figure>

业界常见的做法是「简单截断」，只保留最近的 N 条消息，旧的扔掉。但这对于编程 Agent 来说是灾难性的：你可能 20 轮前读过一个关键配置文件，现在要改代码时那个文件的信息已经被截掉了，Agent 就会犯低级错误。

另一种做法是「全量摘要」，把整段对话总结成一段摘要。但这很贵（摘要本身就是一次 API 调用），而且有信息损失。

#### 压缩五步走

Claude Code 的核心理念是：**压缩一定有信息损失，所以能不压就不压，必须压的时候从最轻的手段开始**。

它设计了五个从轻到重的压缩手段，就像医院的分诊制度一样：先试最温和的，不行再上猛药。在每次 API 调用前依次尝试：

<figure>

</figure>

为什么要分五步，而不是一步到位做全量摘要？

因为**每一步的「代价」是递增的**。

- 第 1 层几乎没有信息损失，完整内容还在磁盘上，只是不在上下文里了。
- 第 2、3 层有少量信息损失，丢掉了老的工具输出，但模型随时可以重新获取。
- 第 4 层有中等信息损失，对话细节被分段压缩了。
- 第 5 层信息损失最大，整段对话变成一段摘要。

所以 Claude Code 的策略是：**先用代价最小的手段，实在不行再升级**。

大部分情况下，前三层就够用了，根本不需要触发昂贵的全量摘要。

接下来我们一层一层拆解。

#### 第 1 步：大结果存磁盘

**问题是什么？** 想象一下，你让 Agent 读一个 10MB 的日志文件。Read 工具忠实地返回了全部内容，一下子就吃掉了几万 Token。更夸张的是，如果模型同时读了 3 个大文件，一条消息就可能占掉大半个上下文窗口。

<figure>

</figure>

**Claude Code 怎么做？** 它在工具结果进入消息列表**之前**，就先做一道「体检」：

```
async function maybePersistLargeToolResult(
  toolResultBlock: ToolResultBlockParam,
  toolName: string,
): Promise<ToolResultBlockParam> {
  const size = contentSize(content)
  // 单个工具结果超过阈值（默认约 50KB）？
  if (size <= threshold) {
    return toolResultBlock  // 没超，原样通过
  }
  // 超了！把完整内容存到磁盘文件
  const result = await persistToolResult(content, toolUseId)
  // 用一个 2KB 的预览替换原内容
  const preview = buildLargeToolResultMessage(result)
  return { ...toolResultBlock, content: preview }
}
```

它的逻辑很简单：如果单个工具的结果超过约 50KB，就把完整内容写到磁盘上，在消息里只留一个 2KB 的预览摘要。这样模型还是能看到文件的大概内容（前 2KB），但不会撑爆上下文。

<figure>

</figure>

除了单个工具的限制，还有一个**消息级的总量控制**，同一条消息里所有工具结果的总大小不能超过 200KB。如果超了，系统会挑出最大的那几个结果存磁盘，直到总量降到限制以内。

这一层的精妙之处在于：**完整内容并没有丢**，它还在磁盘上。如果模型后面真的需要那个大文件的某个片段，它可以再次调用 Read 工具去读取特定的行范围。

#### 第 2 步：砍掉远古消息

**问题是什么？** 一次长对话可能有上百轮。对话开头那几轮的内容，比如用户最初的探索性提问、模型早期的试探性回答，到了后面几乎完全没用了。但它们仍然占着宝贵的上下文空间。

<figure>

</figure>

**Claude Code 怎么做？** Snip 是最「粗暴」但也最高效的一层，直接把对话开头的一批老消息移除掉，然后插入一个边界标记告诉模型「这之前的内容已经被清理了」。

```
if (feature('HISTORY_SNIP')) {
  const snipResult = snipModule.snipCompactIfNeeded(messagesForQuery)
  messagesForQuery = snipResult.messages
  snipTokensFreed = snipResult.tokensFreed
  if (snipResult.boundaryMessage) {
    yield snipResult.boundaryMessage  // 插入边界标记
  }
}
```

它不做任何摘要，不总结「前面聊了什么」，直接砍掉。听起来很暴力，但对于那些确实已经完全过时的消息来说，这是代价最低的做法，因为它**不需要额外调用大模型来生成摘要**，零 API 开销。

<figure>

</figure>

还有一个重要的细节：Snip 会把「我释放了多少 Token」这个数字（`snipTokensFreed`）传给后面的第 5 层 Auto-Compact。

为什么？因为 Auto-Compact 是根据「当前上下文占了多少 Token」来决定是否触发的。如果 Snip 已经释放了足够的空间，Auto-Compact 就不需要触发了，**避免两层同时做无谓的压缩**。

#### 第 3 步：裁剪老的工具输出

**问题是什么？** 经过前两层之后，上下文里剩下的都是「不太老但也不太新」的消息。

这些消息不能直接砍掉（可能还有用），但里面大量的工具输出其实已经过时了，比如 30 分钟前读的一个文件，现在那个文件可能已经被改过了。

<figure>

</figure>

**Claude Code 怎么做？** Micro-Compact 的核心思想是**时间衰减**：越老的工具结果越不重要，可以被裁剪。但是，不是所有工具的结果都能裁剪：

```
const COMPACTABLE_TOOLS = new Set([
  FILE_READ_TOOL_NAME,    // 读文件 → 可以重新读
  ...SHELL_TOOL_NAMES,    // 执行命令 → 可以重新执行
  GREP_TOOL_NAME,         // 搜索 → 可以重新搜
  GLOB_TOOL_NAME,         // 查找文件 → 可以重新查
  WEB_SEARCH_TOOL_NAME,   // 搜索网页 → 可以重新搜
  FILE_EDIT_TOOL_NAME,    // 编辑文件 → 结果可裁剪
  FILE_WRITE_TOOL_NAME,   // 写文件 → 结果可裁剪
])
```

看到规律了吗？**可以被裁剪的，都是「可重新获取」的工具**，Read 的结果可以再读一次，Bash 的输出可以再执行一次，搜索结果可以再搜一次。

但 AgentTool（子 Agent 的输出）、TaskTool（任务状态）这类工具的结果**永远不会被裁剪**，因为子 Agent 的推理过程是不可重复的，砍掉就真的丢了。

<figure>

</figure>

具体裁剪逻辑是「保留最近 N 个，清理其余的」：

```
// 收集所有可裁剪工具的结果 ID
const compactableIds = collectCompactableToolIds(messages)
// 保留最近 5 个，其余全部清理
const keepRecent = Math.max(1, config.keepRecent)  // 至少保留 1 个
const keepSet = new Set(compactableIds.slice(-keepRecent))
const clearSet = compactableIds.filter(id => !keepSet.has(id))
```

被裁剪的工具结果会被替换成一个标记：

```
export const TIME_BASED_MC_CLEARED_MESSAGE =
  '[Old tool result content cleared]'
```

这样模型看到这个标记就知道「这里原来有内容但被清理了」。如果它后面还需要这些信息，它可以自己决定重新读文件或重新执行命令。

为什么叫「时间衰减」？因为它的触发条件跟时间有关，当距离上一次 API 调用超过一定时间（默认约 60 分钟），说明大模型 API 端的 Prompt Cache 大概率已经过期了。既然缓存已经没了，那清理旧的工具结果也不会浪费之前的缓存投入。

#### 第 4 步：读时投影

**问题是什么？** 经过前三层后，如果上下文还是太大，下一步就得做全量摘要了。但全量摘要代价很高（要额外调一次 API），而且会把整段对话的细节全部丢掉。有没有一个「中间态」，比全量摘要轻，但比 Micro-Compact 重？

<figure>

</figure>

**Claude Code 怎么做？** Context Collapse 引入了一个非常巧妙的概念，**读时投影（Read-Time Projection）**。

什么意思呢？前面三层都是「写时压缩」，直接修改消息列表，把内容替换掉或删掉。但 Context Collapse 不修改原始消息，它只在**调用 API 的那一刻**，动态计算一个「压缩视图」给模型看。

```
// 这是 query.ts 中的调用
// 注意：这是一个"读时投影"——不修改 REPL 的完整历史，
// 只在发送给 API 时计算压缩视图
if (feature('CONTEXT_COLLAPSE') && contextCollapse) {
  const collapseResult = await contextCollapse.applyCollapsesIfNeeded(
    messagesForQuery,
    toolUseContext,
    querySource,
  )
  messagesForQuery = collapseResult.messages
}
```

它的触发有两级阈值：

- **90% 上下文窗口**：主动开始分段压缩旧消息（预留缓冲区）
- **95% 上下文窗口**：紧急压缩更多内容（留足 API 响应空间）

这个设计最精妙的地方是它和第 5 层的配合：**Context Collapse 运行在 Auto-Compact 之前**。

如果 Context Collapse 已经通过「读时投影」把上下文压到了阈值以下，Auto-Compact 就完全不需要触发了。这样模型保留了更多的细节上下文，而不是被一段粗糙的全量摘要替代。

#### 第 5 步：全量摘要

**问题是什么？** 当前面四层都不够用，上下文实在太大了，必须做一次彻底的压缩。这是代价最高但效果最强的一层。

**什么时候触发？** Claude Code 用一个公式计算触发阈值：

```
function getAutoCompactThreshold(model: string): number {
  const effectiveContextWindow = getEffectiveContextWindowSize(model)
  // 有效窗口 - 13K 缓冲区 = 触发阈值
  return effectiveContextWindow - 13_000
}
```

以 200K Token 的模型为例：有效窗口大约 180K（预留 20K 给输出），减去 13K 缓冲区，**当上下文达到 167K Token 时触发**。

<figure>

</figure>

**触发后做了什么？** 三步走：

**第一步：生成摘要**。调用大模型，把整段对话总结成一段结构化摘要。这个摘要不是随便写的，Claude Code 用一个精心设计的 Prompt 要求模型按多个维度来总结：用户的主要请求和意图、关键技术概念、涉及的文件和代码片段、遇到的错误和修复方案、问题解决过程、用户的所有消息（不能遗漏任何一条）、待完成的任务、当前工作状态、建议的下一步。

<figure>

</figure>

为什么要这么细？因为压缩后模型要靠这段摘要来「恢复记忆」。如果摘要漏掉了关键信息（比如「用户还有一个待完成的任务」），模型就会忘记这件事。

**第二步：替换旧消息**。把压缩边界之前的所有消息删掉，替换为刚才生成的摘要。同时插入一条边界标记消息，记录压缩前的 Token 数，方便后续追踪。

<figure>

</figure>

**第三步：Post-Compact Restoration（压缩后恢复）**。这是整个流程中最关键的一步，压缩完不是就完了，还要**主动恢复最重要的上下文**：

```
export const POST_COMPACT_MAX_FILES_TO_RESTORE = 5
export const POST_COMPACT_TOKEN_BUDGET = 50_000
export const POST_COMPACT_MAX_TOKENS_PER_FILE = 5_000
export const POST_COMPACT_SKILLS_TOKEN_BUDGET = 25_000
```

系统会从文件状态缓存（`fileStateCache`）中找出最近访问过的文件，按最后访问时间排序，挑选最多 5 个、总共不超过 50K Token 的文件内容重新注入。同时恢复活跃的 Skill（不超过 25K Token），如果有进行中的 Plan 也会恢复 Plan 文件。

<figure>

</figure>

为什么要做恢复？因为压缩后模型「失忆」了，它不记得刚才读过的文件内容了。

如果不恢复，模型的第一反应就是「让我重新读一下文件」，白白浪费一轮工具调用。主动恢复最近的文件内容，可以让模型**无缝继续工作**，体验上几乎感觉不到压缩发生过。

还有一个兜底机制：如果全量摘要连续失败 3 次（比如 API 超时），系统会自动放弃，不会无限重试，这就是**熔断器** 模式，防止一个失败的压缩操作拖垮整个 Agent。

#### 小结

回顾一下这五步压缩策略，它们体现了一个核心设计哲学：**能轻则轻，逐步加码**。

| **层级** | **手段**       | **信息损失** | **API 开销**        | **触发条件**      |
|----------|----------------|--------------|---------------------|-------------------|
| 第 1 层  | 大结果存磁盘   | 几乎为零     | 零                  | 工具结果超 50KB   |
| 第 2 层  | 砍掉远古消息   | 低           | 零                  | 消息过时          |
| 第 3 层  | 清理老工具输出 | 中低         | 零                  | 缓存过期/数量超限 |
| 第 4 层  | 读时投影压缩   | 中           | 低                  | 上下文达 90%      |
| 第 5 层  | 全量摘要       | 高           | 高（一次 API 调用） | 上下文达 ~93%     |

越往下代价越高，但效果也越强。大部分场景下前三层就足够了，它们完全不需要额外的 API 调用，只是「搬运」和「裁剪」数据。只有在极端情况下，才需要触发昂贵的全量摘要。

这种设计的另一个好处是**各层相互协调**。第 2 层 Snip 会告诉第 5 层「我已经释放了多少 Token」，避免重复压缩。第 4 层 Context Collapse 在第 5 层之前运行，如果它够用了，第 5 层就不触发。每一层都在为下一层「减负」。

### 写在最后

说真的，啃完这 51 万行源码，我整个人是有点懵的。

Claude Code 源码里有太多优秀的 Agent 技术落地方案。

比如压缩上下文要分五步走、记忆要分四种类型存、System Prompt 设计……

每一件事单拿出来都算不上什么黑科技，但全串在一起，就是一套能把一匹野马驯成耕牛的缰绳系统。

这给我一个很大的启发：做 Agent，别老盯着模型发呆。模型是发动机，但一辆车能不能安全上路，靠的是刹车、方向盘、安全带。这些「不起眼」的东西，才是真正决定成败的。

最后调侃一句，Claude Code 源码泄漏这波操作，无疑将极大地缩短国内外 AI Agent 的信息差，全面利好国产 Agent 的爆发式发展！

好了，今天就聊到这。如果觉得有收获，别忘了点赞转发，这对小林真的很重要！

## Claude Code 主循环 Query 图解：一轮对话是怎么跑起来的？

> Source: https://xiaolinnote.com/claudecode/source/cc_query_loop.html

大家好，我是小林。

最近有位林友跟我说，他去面试，被问到一道很硬核的题目：「介绍一下 Claude Code 主循环 Query 的流程？」

这题面试官想考的，其实不是 Claude Code 本身，而是看你对一个工业级 Agent 的循环架构有没有真正想过。

你是停在「调模型、跑工具」这种皮毛上，还是真的往下挖过：模型吐字的时候怎么控？出错了怎么兜？状态怎么跨轮不丢？这才是面试官想听你聊的东西。

很可惜，这位林友当时只答到「就是一个 while 循环，调模型、跑工具、再调模型」就接不下去了，面试官摇了摇头让他先回去等通知。

今天这篇文章，我就把 **Claude Code 源码里这个主循环（叫 queryLoop）的设计**，由浅入深给大家扒一遍。

整个主循环的实现核心在一个 1729 行的文件里，我会带着大家解答下面这些疑问，比如：

- 敲完回车，Claude Code 内部到底发生了什么？
- 从 ask 到 queryLoop：一句话怎么穿过四层调用？
- Agent 怎么知道「该用工具」还是「该停下来」？
- 模型还在吐字、工具就开跑了？这怎么做到的？
- 工具跑挂之后，对话凭什么还能继续？
- 模型输出被截断了，怎么让用户根本感觉不到？

文章依旧硬核到底，发车！

### 一、敲完回车，Claude Code 内部到底发生了什么？

我们先从一个最朴素的场景开始。

你在终端打开 Claude Code，敲一句「帮我修复这个 bug」，然后回车。30 秒后，Claude 给你返回了修复方案，并且把代码也改好了。

这 30 秒里，到底发生了什么？

很多没接触过 Agent 开发的同学，第一反应是：「这有啥，就是把你的问题发给大模型，大模型回个答案嘛。」

如果真这么简单，那 Claude 怎么知道你说的「这个 bug」具体在哪个文件？它又是怎么把代码改到磁盘上的？

<figure>

<figcaption>多轮循环示意图</figcaption>
</figure>

真实情况是这样的：你这句话发出去之后，大模型第一次回复你的可能不是答案，而是「我需要先看一下你提到的那个文件」。

Claude Code 听到这个，就去读那个文件，把文件内容塞回去再问一遍。大模型这时候可能又说「我得跑一下测试看看现在的报错」，于是 Claude Code 又去跑测试，把报错塞回去……

来来回回好几轮，模型才会说「我知道了，这个 bug 是因为 X，我已经把修复写好了」。

这时候这一轮对话才真正结束。

**这个不停「调模型 → 看结果 → 决定下一步」的循环，就叫主循环（Query Loop），是整个 Agent 的「心脏」。**

<figure>

<figcaption>Agent 心脏示意图</figcaption>
</figure>

理解到这一层还不够。主循环这个东西，你们听上去会觉得很简单：

```
while (true) {
  调模型()
  if (模型说要用工具) {
    执行工具()
  } else {
    break
  }
}
```

如果真这么写，能跑通一个 demo，但在真实生产环境里会被千锤百炼出无数个 bug：

- 用户中途按 Ctrl+C 怎么办？
- 工具跑到一半挂了，下一轮调模型会被 API 直接拒收，怎么救？
- 模型输出超长被截断了，怎么续？
- 上下文太长塞不进去了，要不要压缩？什么时候压？
- 跑了 50 轮还没收敛，要不要强行止损？

Claude Code 的 1729 行主循环代码，80% 都在处理这些「异常路径」，真正的「正常路径」可能 200 行就写完了。这也是我说它是一份工业级教科书的原因。

<figure>

<figcaption>主循环代码占比柱状图</figcaption>
</figure>

接下来我们一层一层剥开。

### 二、从 ask 到 queryLoop：一句话怎么穿过四层调用？

要讲清楚主循环，得先看你在终端敲的那句话，在代码里到底走了几道关。

为什么要分层？我打个生活类比。

你家水龙头流出来的水，看上去就是「打开龙头水就来」，背后其实经过了四道关：水厂净化 → 街道主管线 → 小区水箱加压 → 你家管道末端。每一道关都有自己的职责，不能直接把水厂接到你家水龙头，那样既不安全也不灵活。

Claude Code 的主循环也是这样分层的：最外面一层管「你怎么用它」（就像水龙头），往里一层管「这次对话怎么记账」（就像小区水箱缓存水量），再往里一层管「怎么把事件源源不断吐给外面」（就像主管线持续供水），最里面那一层才是「真正干活的循环本体」（就像水厂在 24 小时净化）。

这四层从外到内分别叫 `ask`、`QueryEngine.submitMessage`、`query`、`queryLoop`。

这四层在源码里的调用关系，简化成伪代码长这样：

```
// 第一层 ask：SDK 入口，一次性调用
async function* ask(params) {
  const engine = new QueryEngine(config)
  yield* engine.submitMessage(params.prompt)
}

// 第二层 QueryEngine：管这次对话的会话状态
class QueryEngine {
  async *submitMessage(prompt) {
    // 处理 /斜杠命令、组装系统提示、注入上下文 ……
    yield* query({ messages, systemPrompt, tools, ... })
  }
}

// 第三层 query：流式包装层
async function* query(params) {
  yield* queryLoop(params)
}

// 第四层 queryLoop：核心循环本体
async function* queryLoop(params) {
  while (true) {
    // 准备消息 → 调模型 → 判断 → 执行工具 → 塞回结果
  }
}
```

注意每一层都是 `yield*` 接力往下传，意味着最里面 `queryLoop` 抛出的每一个事件，都能原封不动一路冒到最外面 `ask` 的调用方手里。这是「边干边吐」这条流水线能贯通的关键。

<figure>

<figcaption>四层调用链水流类比图</figcaption>
</figure>

我们从外往里看每一层。

**最外面那层叫 `ask`**，是给 SDK 一次性调用的便捷入口。你写个脚本想跑一次 Claude，最少代码量就是 `await ask({ prompt, ... })`。它内部其实只做一件事：新建一个 `QueryEngine` 实例，然后转交给它。

**第二层 `QueryEngine`** 才是真正管「会话」的家伙。它维护着这次对话的消息历史、文件缓存、权限拒绝记录这些跨轮持久化的状态。你可以把它想象成一个对话窗口的「记账本」，每说一句话、每读一个文件，都得在它这里登记。

**第三层 `query`** 长得很特殊，它是个「会一边干活一边吐结果」的函数。普通函数是干完所有事情、最后一次性 `return` 一个结果。但 `query` 不一样，它会在干活的过程中，每完成一件事就立刻把结果抛出去给外面用。这种「边干边吐」的函数，在 JavaScript 里有个专门的名字，叫**异步生成器**（async generator）。

**最内层 `queryLoop`** 才是 `while (true)` 循环本体，是真正干活的「心脏」。

讲到这里你可能会问：为啥要分这么多层？尤其是 `query` 这一层，它就只是个简单的包装，为啥不直接把 `queryLoop` 暴露出去？

这就要讲到主循环最关键的一个设计选择：**为什么必须用异步生成器？**

我们用一个生活类比来理解。

<figure>

<figcaption>水龙头 vs 水桶类比图</figcaption>
</figure>

普通的 `async` 函数，就像让你接一桶水。你打开水龙头之后，必须等水接满了，整桶端走，才能用。对应到 Agent 上，就是「你问一句，等几十秒模型把完整答案吐完，你才能看到结果」。

异步生成器就是水龙头本身。模型每吐出一个字符、每跑完一个工具，立刻就能流出来给你看。

这就是为什么你在 Claude Code 里敲完回车，能立刻看到模型一个字一个字地在打字，而不是干等几十秒突然蹦出一大段文字。这种「边干边吐」的能力，不仅是用户体感，更是后面第五章会讲到的「流式 + 并行工具执行」的前提条件。

<figure>

<figcaption>事件接力流水线示意图</figcaption>
</figure>

异步生成器的语法长什么样？我们看一眼源码（来自 `query.ts`），里面有两个关键标记：

```
export async function* query(params) {
  const terminal = yield* queryLoop(params, ...)
  return terminal
}
```

第一个是函数前面那个 `*`（写法是 `async function*`），它告诉 JS 这是一个异步生成器，不是普通函数。第二个是 `yield*` 这个表达式，它的意思是「把 `queryLoop` 里所有抛出来的东西，原封不动地接力往外抛」，专门用来串联两个生成器。

这层包装看上去什么都没做，但它的作用是把对话生命周期的收尾工作（比如最后还要通知一些清理事件），和核心循环逻辑解耦开，让 `queryLoop` 只管「转圈」，不操心「转完之后还要给谁打招呼」。

<figure>

<figcaption>四层职责对照表</figcaption>
</figure>

四层调用链解释完了，我们终于可以走进最里面那间屋子，看看主循环转一圈到底干了啥。

### 三、主循环转一圈，到底跑了哪些动作？

`queryLoop` 这个函数有近 1500 行，但骨架就是一个 `while (true)`。每一轮迭代，按顺序做五件事。

<figure>

<figcaption>主循环五步流程图</figcaption>
</figure>

这张图就是整个主循环的「素描」。如果你更喜欢从代码视角看，把这个流程翻译成伪代码就是：

```
async function* queryLoop(params) {
  let state = { messages: [...], turnCount: 1, ... }

  while (true) {
    // 1. 准备消息：必要时压缩
    const messagesForQuery = maybeCompact(state.messages)

    // 2. 流式调大模型，边收边处理
    let toolUseBlocks = []
    let needsFollowUp = false
    for await (const chunk of callModel(messagesForQuery)) {
      yield chunk  // 文字块即时抛给用户
      if (chunk 是 tool_use 块) {
        toolUseBlocks.push(chunk)
        needsFollowUp = true
      }
    }

    // 3. 判断继续还是结束
    if (!needsFollowUp) {
      return { reason: 'completed' }
    }

    // 4. 执行所有 tool_use（可并行/串行）
    const toolResults = await runTools(toolUseBlocks)

    // 5. 塞回结果进下一轮
    state = {
      ...state,
      messages: [...messagesForQuery, ...assistantMessages, ...toolResults],
      turnCount: state.turnCount + 1,
    }
    // continue 回到 while 头部
  }
}
```

伪代码里这个 `state` 对象就是第六章要展开讲的「跨轮状态对象」，这里先记个脸熟。

下面我把每一步拆开讲。

**第一步：准备消息。**

每一轮开始，主循环要拿到「截至目前的完整对话历史」，准备喂给模型。这里有个细节很容易被想当然：消息历史不是每轮开头都主动压一遍的，而是**被动触发**。主循环先按原样把消息发出去，**只有 API 返回 `prompt_too_long`（413）拒收**了，才补救压缩一次再重试这一轮。如果压完还是塞不下，就老老实实退出。

为什么不主动压？因为压缩本身也要烧 token 调一次小模型做摘要，能不压就不压。主循环的策略很朴素：先试试看，被拒了再补救。

`State.hasAttemptedReactiveCompact` 这个标志位的作用，是**同一轮内防止反复压**。一次压完还是塞不下，就别再压第二次了，避免陷入「压缩 → 还是塞不下 → 再压」的死循环。新一轮开始时这个标志会被重置。

Claude Code 的压缩策略我之前专门写过一篇（公众号文章《<a href="https://mp.weixin.qq.com/s/NBR6dRr3iO7KCTEChk8QUg" target="_blank" rel="noopener noreferrer">Claude Code 上下文压缩机制详解</a>》），这里就不重复展开了。

<figure>

<figcaption>消息历史滚雪球示意图</figcaption>
</figure>

**第二步：流式调大模型。**

讲这一步之前，我得先帮大家把一个后面会反复出现的概念铺一下，否则后面看代码会很懵。

**模型回话的两种姿势**

大模型每次回复你的内容，并不是只有一种格式。仔细看 Claude API 的返回，你会发现模型每次回复，里面可能塞着两种东西：

- 第一种是「文字块」，就是普通的文字内容，比如「这个 bug 是因为变量没初始化」。这是模型在跟你说话。

- 第二种是「工具调用块」，长得像 `{ 工具名: "Read", 参数: { 文件路径: "config.ts" } }` 这种结构化的请求。这是模型在告诉你「我现在想用某某工具，参数是这些，麻烦你帮我跑一下」。

这种「工具调用块」在 Anthropic API 协议里有个专门的名字，叫 `tool_use` 块（直译就是「使用工具」的意思）。后面的文字里我会经常用这个词，你只要记住「模型说要用工具的请求，就叫一个 `tool_use` 块」就够了。

<figure>

<figcaption>文字块 vs tool_use 块对比图</figcaption>
</figure>

铺垫完毕，回到第二步。

调模型这一步是用流式接收的，模型每吐出一小段内容（可能是几个字符），主循环立刻就能拿到。收到的可能是文字块的一部分，也可能是 `tool_use` 块的一部分。

主循环一边收，一边干两件事：把文字块往上抛出去（让用户能即时看到模型在打字），把 `tool_use` 块攒起来准备等会儿执行。

**第三步：判断继续还是结束。**

模型流完之后，主循环要回答一个最核心的问题：**这一轮对话结束了，还是要继续？**

这是整个主循环的「决策点」，我下一章专门讲。

**第四步：执行所有 tool_use。**

如果判断要继续，就把模型这一轮要求的所有工具调用都跑掉，拿到每个工具的执行结果。Claude Code 在这一步有一个非常骚的「流式并行」设计，第五章细讲。

**第五步：塞回结果，进下一轮。**

所有工具结果跑完了，把它们当作「user 角色的消息」塞回消息历史里，更新一下「跨轮要带的状态对象」（这个状态对象第六章会专门讲），然后 `continue` 回到 `while` 头部。

<figure>

<figcaption>五步骨架环形图</figcaption>
</figure>

注意一个细节：每跑一轮，消息历史都会变长（assistant 的回复 + 工具结果）。如果主循环跑了 20 轮，那第 20 轮发给模型的消息历史里，就有前 19 轮所有的对话和工具结果。

这也是为什么压缩机制要存在：消息历史会像滚雪球一样越滚越大，不压就会爆炸。

<figure>

<figcaption>一轮迭代消息累积示意图</figcaption>
</figure>

主循环的骨架就这样。但「骨架」不等于「细节」，真正考验工程功力的是后面三章要讲的东西。

### 四、Agent 怎么知道「该用工具」还是「该停下来」？

承接上一章的「决策点」，主循环第三步那个判断到底怎么做的？

我跟你说，这个判断简单到出乎你的意料：**就看模型返回里有没有 `tool_use` 块**。

主循环内部用一个布尔变量记录「这一轮要不要继续」，这个变量在源码里叫 `needsFollowUp`（直译就是「需要后续动作」）。判断逻辑就一条：

- 有 `tool_use` 块 → 把 `needsFollowUp` 标记为 true → 执行工具 → 继续循环
- 没有 `tool_use` 块 → 模型在跟你说话 → 结束循环

退出的时候，主循环还会附带一个「退出原因码」（在源码里叫 `reason`），告诉外面是因为啥结束的。最常见的退出原因叫 `completed`，意思就是「正常完成」。代码大致长这样：

```
if (!needsFollowUp) {
  // 模型没要求调工具，对话正常结束
  return { reason: 'completed' }
}
// 有 tool_use，跑完工具后 continue 回循环头
```

就这么个判断，没有什么复杂的状态机，没有什么意图识别，没有什么投票表决。

为啥这么简单？因为「决定下一步做什么」这件事，本身就是大模型的工作。你只要看它最后吐出来的是「文字」还是「工具调用请求」，就知道它的意图了。

我打个比方。你跟工位旁边的同事说：「帮我看下这个 bug。」他要么回你：「我得先看看你那段代码。」（动手 → 用工具），要么直接告诉你：「这是因为 X 导致的，你这样改一下就好了。」（停下 → 给答复）。

Claude 主循环的判断逻辑就是这么朴素。

<figure>

<figcaption>tool_use 决策类比图</figcaption>
</figure>

**但是！「停下来」并不只有「正常完成」这一种姿势。**

这才是真正的工程深水区。

在 Claude Code 的主循环里，退出原因有十多种，每一种背后都对应着一个曾经踩过的坑：

- **正常完成**（`completed`）：模型说「搞定了」，对话结束。这是最幸福的一种
- **轮数封顶**（`max_turns`）：调用方传了 `maxTurns` 参数时生效，跑到上限就强行止损，防止模型跑飞烧钱（不传则不封顶，由其他退出条件兜底）
- **流式调模型时被中断**（`aborted_streaming`）：模型还在吐字，用户按了 Ctrl+C
- **工具执行时被中断**（`aborted_tools`）：工具跑到一半，用户按了 Ctrl+C
- **输入超长救不回来**（`prompt_too_long`）：消息历史太长，连压缩都塞不进上下文窗口
- **输出超长救不回来**（`max_output_tokens_recovery`）：模型输出被截断，尝试 3 次续写后还是不行
- **被用户钩子拦下**（`stop_hook_prevented`）：用户配置了自定义钩子主动喊停（比如规定 git push 前必须先跑 lint，lint 没过就拦下，这个钩子机制叫 Stop Hook）
- **图片格式错误**（`image_error`）：传图的时候格式不对

这里列的是最有代表性的 8 个。源码里实际定义了 17 种 `reason`，剩下的多是细分子情况，原理一致，不影响理解。

看到这十几种 reason，你应该能感受到一件事：**一个工业级的 Agent 主循环，它要兜住的从来不是「正常退出」，而是十几种「异常退出」。**

我们用一张图把这些退出 reason 按性质归一下类：

<figure>

<figcaption>主循环退出原因分类图</figcaption>
</figure>

这张图能让你直观感受到：**正常完成只是分支里最小的一支，其他三支才是大头**。

每一种异常退出，都不是一个简单的 `throw new Error`，而是需要：

- 第一，识别这种错误状态（每种错误的特征不一样）；
- 第二，做必要的清理（比如取消正在跑的工具、解锁资源）；
- 第三，返回结构化的 `reason`，让外层调用者知道是什么原因结束的，方便上报、重试或者给用户友好提示。

<figure>

<figcaption>主循环工程量冰山图</figcaption>
</figure>

这才是「主循环」三个字背后真正的工程量。

### 五、模型还在吐字、工具就开跑了？这怎么做到的？

讲完决策，我们看主循环第四步：执行工具。

这一步是 Claude Code 整个主循环里最骚的设计，我必须单独开一章来讲。

我们先想想朴素做法是什么样的。

<figure>

<figcaption>朴素串行 vs 流式并行时间线</figcaption>
</figure>

朴素做法是这样的：

- 第一步，等模型完整流完，把所有 `tool_use` 块都收齐。
- 第二步，挨个执行这些工具，等所有工具跑完。
- 第三步，把结果塞回去，进下一轮。

这个流程没毛病，能跑通。但仔细想想，**模型在「吐字」期间，工具在干等。工具开始执行了，模型又在干等。两段时间完全串行，CPU 和网络都在浪费**。

Claude Code 的做法是这样：

**模型一边流式输出，主循环就一边监听。一旦识别出一个完整的 `tool_use` 块，立刻把它丢给后台开始执行，不等模型流完。等模型流完，最早开始执行的那个工具，可能已经回结果了。**

这个负责后台执行的对象叫 `StreamingToolExecutor`（直译就是「流式工具执行器」）。它的核心就两个动作：「边收边开跑」和「最后一次性收结果」。

我把源码精简到最小，就两行：

```
// 模型流式输出时，每识别到一个工具调用就立刻丢到后台开跑
streamingToolExecutor.addTool(toolBlock, message)

// 模型流完后，把所有已完成的工具结果一次性收回来
const toolUpdates = streamingToolExecutor.getRemainingResults()
```

第一行 `addTool`，发生在主循环监听到模型吐出一个 `tool_use` 块的瞬间。这个工具立刻在后台开跑，主循环不等它，继续监听模型的下一段输出。

第二行 `getRemainingResults`，发生在模型完整流完之后。这时候所有工具都已经在后台跑了一会儿了，这里只是把已完成的结果一次性收上来。

<figure>

<figcaption>点菜类比示意图</figcaption>
</figure>

用「点菜」打个比方就清楚了。

传统模式是：你完整说完「我要一个红烧肉、一个清炒空心菜、一个鱼香肉丝」，服务员拿着单子跑去厨房，厨房才开始一道一道做。

Claude Code 的模式是：你说「我要一个红烧肉」，服务员立刻冲到后厨喊「红烧肉一份！」，你接着说「再来一个清炒空心菜」，服务员又喊「空心菜一份！」。等你点完，红烧肉已经在锅里炒了一半了。

最终上菜时间，比传统模式快了一大截。

<figure>

<figcaption>最终用时对比柱状图</figcaption>
</figure>

**但并不是所有工具都能这么并行**。这里有个边界要讲清楚。

Claude Code 的工具在定义的时候，每个工具都有一个属性，标明它是「只读」还是「会改东西的」。

- 只读工具（比如 Read、Grep、Glob）：可以随便并行。你同时读 10 个文件，互不干扰
- 会改状态的工具（比如 Edit、Write、Bash）：必须串行。两个写文件操作如果并发跑，可能互相覆盖

这个属性是在工具定义时就声明的。

如果开发者一时忘了写，框架会 **fail-closed 兜底**，默认当成「会改东西的工具」（值为 `false`），强制串行执行。

也就是说，宁可错杀让你慢一点，也绝不会因为漏写而出现并发踩踏。

<figure>

<figcaption>只读 vs 写操作并行性对照图</figcaption>
</figure>

讲到这里，我想抛出一个延伸思考：**Claude Code 让 Agent 跑得快的秘密，不在于模型多聪明，而在于工程上把每一段空闲时间都榨干**。

这种「流式 + 并行」的设计，是把模型生成时间和工具执行时间高度重叠，最大化利用每一秒。

这才是真正的工程功力。

### 六、出错的时候，主循环凭什么让你毫无察觉？

前面五章把主循环的正常路径讲完了。但回到开头那个问题，你按下回车后那 30 秒里，主循环转的可不只是「调模型 + 跑工具」这么干净。

中间随便一个意外：网络抖一下、你按一次 Ctrl+C、模型一口气想说太多话，都可能让对话卡死、报错或者结果残缺。

Claude Code 之所以让你**感觉不到这些意外**，是因为主循环里藏了一堆兜底机制。这一章就挑三个最关键的细节讲讲：你看不见的部分，主循环都帮你处理了什么。

#### 6.1 跨轮状态对象：为什么不能用「全局变量」糊弄？

我们先想一个朴素问题：主循环是个 `while (true)`，那「这一轮发生了什么」要怎么传给下一轮？

最朴素的做法是用闭包变量或者全局变量，比如「重试次数」就放一个外部变量，每轮 +1。能跑，但调试起来是噩梦：你看到代码继续循环了，根本不知道是因为哪个标志位被改了。

Claude Code 的做法是把所有跨轮要传的字段，**全部打包到一个状态对象里**，每一轮迭代开头读出来，结尾构造一个新的写回去。这个状态对象在源码里叫 `State`。

`State` 的关键字段长这样（我精简了一下，只保留最有代表性的几个）：

```
type State = {
  messages: Message[]                    // 累积的对话消息历史
  turnCount: number                      // 当前是第几轮
  maxOutputTokensRecoveryCount: number   // 输出截断已经恢复了几次
  hasAttemptedReactiveCompact: boolean   // 本轮是不是已经触发过压缩了
}
```

<figure>

<figcaption>状态对象跨轮传递示意图</figcaption>
</figure>

注意看后面两个字段，一个是「已经恢复了几次」，一个是「本轮是不是已经压缩过了」。它们的本质都是「计数器」或「防重复标志」（同一轮内只让某件事发生一次），目的就一个：**避免无限循环**。

举个例子：模型输出超长被截断了，主循环尝试恢复一下。如果恢复又失败，再来一次。但你不能无限重试，否则就死循环了。所以要计数：**最多 3 次，超过就老老实实退出**。

再比如：上下文塞不下了，主循环触发一次压缩。但同一轮内只能压一次，否则连续压缩没意义还浪费 token。所以要标志位：本轮已经压过了，就别再压了。

这些状态如果全藏在闭包变量里，调试的时候你根本不知道循环是因为什么继续的。摊在一个状态对象里，每一轮的「现场」都一目了然。

<figure>

<figcaption>闭包变量 vs 状态对象对照图</figcaption>
</figure>

这就是为什么你用 Claude Code 跑几十轮工具也不会陷进死循环，所有「还要不要继续」的标志都摊在明处，每一轮的「现场」都能干净结束，不会因为某个藏起来的变量没复位就鬼打墙。

#### 6.2 工具跑挂之后，对话凭什么还能继续？

这是一个 Anthropic API 的「冷知识」，没踩过坑的人根本想不到。

我先把这条 API 协议规矩说清楚：

第三章我们讲过，模型回话时会吐出 `tool_use` 块（工具调用请求）。工具跑完之后，要把执行结果以另一种格式塞回消息历史，这种「工具执行结果」叫 `tool_result` 块。

**API 协议有一条死规定：每个 `tool_use` 块，必须有一个对应的 `tool_result` 块跟它配对。如果消息历史里有个孤零零的 `tool_use` 没人接，下一轮请求直接被 API 拒收，报错说「工具调用 ID 对不上」。**

正常情况下没问题，工具跑完就有结果，配对很自然。但出错的时候呢？

比如：模型刚吐完 `tool_use` 块，网络突然断了，工具根本没开始跑；或者工具跑到一半，用户按了 Ctrl+C；或者模型出了问题，主循环临时切换到备用模型（这种切换机制叫「模型降级」），前一个模型吐出的 `tool_use` 就没机会执行了。

这些情况下，消息历史里就有了「孤立的 tool_use 块」。你不处理，下一轮想再调模型，API 直接拒收，会返回一个「请求格式不对」的错误。

<figure>

<figcaption>孤立 tool_use 块三种场景图</figcaption>
</figure>

Claude Code 的方案听起来有点奇葩：**合成一个假的工具结果塞回去，内容写「这个工具因为 XX 错误没执行」**。主循环里专门有段代码干这事，函数名也很直白，直译过来就是「把缺失的工具结果块补上」（源码里叫 `yieldMissingToolResultBlocks`）。

这段补救代码就三个关键：标明类型是 `tool_result`、内容是错误消息、附上对应的工具调用 ID 让 API 能配对上。简化后长这样：

```
yield 一条用户角色消息({
  类型: 'tool_result',           // 标明这是个工具结果
  内容: '这个工具没执行成功',     // 解释为啥失败
  is_error: true,                // 打个错误标记
  tool_use_id: 对应的工具调用ID, // 让 API 能配对上
})
```

<figure>

<figcaption>孤立 tool_use 补救方案图</figcaption>
</figure>

这种设计可以叫「先认错，再继续」。API 协议是死的，但通过合成一个明确标了「这是错误」的假结果，主循环至少不会因为一个错误就彻底卡死，还能让模型知道「上个工具失败了」，从而做出合理的下一步决策。

这就是为什么你用 Claude Code 按 Ctrl+C 打断之后，下一句话依然能接得上，主循环替你把那个「没人接的工具调用」伪造了一个错误结果，让 API 协议这个死规矩不会因为一次中断就把整个对话废掉。

#### 6.3 模型输出被截断了，怎么让用户根本感觉不到？

最后一个细节，关于「模型输出超长被截断」这个常见问题。

模型每一次返回都有个最大 token 数限制。Claude Code 常规默认是 32k，但在某些灰度场景下（背后是个可以开关的「特性开关」控制位）会被压到 8k 以控费。一旦模型一口气想吐 12k 的内容，到 8k 就被强行截断，剩下的 4k 没出来。这种状态在 API 协议里叫「输出 token 上限错误」。

换作一个简单点的实现可能就直接弹个错误框：「输出过长了，请重试」。或者更糟，干脆忽略截断，让你看到一段戛然而止的回答。

Claude Code 的处理是两段式的，非常有意思。

**第一段：静默升档。**

模型第一次触发 8k 截断时，主循环不报错，悄悄把上限调到 64k 然后重试这一轮。用户完全感知不到出过错，只是觉得「这次模型答得稍微慢了点」。

<figure>

<figcaption>8k 到 64k 静默升档示意图</figcaption>
</figure>

**第二段：在下一轮消息里轻轻推一把，让模型自己续写。**

如果调到 64k 还截断，那是真的输出太长了。

这时候主循环会在下一轮的消息里，塞一段话给模型：「上次输出超长被截断了，请直接从断点续写，不要道歉、不要回顾你刚才在干什么、把剩余工作拆成更小的块」（英文原文是 `Output token limit hit. Resume directly — no apology, no recap...`，后面还会指示模型「从断点续写、把剩余工作拆小」）。

让模型从断点继续往下写。

这种「在消息里轻推一把提示模型续写」的动作，英文里叫 nudge（轻推）。但 nudge 不能无限循环，会陷入死锁。

所以源码里写死了 3 次上限：续写 3 次还不行，主循环就老老实实退出，返回原因「输出超长救不回来」。

<figure>

<figcaption>max_output_tokens 两段式恢复流程图</figcaption>
</figure>

我想在这里抛出一个延伸思考：**工业级的容错，不是「报错弹窗」，而是用户根本感觉不到出过错**。

你用 Claude Code 这么多天，可能从来没注意过模型输出被截断这件事，因为它在背后默默地帮你处理了。这就是好工程的样子。

### 七、写在最后：主循环背后藏着哪些设计哲学？

从整篇文章扒下来，我觉得 Claude Code 主循环背后有四个设计哲学，值得反复琢磨：

**一、边干边吐，让感知前置。** 用异步生成器把每个中间事件即时抛出去，用户体感「响应快」就是这么来的。

**二、状态显式管理。** 该计数的字段全部摊在一个状态对象里，不藏在闭包变量里。能调试的状态才是可靠的状态。

**三、引擎层不掺业务。** 主循环不知道工具具体在干嘛，新增能力零侵入，只需要往工具池里塞一个新工具就完事。

**四、错误恢复优先于优雅。** 每个失败路径都有兜底。宁可丑陋地合成一个假的工具结果塞回去，也不能优雅地崩溃。

<figure>

<figcaption>面试场景呼应图</figcaption>
</figure>

回到开篇那个面试场景。

再被问到「Claude Code 主循环 Query 是怎么跑的？」.

你能从「为什么主循环不能简单写成 while true」一路讲到「工具跑挂之后怎么让对话不卡死」「模型输出截断了怎么悄悄帮用户续上」，再补一句「主循环用流式监听 + 后台并行，把模型生成时间和工具执行时间叠在一起」，面试官就知道你不是在背概念，而是真的拆过源码、想过工程问题。

这才是这道面试题真正想考的东西。

如果你觉得这篇文章对你有启发，欢迎点个「在看」和「赞」，这是对小林最大的肯定和帮助。

我们下一篇见啦！

## Claude Code 上下文管理图解：Compact 压缩机制怎么实现？

> Source: https://xiaolinnote.com/claudecode/source/cc_compact.html

大家好，我是小林。

最近有位林友跟我说，他去面试某大厂 AI 岗，被面试官追问了不少 Claude Code 源码相关的问题。

<figure>

<figcaption>img</figcaption>
</figure>

还好他面试前刚好啃过我公众号那两篇 Claude Code 源码分析的文章，几乎都答上来了，顺利拿下了 offer。

他说的那两篇分别是：

- <a href="https://mp.weixin.qq.com/s?__biz=MzUxODAzNDg4NQ==&amp;mid=2247556000&amp;idx=1&amp;sn=01e3a55e22467c677af3e75f9a6d7c62&amp;scene=21#wechat_redirect" target="_blank" rel="noopener noreferrer">万字长文图解 Claude Code 源码：Agent工作模式、记忆机制、上下文窗口管理机制等</a>
- <a href="https://mp.weixin.qq.com/s?__biz=MzUxODAzNDg4NQ==&amp;mid=2247557309&amp;idx=1&amp;sn=db872d9df4336797d2c364b5c4e4e880&amp;scene=21#wechat_redirect" target="_blank" rel="noopener noreferrer">万字长文图解 Claude Code 源码：多Agent实现机制</a>

基本把 Claude Code 比较核心的原理都剖析了一波，也收获了很多读者的好评。

<figure>

</figure>

然后又有读者跟我反馈，想让我再深入聊聊 「**Claude Code 的上下文窗口是怎么管理的？**」

原因很简单，因为他面试被深入问到了，这可以说是 agent 开发最常考察的知识点了。

只要简历上做过 agent 项目」，面试官几乎都会追问一句：**你这个 agent 项目的上下文是怎么管的？**

那 Claude Code 这个被业内认证「最强 agent 范本」之一的产品，到底是怎么做的？

我把 Claude Code 的 compact（压缩） 相关的源码翻了好几遍，越看越觉得有意思。别急，这篇文章我会一层一层拆给你看。

读完之后，你不光能答出这道面试题，**说不定还能把这套思路借鉴到你自己的 agent 项目里**。

<figure>

</figure>

我会按这几个层次展开：

- 什么是上下文窗口？为什么 agent 跑起来分分钟爆窗口？
- 业界常见的上下文方案，为什么不够看？
- Claude Code 的 5 层压缩金字塔
- Auto-Compact 的整体思路
- 什么时候触发压缩？
- 压什么、留什么、丢什么？
- 摘要 prompt 是怎么设计的？
- 压完之后怎么接续对话？
- 这道面试题该怎么答？

如果你能看到最后，下次面试官再问到这道题，至少能让对方眼前一亮。

------------------------------------------------------------------------

### 一、先聊聊上下文窗口

聊 Claude Code 怎么管之前，我们先退一步，把「上下文窗口」这个概念本身搞清楚。如果你已经很熟了，可以跳过这一节直接看第二节。

#### 上下文窗口是个啥？

大模型跟人不一样，它没有真正的「记忆」。每次你问它一个问题，本质上都是「从头看一遍」：把 system prompt、所有历史对话、当前问题，一股脑塞进去，然后生成回复。

塞进去的这堆东西，加起来的总长度有个上限，这个上限就叫「**上下文窗口**」。单位是 token，你可以粗略理解成「字」，但中英文有差别（中文一个字往往占 1 到 2 个 token，英文一个单词平均算 1 个 token）。

打个比方，上下文窗口就像是模型的「工作台面」。你能放多少东西在这张桌子上，模型就能看见多少东西。桌子的大小是固定的，超出去的东西，要么放不下，要么得挤掉一些已有的。

那这张桌子现在有多大？看模型。当前主流模型的窗口大小大概是这样：

- GPT-4 早期版本：8k token（约 1 万多字）
- Claude 3.5 Sonnet：200k token（约 30 万字）
- Claude Opus 4.7 的 1M 版本：1M token（约 200 万字）
- 部分 Gemini 模型甚至能到 2M

200 万字听起来很大对吧？拿来塞一整套《三体》三部曲都绰绰有余。但你真去做 agent，会发现这张桌子还是不够大。

<figure>

<figcaption>模型工作台示意图</figcaption>
</figure>

#### 为什么 agent 更费窗口？

普通聊天为啥不太担心窗口？因为聊天就是「你一句我一句」，每轮可能也就几十到几百个 token。一个 200k 的窗口能撑上千轮对话。

但 agent 就不一样了，窗口压力来自三处叠加。

**第一处，开局就是大头**。一个 agent 开局，光是 system prompt + 工具描述 + <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 这些「固定开支」，就要塞进 5k 到 10k token。你还没开始干活，桌子就已经占了一角。

**第二处，工具调用会在上下文里留下两条记录，而且持续累积**。这一点很多人不知道，是上下文爆得快的元凶之一。普通聊天里模型回一句话，上下文就多一条文本。但工具调用不一样：模型先发一条 tool_use（告诉系统我要调什么工具、参数是啥），工具跑完再回一条 tool_result（把结果塞回上下文），这两条都会留下来。

举个例子，agent 要读 `a.py` 这个文件，对话里会变成这样：

- 第一条 `tool_use: Read("a.py")`，相当于模型说「我要调 Read 工具读 <a href="http://a.py" target="_blank" rel="noopener noreferrer">a.py</a>」
- 第二条 `tool_result: <a.py 的完整内容>`，是系统把文件读完、把内容返回给模型

真正吃窗口的不是「两条 vs 一条」这个数量差，而是 tool_result 里经常塞着几千 token 的文件内容或命令输出，并且会一直留在后续每一轮里被反复加载。调一次工具只是一次开销，但它留下的内容会在接下来每一轮被重新计费一次。

**第三处，大文件 Read 杀伤力巨大**。agent 经常要读源代码文件，一个稍微大点的源文件几千 token 起步，一万 token 也很常见。读三五个文件，桌子上就堆满了。

算笔账你就有概念了：一个稍微复杂点的编码 agent，开局 8k 加上读 5 个平均 5k 的文件等于 33k，再加上对话和中间的 grep、edit 工具调用，跑个十几轮，几万 token 没了。如果是那种持续好几个小时的长任务，分分钟逼近 200k 的天花板。

#### 加大窗口能解决吗？

你可能会说，那模型厂商不是在卷长上下文吗？卷到 1M、2M，问题不就自然解决了？

这个想法有道理，但治标不治本，三个硬伤摆在那儿。

**第一个硬伤是钱**。上下文越长，单次推理的 token 消耗就越大，账单也跟着膨胀。一个长跑的 agent，如果不做上下文管理，token 消耗会按几何级数往上涨。

**第二个硬伤是慢**。Attention 机制的计算复杂度跟序列长度是平方相关的，上下文越长，模型生成第一个 token 的延迟（业内叫 TTFT，也就是「从你按下回车到看到第一个字」的那段等待）就越高。一个 5000 token 的对话 1 秒返回，一个 150k 的对话可能要等十几秒。

**第三个硬伤最致命，叫做 Lost in the Middle**。这是个挺有名的现象：当上下文非常长时，大模型对首尾信息记得清楚，对**中间段**则记忆模糊。你以为塞进去就是塞进去了？不一定，中间那一大段，模型可能瞟一眼就过去了。这是注意力机制的固有特性，跟窗口多大没关系，所以哪怕窗口扩到 1 亿 token，中间段的信息照样看不清楚。

<figure>

<figcaption>token 消耗结构图</figcaption>
</figure>

所以你看，光靠模型厂商扩窗口救不了 agent。要救，必须 agent 自己主动管理上下文。

**管得好的 agent 能聊上千轮、跑通复杂任务，管不好的 agent 几十轮就开始胡言乱语、忘东忘西。**

那行业里现在都怎么管？有几个常见方案，我们先看看它们都长啥样，再看看 Claude Code 凭什么能在面试中说服面试官。

### 二、常见方案为什么不够看？

聊这个之前，我先做个铺垫。你去 Github 上扒一扒开源的 agent 框架，会发现上下文管理的方案大概就那么几类，我们一个一个看。

#### 方案一：滑动窗口

这是最常见的一种。逻辑也最直白：你设个阈值，比如对话超过 50 轮、或者总 token 超过 100k，就开始砍。从最老的消息开始往后砍，保留最新的若干条。

听起来不错对吧？又简单又快。

但你想啊，agent 跟普通聊天机器人不一样。一个 agent 跑复杂任务，最关键的决策往往就在最开始。比如用户开局说了一句「我们这次的目标是 A，注意一定要避免 B」，这种全局性指令，你把它砍掉了，后面 agent 就开始干 B 那种被禁止的事，等于完全失控。

而且工具调用是有「上下文依赖」的。你前面用 Read 读了一个文件，把文件内容存进了 tool_result。后面 agent 引用了这个内容做决策。如果你把那个 tool_result 砍了，后面那段引用就变成无源之水，模型一脸懵：我之前说的是基于啥来着？

所以滑动窗口本质是「用遗忘换续航」。对话能聊下去，但 agent 的脑子被打了。

<figure>

<figcaption>滑动窗口示意图</figcaption>
</figure>

#### 方案二：每 N 轮做摘要

这是稍微进阶一点的方案。每过 10 轮、或者每过 50k token，触发一次摘要：把这一段对话扔给一个小模型，让它生成一段话总结，然后用这段总结替换原来的消息。

这个思路对不对？对。比滑动窗口好多了，至少信息没全丢。但你仔细一品，问题也不少。

第一，**触发时机太死板**。10 轮可能是个非常重要的关键节点，你这一刀切下去，模型把那 10 轮压缩成一段话，关键细节就丢了。5 轮可能根本啥都没干，你也去压一遍，反而把好好的对话切碎了。

第二，**摘要的粒度也粗**。一段话能装多少信息？对话里那些细微的状态、错误的修复过程、用户中途改的需求，全压成几句话，agent 接着干的时候很容易丢失这些细节。

所以「每 N 轮摘要」是一种「机械主义」的方案，看似在管理上下文，其实是在按节奏粗暴破坏对话连贯性。

<figure>

<figcaption>每 N 轮摘要示意图</figcaption>
</figure>

#### 方案三：向量召回历史

这个方案就更「高大上」了。逻辑是这样：把所有历史消息切成片，丢到向量数据库里。每次 agent 要回答新问题，先用问题去召回 top-k 个最相关的历史片段，然后塞进上下文。

<figure>

<figcaption>向量召回历史示意图</figcaption>
</figure>

这套思路在 RAG 系统里用得很普遍，所以很多人想当然觉得，agent 上下文也可以这么管。

但你拿到 agent 场景来用，会立刻翻车。

第一个问题，agent 的上下文是强时序依赖的。你说「先做 A 再做 B」，向量召回不管顺序，它按相似度召回，可能把 B 召回上来，A 落在了 top-k 外面。模型一看：「哦原来要做 B」，先做 B 去了，整个执行顺序就乱了。

第二个问题，**工具调用是「成对」出现的**，tool_use 和 tool_result 必须一起出现。向量切片可能把这两个切开了，留个 tool_use 在召回结果里，tool_result 没拿到，模型就疑惑：我之前调了这个工具结果呢？

第三个问题更要命：**召回 top-k 一定会漏掉东西**。agent 的关键决策点可能就藏在某一条不起眼的消息里，相似度不一定高，但它就是关键。一旦漏掉，整个对话的逻辑就断了。

所以向量召回这套，搞 RAG 检索文档行，搞 agent 上下文不行。这是两个完全不同的场景。

你看，这三种方案各有各的硬伤。面试官听完，要么觉得你「就这？」，要么觉得你「想得太简单了」。

<figure>

<figcaption>三种方案对比示意图</figcaption>
</figure>

那 Claude Code 怎么干？

它走的是完全不同的路子：**不是「保留加召回」，是「重写整段对话」**。听着是不是有点夸张？别急，我们一步一步拆。

### 三、Claude Code 的 5 层压缩金字塔

聊细节之前，必须先把全景给你，免得只见一棵树忘了整片森林。

Claude Code 的上下文管理**不是一招制敌**，而是一套从轻到重的 **5 层金字塔**。它的设计原则一句话讲清：**能不压就不压，必须压的时候从最轻的手段开始**。

5 层从底（轻）到顶（重）大概是这样。

<figure>

<figcaption>5 层压缩金字塔示意图</figcaption>
</figure>

#### 第 1 层：大结果存磁盘

agent 调一个 Read 工具读了 10MB 的日志？这种「单工具结果超 50KB」的情况会直接被拦截：完整内容写到磁盘文件，消息里只留一个 2KB 的预览。

完整内容**没丢**，模型需要时可以再次 Read 拿回来。零 API 开销。同一条消息里所有工具结果加起来还有个 200KB 总量上限，超了就挑大的存磁盘。

<figure>

<figcaption>大结果存磁盘示意图</figcaption>
</figure>

#### 第 2 层：Snip 砍掉远古消息

对话开头那几轮探索性提问可能已经完全没用了。Snip 这一层就负责把它们删掉，再插入一条「这之前的内容已被清理」的边界标记。

不过这里有个容易被一句话带过的细节得说清楚：到底删哪些远古消息，不是写死的规则，而是模型在正常回答的那一回合里，顺手用一个专门的 snip 工具，按消息 id 把没用的那几条标记出来，真正的删除动作才在本地完成。

所以它跟第 5 层 Auto-Compact 不一样。Auto-Compact 会专门「另发一次摘要请求」，Snip 没有这种独立调用，只是搭着模型本来就要跑的那一回合做的，顶多多注入一小段提示、给每条消息加个 id 标签，多花一点 token。换句话说它不是字面意义上的「零 API、纯本地」，但确实是一层很轻的开销。Snip 释放出来的 token 数还会传给第 5 层 Auto-Compact，避免两层重复压缩。

<figure>

<figcaption>Snip 砍头部示意图</figcaption>
</figure>

#### 第 3 层：Micro-Compact 时间衰减

距离上一次 API 调用超过约 60 分钟时触发。逻辑是：超过这个时间，大模型 API 端的 Prompt Cache 大概率已经过期了，那不如顺手清掉旧的工具结果。

具体做法：把「可重新获取」的工具结果（Read / Bash / Grep / Glob / WebSearch / Edit / Write）清空，只保留最近 5 个。子 agent 输出、Task 状态这类「不可重复」的结果绝不裁剪。

<figure>

<figcaption>Micro-Compact 时间衰减示意图</figcaption>
</figure>

#### 第 4 层：Context Collapse 读时投影

上下文达到 90% 时触发，95% 时升级。这一层最巧妙：**不修改原始消息**，只在调用 API 的那一刻动态计算一个压缩视图给模型。

原对话历史在本地完整保留，模型那边看到的是压缩版。这种「写时不动、读时投影」的思路在很多数据库里也常见。

<figure>

<figcaption>Context Collapse 读时投影示意图</figcaption>
</figure>

#### 第 5 层：Auto-Compact 全量摘要

最重的兜底。上下文逼近窗口上限时触发（具体阈值是从原始窗口里留出约 33K 缓冲，由 20K 摘要输出预留加 13K 额外冗余组成，下文第五节细拆），**把整段对话全送进摘要器重写一份**，配套做文件恢复、缓存清理、消息重组。

这一层是真正意义上的「全量重写」，代价最高（要一次 API 调用）但效果也最强。

<figure>

<figcaption>Auto-Compact 全量摘要示意图</figcaption>
</figure>

#### 为什么要分 5 层

这种「从轻到重」的设计有个隐藏的好处：绝大部分场景**根本走不到最顶层**。前面几层都在替后面「减负」，大文件没塞进上下文，Snip 就不会触发；Snip 把空间释放出来了，后面的重型压缩自然也不需要跑。

但这里有个点必须澄清，免得你翻源码时犯迷糊：第 4 层 Context Collapse 和第 5 层 Auto-Compact 不是「叠着用」的关系，而是**二选一**。Claude Code 内部有个开关来决定这次走哪一套，同一时刻只有一个在管事。源码里写得很直白，一旦 Context Collapse 启用，Auto-Compact 的触发判断会被直接短路掉，token 烧得再多也不会触发它。

为什么不让两个一起上？因为它们的触发线咬得太近。Context Collapse 在 90% 开始动作、95% 兜底，Auto-Compact 正好卡在中间的 93%。两个同时跑，Auto-Compact 会抢在前面把活干完，反手把 Context Collapse 正准备细粒度保存的上下文一锅端掉。与其让它们打架，不如用一个开关明确分工。

> 配图意见：Context Collapse 与 Auto-Compact「二选一」示意图。画一个开关分别指向两条触发线（Context Collapse 的 90%/95% 对 Auto-Compact 的 93%），同一时刻只有一条高亮、另一条灰掉，重点表达「互斥」而不是「叠加」。

还有个细节能帮你少走弯路：Context Collapse 目前还是 Anthropic 内部灰度的实验特性，对外发布的版本会把这部分代码整个裁掉。所以你现在翻公开源码，大概率只能看到 Auto-Compact 这一条线，这也是本文把它当兜底主力来拆的原因。

5 层里有 2 层是纯本地、完全零 API 开销（大结果存磁盘和 Micro-Compact），Snip 那层也只是搭着模型正常回合顺手标记、不另发请求，同样很轻。真正贵的是最顶上那套全量重写，而且只有在前面几层都压不动的极端情况下才会动用。

**这篇文章我们只聚焦最顶上的第 5 层 Auto-Compact**，因为它是面试官最爱拷打的、设计最精妙的、也是最有面试加分价值的。前 4 层我在之前那篇《学习 Claude Code 源码》里讲过完整全景，没看过的可以翻一下。

接下来 5 节，全部是 Auto-Compact 的深度拆解。

### 四、Auto-Compact 的整体思路

Auto-Compact 作为金字塔最顶层的兜底机制，核心就三件事。

第一件，**绝对阈值触发**。不按轮数、不按百分比，按 token 数距离窗口上限的固定缓冲。

第二件，**全量重写对话**。这个最反直觉：所有历史消息，不分新旧，全部送进摘要器重新写一份。

第三件，**关键信息走另外的恢复通道**。文件内容、记忆文件（也就是 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 这类）、异步任务的状态，这些东西不靠摘要保住，靠「重新注入」。

这里顺便插一句「异步任务」是啥：Claude Code 支持主 agent 派几个后台子 agent 同时干活，比如让一个去搜文档、另一个去跑测试。这种正在跑的子任务状态也属于关键信息，压缩时必须保住，不然主 agent 醒来后会不知道子任务进展到哪了。

光这么说估计还是有点抽象，打个比方你立马就懂了。

好比你是一家公司的老板，每个季度要做总结。一种做法是「把所有的会议记录都翻一遍」。这种方法你能记住一堆细节，但脑子里乱成一锅粥。

Auto-Compact 的做法是另一种：会议记录全部归档进档案室（这部分等于丢了，对话里看不到了）；桌面上换上一份新写的「精华版季度纪要」（这是摘要）；员工手册（也就是 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>）该挂哪挂哪不动，下次开会还能翻（下一轮重新加载）；最近用过的几份关键文档（最近 read 的文件），放到桌面随手翻的位置（重新注入为附件）。

这套做法的精髓在哪？在于它承认对话历史是会过时的。过去的细节虽然有价值，但 95% 都不需要原样保留，重要的是「我的目标是什么、我做到哪一步了、犯过哪些错、下一步该干嘛」这种结构化的状态。

<figure>

<figcaption>Auto-Compact 全景图</figcaption>
</figure>

OK，全景有了。下面四节，把这三件事一件件拆开看，先看触发时机。

### 五、什么时候触发压缩？

如果让你来设计你会怎么定？常见思路无非这几种：按对话轮数（每 50 轮压一次）、按 token 数比例（占满窗口 80% 就压）、按时间间隔（每隔 30 分钟压一次）。

这些方案听起来都挺合理。但 Claude Code 都没用，它用的是一个更工程化的思路：**距离上限的固定缓冲**。

#### 阈值是怎么算的？

直接看源码：

```
export const AUTOCOMPACT_BUFFER_TOKENS = 13_000

export function getAutoCompactThreshold(model: string): number {
  const effectiveContextWindow = getEffectiveContextWindowSize(model)
  return effectiveContextWindow - AUTOCOMPACT_BUFFER_TOKENS
}
```

它的触发线是这么算的：拿到模型的「有效上下文窗口」，再减去一个固定值 13k，得到的就是触发阈值。token 数一旦超过这条线，就开始压。

注意这里有个容易被绕进去的地方：「有效上下文窗口」并不等于模型原始的窗口大小。在算它的时候，Claude Code 已经先从原始窗口里抠掉了一块，专门留给摘要自己的输出用。这块预留藏在另一个常量里：

```
// Based on p99.99 of compact summary output being 17,387 tokens.
const MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000
```

上面这两段都在 autoCompact 那个文件里。看注释你就明白了：**摘要任务实际输出长度的 p99.99 分位是 17,387 token**，Anthropic 真的去跑了大规模数据统计，把分布画出来，发现哪怕到 99.99% 这个极端分位，摘要输出也就 17.3k 左右。然后向上取整、再留一点安全冗余，凑成一个整 20k，作为「给摘要输出预留的位置」。

所以这里其实是**两层缓冲叠在一起**，别搞混了：

第一层是这 20k，从原始窗口里减掉，确保压缩时摘要那一大段输出有地方写得下，这一层才是 p99.99=17.3k 那个数据推出来的。

第二层才是前面代码里的 13k，它是在「原始窗口减掉 20k」之后，再额外往回缩的一道安全线，让压缩稍微提前一点触发、别贴着边界跑。源码注释并没有给这 13k 挂上 p99.99 的依据，它就是一道独立的二层保护。

换句话说，**真正离原始窗口的距离是 20k + 13k ≈ 33k**，而不是单看到的 13k。如果你只盯着 13k 还拿它去跟 17.3k 比，会觉得「留的冗余怎么比实际输出还小」，那是因为漏看了前面那 20k。能想到这一层、反推出「留冗余的数应该比 17.3k 大」的读者，思路是很对的。

<figure>

<figcaption>摘要输出长度的 p99.99 分布示意图</figcaption>
</figure>

这个细节看着不起眼，但很显工程师味儿。绝对值阈值的好处是可预测：不管模型窗口未来扩到 500k 还是 1M，触发线永远是「上限减掉这固定的 20k + 13k」，不会因为窗口变大就跟着膨胀。如果用 80% 这种比例，模型窗口越大，剩余预算就越大，但摘要任务实际需要的预算其实没怎么变，那就是浪费。

#### 手动和自动触发的区别

讲到这儿你应该注意到一件事：触发的逻辑是自动判断的。但 Claude Code 还提供了一个手动入口：`/compact` 命令。

那么手动和自动，走的是不是同一套逻辑？

答案有意思：核心压缩函数是同一个，但传入的参数不一样。

手动模式：你可以传一段 customInstructions（直译就是「自定义指令」），告诉摘要器「这次压缩，请特别关注 XXX 方面」。比如你正在调一个特定 bug，希望摘要的时候把这个 bug 的上下文重点保留。

自动模式：不接受用户指令，而且会偷偷打开一个开关叫 `suppressFollowUpQuestions`（直译就是「禁止生成后续提问」）。这个开关的意思是：摘要里禁止生成「需要进一步确认」类的问题。为啥要这么干？因为自动压缩通常发生在 agent 正在干活的时候，你不想压完了对话被一个新问题打断节奏。

<figure>

<figcaption>手动 vs 自动触发对比图</figcaption>
</figure>

#### 熔断和递归守卫

自动模式还多了一个安全机制：**circuit breaker（电路断路器）**。如果 auto-compact 连续失败 3 次，系统就停止重试。

这个机制其实是踩坑踩出来的。源码注释里讲，曾经有 1000 多个会话因为反复触发压缩失败、不停重试，把 API 账单当烟花放。Anthropic 工程师肯定开过紧急复盘会，最后定下来连续失败 3 次就熔断。**这种带着血味儿的设计，才是从生产环境里活下来的**。

<figure>

<figcaption>circuit breaker 熔断状态机</figcaption>
</figure>

还有一个细节也挺有意思，叫**递归守卫**。Claude Code 在跑摘要任务的时候，本质上也是开了一个子 agent 去调模型生成摘要。这个子 agent 自己也是要消耗 token 的，那它会不会因为消耗多了又触发 auto-compact，进入死循环？

不会。源码里就这么一句判断：

```
if (querySource === 'session_memory' || querySource === 'compact') {
  return false
}
```

`querySource` 是当前查询的「来源标签」。压缩任务跑起来时会被标成 `compact`，会话记忆任务会被标成 `session_memory`。一旦判断来源是这两种之一，直接 return false 不再触发压缩。短短三行，就把无限递归这个坑堵死了。

<figure>

<figcaption>触发时机示意图</figcaption>
</figure>

聊完触发，我们看下一个最反直觉的设计：**全量重写**。压缩的时候，到底压什么？留什么？

### 六、压什么、留什么、丢什么？

来到第二个核心设计：压缩的取舍。

#### 全量重写整段对话

这里换个角度问一下：如果让你来设计压缩机制，遇到一段 200 轮的对话，你会怎么处理？

绝大多数人的直觉是：保留最近 20 轮（最相关的），把前面那 180 轮压成一段摘要塞进去。这也是大多数 agent 框架的做法。

但 Claude Code 不是这么干的。它的做法非常激进：**所有 200 轮，全部送进摘要器，重新写一份**。

<figure>

<figcaption>「直觉做法 vs Claude Code 做法」对比图</figcaption>
</figure>

第一眼看到这个设计你大概率会愣一下：最近的对话也不保留？那 agent 不是丢了眼前正在做的事吗？

不过转念一想还真有道理。还记得第一节讲的 **Lost in the Middle** 吗？就算你保留最近 20 轮，模型对中间几轮其实也看不清楚。与其留一堆半模糊的信息，不如全部压成结构化的精华，让模型一眼就能看清。

别急，最近的对话也有另外的恢复通道，等会儿讲。先看压缩之后的对话长啥样：

```
export function buildPostCompactMessages(result: CompactionResult): Message[] {
  return [
    result.boundaryMarker,      // 压缩边界标记
    ...result.summaryMessages,  // 摘要消息
    ...result.attachments,      // 文件、技能、计划等附件
    ...result.hookResults,      // hook 执行结果
  ]
}
```

读一下你就明白了。压缩之后的整个对话历史，被重写成了**四段式结构**。

第一段，**边界标记**。一个特殊消息，记录这次压缩是自动还是手动、压缩前 token 是多少、最后一条消息的 ID 是啥。等于打个时间戳。

第二段，**摘要消息**。这就是大头，前面 200 轮全部被压缩进这里。

第三段，**附件**。包括最近读过的文件、当前的计划文件、激活的技能、正在运行的异步任务状态等等。这就是「另外的恢复通道」。

第四段，**hook 结果**。用户配置的 hooks 在压缩时也会执行，结果一并注入。

所以你看，最近的对话不是没保留，是换了一种形式保留。最关键的几个上下文（文件状态、任务状态），单独走附件通道；语义层面的进度，走摘要；操作层面的最近行为，靠模型的常识去推断。

这套设计的妙处在于：它承认对话的不同信息有不同的「半衰期」。

举两组具体例子你就懂了。

像「用户想给登录接口加验证码」「之前的技术方案改成了 JWT」「我们决定用 Redis 做缓存」这种语义信息，压成一两句话基本不损失，摘要器可以总结。

像「<a href="http://a.py" target="_blank" rel="noopener noreferrer">a.py</a> 第 42 行有个 bug 在改」「文件已经读到第 100 行」「子任务 X 跑完了，输出在 result.txt」这种状态信息，差一个字 agent 就接不上了，必须原样恢复。

<figure>

<figcaption>信息半衰期对照图</figcaption>
</figure>

语义信息走摘要、状态信息走附件，各管各的，效率最高。

记住一个画面：原来 200 条消息的对话，压缩后只剩 4 段东西（边界标记 + 摘要 + 附件 + hook），其他都不见了。

不过你可能立马有个新疑问：200 轮消息全送进摘要器，摘要器自己不会被撑爆吗？这就引出了 Claude Code 的下一手，先做预处理。

#### microcompact 预处理

Auto-Compact 真要跑之前，还会先做一步预处理，叫 microcompact。

什么意思？就是先把对话里那些占大头的「工具调用结果」清空，只留一个元数据占位符。涉及的工具包括 Read、Bash、Grep、Glob、WebFetch、WebSearch、Edit、Write 这些。这些工具的输出动辄几千几万 token，先把内容清掉，对话瞬间瘦一圈。

然后再把瘦完之后的对话送进摘要器，摘要器的负担也小很多，生成的摘要质量也更高。

顺便插一句澄清：microcompact 本身也是第 3 节金字塔里的**第 3 层独立机制**，会按时间衰减（距离上次 API 调用超过 60 分钟）单独触发，并不是只在 Auto-Compact 时才跑。本文为了叙事流畅，重点讲它在 Auto-Compact 流程里扮演的「预处理」角色，但你要知道它有自己的独立生命周期。

<figure>

<figcaption>microcompact 前后瘦身对比图</figcaption>
</figure>

#### 文件恢复：5 个、5K、50K

那有人会问：内容都清空了，agent 后面怎么继续工作？比如最近读的那个文件，内容都没了，agent 怎么知道里面是啥？

这就要讲 Claude Code 的「文件恢复」策略了：

```
export const POST_COMPACT_MAX_TOKENS_PER_FILE = 5_000
export const POST_COMPACT_TOKEN_BUDGET = 50_000
export const POST_COMPACT_MAX_FILES_TO_RESTORE = 5
```

三个常量，看着不起眼，其实是工程化的精华：压缩之后，最多重新加载 5 个文件；每个文件最多塞 5k token；总预算不超过 50k token。

那怎么选这 5 个文件？按「最近活跃度」排，最近被 Read 过的优先。

这个设计实际上是把「最近文件」这个概念量化了。不是模糊地说「保留最近的」，而是用 5 个文件、5k 每文件、50k 总额这三个参数，把它工程化定义死。这样不管对话多复杂，文件恢复的开销都是可控的。

<figure>

<figcaption>文件恢复策略示意图</figcaption>
</figure>

<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md 不进摘要</a>

讲完文件，看另一个值得讲的东西：记忆文件，比如 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>。

你可能会猜：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 这种永久性的指令，压缩之后会被塞进附件里吧？

错。Claude Code 这里做了一个很反直觉的决策：**<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 不会被注入到压缩后的消息里**。

那它怎么生效？答案是：通过「清空缓存」让它在下一轮对话自动重新加载。

逻辑是这样的：Claude Code 内部有个叫 `getUserContext` 的缓存（用来存「用户级上下文」，<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 的内容就放在这）。每次对话开始，会先查这个缓存。压缩的时候，系统把这个缓存清掉。这样下一轮对话发起的时候，Claude Code 发现缓存空了，就会从磁盘重新读 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>。

为什么这么干？因为 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 是「永久存活」的上下文，每一轮都会自动重新加载，不需要塞到压缩后的消息里占地方。这种「全局指令」和「当下对话」是两套机制管理的，互不干扰。

<figure>

<figcaption><a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 双通道处理示意图</figcaption>
</figure>

#### system prompt 和异步任务

还有一个角色得提一下：**system prompt**。

system prompt 完全不参与压缩。压缩之后，会用一个叫 `buildEffectiveSystemPrompt` 的方法（直译就是「构建有效的 system prompt」）重新构造一份新的，注入最新的工具列表、最新的权限设置、最新的 MCP server 列表。等于压缩完顺便把「操作手册」刷一遍。这样如果你在对话中途加了一个新工具，压缩之后 agent 就能立刻用上。

最后还有一类东西要保住：异步任务的状态。

如果你的 agent 起了几个子 agent 在后台跑任务，压缩的时候不能把这些任务状态丢了。Claude Code 把这些任务的状态（正在跑、跑完了、出错了）作为附件重新注入，确保压缩之后主 agent 依然能看到子任务的进度。

<figure>

<figcaption>压缩前后消息链对比图</figcaption>
</figure>

看到这儿你应该明白：Claude Code 的取舍逻辑，本质是信息分类管理。语义信息进摘要、状态信息走附件、永久信息靠缓存清理重新加载、操作配置每次重建。各司其职。

整节读下来如果只让你记一句话，那就是：**不同信息走不同通道，谁也别碰谁的地盘**。

下一个问题，最有料的部分来了：摘要器收到一大堆消息之后，到底是怎么压缩的？它的 prompt 是怎么写的？

### 七、摘要 prompt 是怎么设计的？

终于来到最有料的环节。

很多人觉得，让大模型写个摘要嘛，prompt 不就是「请总结一下以下对话」嘛？告诉你结论：Claude Code 的摘要 prompt 长达**两百多行**，光是「禁止工具调用」这一条警告就在里面重复了两次。听到这儿你大概能猜到，坑不少。

一点一点拆。

#### 防呆设计：禁止工具调用

打开 prompt 文件的第一眼，你会看到这么一段：

```
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.
- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- Tool calls will be REJECTED and will waste your only turn.
- Your entire response must be plain text.
```

翻译一下：「严正警告，只能返回文本，不许调用任何工具。Read、Bash、Grep 这些都不行。你调用了我们会拒绝，你只有这一次机会，调了就废了」。

为什么要这么夹着喊？因为做摘要的本质是让模型只生成文字，把对话压缩成一段总结。但是模型很容易看到对话里提到「这个文件之前 Read 过」就自己手痒去 Read 一下，或者看到一个 bug 就想去 Bash 跑一下。

这种行为在普通对话里很合理（agent 就是要主动调工具嘛），但在做摘要时是大忌：要它总结历史，不是要它做新动作。一旦它去调工具了，这次摘要就废了，得重新来。

源码注释里讲了这是早期 Sonnet 4.6 留下的坑：那个版本的模型经常无视一次警告，所以工程师们干脆「前后包夹」，在 prompt 的开头讲一遍，结尾再讲一遍。

你能想象那个工程师改完 prompt、跑测试一看模型又偷偷调了 Read 的表情吗？干脆前后各喊一遍：「我说了不许调工具」「我再说一遍不许调工具」。这种细节读起来挺有画面感的，**再厉害的模型也有自己的「小动作」，prompt 工程很多时候就是在跟模型的惯性做拉扯**。

<figure>

<figcaption>摘要 prompt「前后包夹」示意图</figcaption>
</figure>

#### 输出格式：XML + 9 部分清单

讲完防呆设计，看正经的输出要求。

Claude Code 要求摘要的输出长这样：

```
<analysis>
[模型的推理草稿，分析对话哪些重要]
</analysis>

<summary>
[结构化的摘要，按 9 个清单分块]
</summary>
```

`<analysis>` 块是「草稿区」，让模型先想清楚再写。这块最终会被剥离掉，不进入压缩后的对话。它的存在纯粹是为了让模型有个「思考的空间」，写出来的摘要质量更高。

`<summary>` 块才是真正进入对话的内容。

<figure>

<figcaption>XML 输出结构示意图</figcaption>
</figure>

这块必须包含 **9 个固定章节**：

<figure>

<figcaption>摘要 prompt 结构示意图</figcaption>
</figure>

1.  Primary Request and Intent（主要请求和意图）
2.  Key Technical Concepts（关键技术概念）
3.  Files and Code Sections（涉及的文件和代码段）
4.  Errors and fixes（碰到的错误和修复方式）
5.  Problem Solving（解决的问题）
6.  **All user messages**（所有用户消息）
7.  Pending Tasks（待办任务）
8.  **Current Work**（当前正在做的事）
9.  Optional Next Step（下一步建议）

这 9 项里其他几项都好理解，重点拎出来讲两项：第 6 项和第 8 项，这俩才是设计的灵魂。

#### 第 6 项：枚举所有用户消息

光看名字你可能以为是「把所有用户消息复制一遍」？不是。它的意思是：**摘要里必须把所有不是 tool result 的用户消息列出来**。

为什么这么强调？因为 agent 是为人类服务的。用户在对话中可能在第 30 轮的时候改了一次需求、在第 80 轮的时候提了一个新约束、在第 150 轮的时候说「之前那个方向放弃吧」。这些信号是任务方向变化的关键，摘要里丢了一个，agent 接下来可能就走偏了。

所以这一项不是「概括」，是「枚举」，一个不能落。摘要器宁可篇幅长一点，也要把用户的所有发言都列清楚。

#### 第 8 项：最细颗粒度的当前进度

这一项的要求是：**用最细的颗粒度描述 agent 当前正在做什么**。

注意是「最细」。不是「正在调试」，而是「正在调试登录模块的 token 刷新逻辑，刚发现 cookie 过期判断有 bug，正准备改 auth.ts 文件的 refreshToken 函数」。

为什么要这么细？想象一下压缩之后下一轮对话，agent 收到摘要后第一件事就是问自己：「我刚才做到哪了？」。摘要里这一项越细，agent 接续工作就越流畅。粗了的话，agent 会陷入「我好像在调试什么东西，但具体是啥？」的迷茫，可能会重新探索一遍上下文，浪费 token 和时间。

所以你看，9 个清单不是凑数。每一项都有它要解决的问题：意图不丢、技术方向不丢、文件状态不丢、错误教训不丢、用户每句话不丢、待办不丢、当前进度不丢。

剩下的 7 项也都各司其职，比如 Errors and fixes 让 agent 不会重复踩同一个坑，Pending Tasks 让 agent 知道还有哪些事没干完。

<figure>

<figcaption>第 6 项 vs 第 8 项对比图</figcaption>
</figure>

#### 摘要用什么模型？

最后一个值得讲的细节：你以为摘要任务会用一个便宜的小模型？

错。Claude Code 用的是**当前对话的同一个模型**（也就是主 agent 用的那个，比如 Opus 4 或者 Sonnet 4）。

为什么不省钱用个小的？两个原因。

第一，摘要质量要保证。便宜的小模型生成的摘要丢东西多，下一轮 agent 就接不上。摘要本身是 agent 接续工作的「灵魂文件」，省这点钱不值。

第二，Prompt Cache 复用。这里插一句解释，prompt cache 是大模型 API 的一个能力：如果你这次请求和上次请求的开头部分（比如 system prompt、工具描述）是一样的，这部分会被缓存住，下次只算一次钱、推理也更快。Claude Code 的主对话本来就在用 prompt cache，压缩用同一个模型，能复用 system prompt 那部分的 cache，省下来的钱比换小模型省的还多。

聊完压缩，看最后一个机制问题：压完之后，对话怎么接得上？

### 八、压完之后怎么接续对话？

压缩本身讲完了，但还有一个工程问题：摘要生成完了，怎么把它塞回去，让对话无缝接下来？

这个细节其实挺关键。如果接得不好，模型下一轮会一脸懵：「我前面好像聊了好多东西，怎么突然就一段摘要？我是不是该重头来一次？」

#### 压缩流水线五步走

Claude Code 是这么处理的，整个流水线大概是这么走：

第一步，把当前所有消息送进摘要器，生成摘要文本。

第二步，清空各种缓存，包括 readFileState（文件状态缓存）、loadedNestedMemoryPaths（嵌套记忆文件路径缓存）、getUserContext（用户上下文缓存）。这一步保证下一轮对话所有的状态都从干净的起点开始。

第三步，**并发**生成各种附件：最近的文件、异步 agent 的状态、技能配置等等。注意是并发，不是顺序，这能省一些时间。

第四步，调用前面讲过的 buildPostCompactMessages 把所有东西组装成新的消息链。

最后一步，新的消息链替换掉旧的，对话继续。

整个过程对外是透明的。用户那边看到的就是一句轻飘飘的提示「Compacted」，然后 agent 接着干活。

#### 旧消息真的丢了吗？

那旧的消息哪去了？真的丢了。

除非你打开了 Kairos 模式（一种 transcript 备份机制），那种情况下旧消息会写到本地的一个 transcript 文件里，可以事后回查。但在对话本身里面，旧消息是回不来的，压缩是一次性的、破坏性的操作。

听起来挺暴力对吧？但其实是个明智的设计。如果允许「回滚」，整个对话状态就会很复杂，要维护「压缩前/压缩后」两套消息，token 也省不下来。一刀切，反而干净。

<figure>

<figcaption>旧消息「真丢了」示意图</figcaption>
</figure>

#### 让模型无缝接活的小心思

不过 Claude Code 留了一个小心思：摘要的开头会被包装成这样一句话：

「本会话是从之前一次因上下文耗尽而中断的对话延续过来的。以下摘要概述了之前的对话内容。」

这句话很重要。它告诉模型一件事：**你不是从头开始的，你是接着干**。模型看到这句话，就不会傻乎乎地问「请问您想做什么」，而是直接顺着摘要的「Current Work」往下接活。

而且摘要的末尾还会带一个 transcript 文件路径。意思是说：「如果你需要查之前对话里的某个具体代码片段或者细节，可以读这个文件」。这给了 agent 一个「翻底牌」的兜底通道。

最后讲一个看着小但效果好的开关，叫 suppressFollowUpQuestions（前面也提过，中文直译就是「禁止生成后续提问」）。

这个开关只在 auto-compact 时打开。它的作用是：禁止摘要器在 Current Work 那部分生成「需要进一步确认」类的问题。

为什么？想象一下场景：用户正在让 agent 干一个长任务，token 烧着烧着触发了自动压缩。如果摘要里塞了一个问题：「请问您是希望优先 A 还是 B？」，那 agent 下一轮就会卡在这个问题上等用户回答，整个任务流被打断了。

而手动 `/compact` 的时候，这个开关是关闭的，因为用户主动触发压缩，本来就是个干预动作，模型问个问题确认一下也没毛病。

<figure>

<figcaption>压缩接力流程图</figcaption>
</figure>

到这儿，Claude Code 的整套压缩机制就拆完了。

### 九、面试该怎么答？

原理讲完了，回到文章开头那道面试题。下面这套节奏建议你背一下，下次面试官再问「Claude Code 的上下文窗口是怎么管理的？」，照着答就行。

你可以**先一句话亮观点**：「Claude Code 用的不是滑动窗口、定期摘要、向量召回那一类常见方案，而是一种叫『全量重写加分通道恢复』的工程化思路。」

这一句话本身就比 90% 的候选人讲得有结构感了。

<figure>

<figcaption>面试话术四层结构卡片图</figcaption>
</figure>

接下来分四层铺细节，节奏要稳。

第一层，触发时机。用的是绝对 token 阈值，公式是「有效上下文窗口减去 13k 缓冲」。这里有个细节最能体现你真翻过源码：所谓「有效上下文窗口」本身就已经先从原始窗口里抠掉了约 20k，专门留给摘要自己的输出，而这 20k 才是基于摘要任务 p99.99 输出长度（实测 17,387 token）向上取整加冗余得来的；13k 是在这之上再加的一道独立缓冲。所以离原始窗口的距离其实是 20k 加 13k 约 33k。这一层讲清楚，面试官就知道你不是临时抱佛脚。

第二层，取舍逻辑。反直觉的点是它不保留最近 N 条，而是把所有历史消息一刀切全部送进摘要器重写一份。关键的状态信息单独走「附件通道」恢复，比如最近读过的 5 个文件（每个最多 5k token，总预算 50k）、异步任务状态、当前计划文件。<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 这种永久指令则不进摘要，而是通过清空 getUserContext 缓存让它在下一轮自动重新加载。

第三层，摘要 prompt 设计。用 9 部分结构化清单约束输出，重点强调「所有用户消息」必须枚举不能落、「当前正在做的事」要精确到文件和函数名。摘要用的是当前对话的同一个模型，不省钱换小模型，目的是保证质量，顺便复用 prompt cache。

第四层，接续机制。压缩之后会清空文件状态缓存、并发生成附件、用 buildPostCompactMessages 把新消息链组装好。摘要外面包一句「本会话是从之前一次因上下文耗尽而中断的对话延续过来的」，让模型知道自己是接力不是从头开始。自动触发时还会打开 suppressFollowUpQuestions 开关，避免摘要里塞新问题打断当前任务。

讲完这四层，**最后用一句话收口**：「Claude Code 这套设计反映的不是『省 token』的小聪明，而是『信息分通道管理』的工程哲学。」这句话留给面试官细品，整段回答就稳了。

整段回答下来大概一分半到两分钟，节奏从「为什么这么做」聊到「具体怎么做」，再升华到设计哲学。面试官想追问任意一层你都能展开（那约 33k 缓冲是怎么拆成 20k 加 13k 的、9 部分清单是哪 9 部分、circuit breaker 怎么防止烧钱），都是文章里讲过的细节，拿来就能用。

### 写在最后

回头看 Claude Code 的这套压缩机制，最戳人的是一句话：**上下文管理不是「省 token」，是「保信息结构」**。

很多 agent 项目把上下文压缩当成一个性能优化问题来做，逻辑是「窗口要爆了，赶紧砍点东西」。

但 Claude Code 不是这么想的，它把上下文管理当成 agent 的「灵魂工程」：意图、进度、错误教训、待办、用户每一句话，这些才是 agent 能接续工作的本钱。token 数只是表面，保住这些结构化信息才是核心。

这种工程思维其实挺值得学。做 agent 时太容易陷入「这个 token 多了、那个 prompt 长了」的细节优化，反而忘了思考「我的 agent 到底依赖哪些上下文信息才能干好活」这个本质问题。

Claude Code 用一套精细的分层管理告诉你答案：不同信息有不同的半衰期，要分别管理。

最后留一个开放问题给你思考：

当未来上下文窗口扩到 1 亿 token，是否还需要 compaction？

我的预测是：仍然需要。

理由是 Lost in the Middle 现象不是上下文窗口大小的问题，是注意力机制的固有特性。窗口再大，中间段的信息照样会被模型「看不清楚」。所以无论窗口怎么膨胀，**主动管理信息结构**这件事永远不会过时。

Anthropic 这套机制对所有做 agent 的同学来说，都是教科书级的范本。

希望这篇文章看完，能让你下次面试时把整套机制讲明白，让面试官眼前一亮。

我们下篇见。

## Claude Code 代码检索图解：为什么用 grep 而不用 RAG？

> Source: https://xiaolinnote.com/claudecode/source/cc_grep.html

大家好，我是小林。

前阵子，有个林友跟我聊起他面试字节 AI Agent 岗的经历。

被面试官问了一个问题：「为什么 Claude Code 不用 RAG 检索代码，而是直接用 grep？」

<figure>

</figure>

我一愣。

是啊，这两年 RAG 火成这样，几乎成了 Agent 的标配。谁去做个 AI 编程工具不上 RAG？

结果 Claude Code 这么个被业内认证「最好用的 AI 编程工具之一」，反其道而行，连 embedding 和向量数据库的影子都没有，就靠 grep 加读文件这种最朴素的方式来获取代码上下文。

是 Anthropic 没钱搞向量库吗？显然不是。

是工程师水平不够吗？更不可能。

那这背后到底是什么设计哲学？为什么他们偏偏不用 RAG？

我把 Claude Code 的检索相关源码翻了个底朝天，越看越觉得这个选择很有意思。

<figure>

</figure>

今天就从源码视角，带你一层层拆开这个问题我会按这个顺序展开：

- 先搞懂代码检索到底要解决什么？
- 看 RAG 派是怎么干的？ RAG 在代码场景有什么坑 ？
- Claude Code 是怎么反向思考的？
- Claude Code 检索三件套怎么组合 ？
- Claude Code 派子 agent 去检索是怎么回事？

读完这篇文章，你不光能答出这道面试题，还能理解 Anthropic 对「Agent 应该长成什么样」的整套思路。

------------------------------------------------------------------------

### 一、先搞懂「代码检索」到底要解决什么问题

很多人一上来就跳进 RAG 和 grep 的对比，但我觉得这样讲很容易让人迷糊。

不如咱们先退一步，把问题本身搞清楚：什么叫代码检索？为什么 Agent 写代码非得有这一步？

你想啊，你打开 Claude Code，跟它说「帮我把 UserService 里那个登录方法改一下，加个验证码逻辑」。模型要完成这个任务，第一件事得是什么？

得是「找到 UserService 这个文件、找到登录方法的具体代码」。代码都没看到，怎么改？

但问题来了：模型自己又不能直接「看」磁盘上的代码，它只能看到你塞进上下文里的文字。

那你说，那把整个项目的代码全塞进去不就完了？

理论上是这么个理。但 LLM 的上下文窗口是有限的。就算用 Claude Opus 4.7 这种已经支持 1M token 的超长上下文模型，听着是真的大对吧？换算一下大概有 200 多万字。但你想想，一个稍微像样点的项目动辄几百万行代码，再算上依赖库源码就更夸张了，光是这些就远超模型的承载量，更别说还要留位置给系统提示、对话历史、工具调用结果。

塞不下，还是塞不下。

<figure>

<figcaption>上下文窗口 vs 代码库体积对比图</figcaption>
</figure>

所以代码检索这一步就出来了：从一堆代码里，**精准地捞出和当前任务相关的几个片段，再塞给模型**。

这有点像你查字典。一本汉语词典几千页，你不会从头读到尾，而是先翻目录、看拼音索引，定位到「龘」这个字所在的那一页，然后只读那一页。代码检索干的就是这个活儿。

<figure>

<figcaption>字典查目录类比图</figcaption>
</figure>

那「怎么查目录」、「怎么定位到那一页」，就是不同方案的差异所在了。

------------------------------------------------------------------------

### 二、绕不开的 RAG：「先建库再查」的经典思路

说到检索，做过 AI 应用的同学第一反应肯定是 RAG。这玩意儿现在是 Agent 圈的「网红技术」，几乎一提到「让 LLM 用外部知识」，就想到它。

那 RAG 到底是怎么做的？我用一个图书馆的故事给你讲明白。

想象你是一个图书馆的管理员。馆里有几十万本书，读者来问「我想看一本关于宋代茶文化的书」，你怎么办？

你不可能跑遍整个馆每本书翻一遍。最聪明的做法是：**提前把每本书都做好分类卡片**。卡片上写清楚书名、作者、关键词、内容摘要，按主题归类好放进抽屉里。读者一来，你照着卡片找对应的抽屉，几分钟就能把书翻出来。

RAG 在代码场景下，干的就是这个事儿。它的工作流可以拆成四步。

**第一步，切片（Chunking）。**

先把代码文件按某种规则切成小片段。常见的切法是按函数切、按类切，或者按固定行数切，比如每 100 行一段。这相当于把厚厚的书拆成一篇篇文章。

<figure>

</figure>

**第二步，向量化（Embedding）。**

每个代码片段过一遍 embedding 模型，转成一串数字（一个高维向量）。你可以理解成给每个片段算了一个「语义指纹」，意思相近的片段，指纹也相近。

<figure>

<figcaption>Embedding 演示：代码经过模型变成数字向量</figcaption>
</figure>

**第三步，建索引（Indexing）。**

把所有的向量存到一个专门的向量数据库里，比如 Faiss、Pinecone、Milvus 这些。这一步就是「把分类卡片整理好，按抽屉摆放」。

**第四步，召回（Retrieval）。**

用户提问的时候，把问题也向量化一下，去库里找最相似的 Top-K 个片段，比如最像的 5 个，然后把这几段代码拼到 prompt 里，丢给 LLM。

<figure>

<figcaption>向量召回 Top-K 示意图</figcaption>
</figure>

讲到这里，你可能已经看出 RAG 的核心套路了。

**它把「找代码」这件事，转成了「算相似度」**。所有的检索逻辑，最后都归结为「在向量空间里找最近的几个邻居」。

这套思路在很多场景下确实好使。比如你做一个客服机器人，公司内部有一万篇 FAQ 文档，用户问问题，RAG 拍拍脑袋就能给你召回最相关的几篇，效果妥妥的。

那放到代码上呢？是不是也一样无敌？

别急，下一节咱们就来扒它的痛处。

------------------------------------------------------------------------

### 三、RAG 在代码场景下的「水土不服」

如果 RAG 真的完美适合代码场景，那 Claude Code 早就上了。问题是它有一堆坑，咱们一个一个看。

#### 痛点 1：代码不像散文，切不动

文章是流式的，你拦腰切一刀，损失不大，前后段都还能独立读。但代码不一样，代码是有严格结构的。

举个例子，一个函数 200 行，按 100 行一段切了。结果上半段是个 `if` 的开头，下半段是 `else` 的结尾，模型看到的两个片段，一个少了 else，一个少了 if，这玩意儿压根没法用。

更糟糕的是，函数 A 调用了函数 B，但是 A 和 B 在不同片段里。模型只看到 A，根本不知道 B 是干嘛的，幻觉概率直接拉满。

<figure>

<figcaption>函数被切两半示意图</figcaption>
</figure>

#### 痛点 2：精确匹配的活，向量干不了

向量召回的本质是「找相似的」，不是「找对的」。这在代码场景就麻烦了。

你跟 Claude Code 说：「帮我看下 `getUserById` 这个函数的实现」。

向量召回会怎么干？它会找一堆「跟用户相关的查询函数」给你：`getUserByName`、`getUserByEmail`、`fetchUserInfo`、`queryUser`，啥都来一遍。但你要的就是 `getUserById` 这一个具体的函数啊。

这就好比你跟同事说「帮我叫一下张三」，结果他把李四王五马六全叫过来了，因为他们都是同事。

向量擅长「模糊」，但代码很多时候要的就是「精确」。

> 配图意见：向量召回错误示意，用户问「找张三」，结果向量库返回「李四王五马六」

<figure>

<figcaption>向量召回错误示意图</figcaption>
</figure>

#### 痛点 3：代码每天在变，索引咋办

这是 RAG 在代码场景的另一个噩梦。

你建好索引了，开发同学一个 commit 改了 20 个文件，新增 3 个函数。索引怎么办？

要重建？整个项目重新切片、重新 embedding，成本不低，频繁 commit 直接性能爆炸。

不重建？模型查到的全是旧版本，用过期信息写新代码，bug 不写都难。

要做增量更新？听着合理，但实现起来一堆边界情况：哪些 chunk 要删、哪些要重新算、跨文件的引用关系怎么同步。说白了，简单事情搞复杂了。

<figure>

<figcaption>索引重建成本图</figcaption>
</figure>

#### 痛点 4：冷启动慢得要命

你在一个百万行的代码库上跑 RAG，光是建索引就要十几分钟甚至更久。

用户打开工具，看着进度条转，等几分钟才能开始用？这体验直接劝退一半人。Claude Code 的设计理念是「打开就能用」，RAG 这套显然不答应。

<figure>

<figcaption>冷启动等待图</figcaption>
</figure>

#### 痛点 5：黑盒，不可解释

这点我觉得最关键。

向量召回回来 5 个片段，你问为什么是这 5 个？没人答得上来。是因为 cosine 相似度高？那为什么这 5 个比那 5 个高？模型说不清，工程师也说不清，因为这玩意儿是黑盒。

出了 bug 你都不知道从哪儿查起。模型答错了，是召回错了？还是召回对了但模型理解错了？追溯链路非常痛苦。

总结一下：RAG 在「静态文档、自然语言、模糊匹配」这种场景是利器，但代码恰好是反过来的，是动态、结构化、需要精确的。RAG 在代码上不是不能用，是用得很别扭。

那不用 RAG，又怎么搞代码检索呢？

------------------------------------------------------------------------

### 四、Claude Code 的反向思路：把检索还给模型自己

在讲 Claude Code 的方案之前，我想先抛个反问：你是怎么定位代码的？

我猜大部分程序员的工作流是这样的。

接手一个陌生项目，第一件事先看下目录结构，`ls` 一下心里有个数。要找某个功能在哪实现，第一反应是 `grep -r` 全局搜个关键字。grep 出一堆候选文件，挨个 `cat` 看下哪个最像。找到了，再用编辑器跳转打开，前后翻翻看上下文。

整个过程就是「找文件 → 找内容 → 看具体代码」三件套，循环往复，直到你找到目标。

注意一下：你没有提前给整个项目建索引、做向量化吧？没有。你就是「现用现找」。

那你猜 Claude Code 的检索哲学是什么？

简单到让人意外：**让模型像程序员一样，自己去找。**

不预处理、不建库、不算向量，每次需要代码就实时去现场查。工具就给三个：

- **Glob**：按文件名 pattern 找文件（对应你的 `find`）
- **Grep**：按内容关键字找代码（对应你的 `grep -r`）
- **Read**：按需读文件内容（对应你的 `cat` 和编辑器跳转）

<figure>

<figcaption>程序员工作流与 Claude Code 三件套对应关系</figcaption>
</figure>

看似原始，但本质上是把「找代码」的决策权还给了 LLM。你不需要提前猜模型会问什么，也不需要给它装个「智能召回引擎」，模型自己想搜什么就搜什么，搜完看一眼结果，再决定下一步。

这思路简单到让你怀疑人生：就这？真的够用？

我一开始也这么想。但后面越扒源码越发现，这套「土味」工具背后藏着一堆精巧设计。下一节我们就把三件套挨个拆开。

不过先别急，这里我先埋个伏笔：单纯靠主 agent 自己一边 grep 一边 read，遇到大型项目会不会上下文爆炸？这个问题第六节再揭开。

------------------------------------------------------------------------

### 五、Claude Code 检索三件套，到底怎么组合？

这一节有点长，因为三件套每个都有讲究。咱们先讲原理，最后再贴源码加深理解。

#### 5.1 Grep：基于 ripgrep，但绝不是简单封装

讲 Grep 工具之前，我想先问你一个问题：既然 Claude Code 已经给模型开放了 Bash 工具，模型完全可以自己跑 `grep -r "xxx" .`，为啥还要单独包一个 Grep 工具？

这问题很多人没认真想过。但你仔细品，这里面有三层考虑。

**第一层，权限统一管控。**

Bash 是个万能工具，模型理论上能跑任何命令。如果让它自己跑 grep，那 rm 也能跑、curl 也能跑、git push 也能跑。Claude Code 把 grep 单独包成工具，相当于在这个高频操作上单独画了一道权限闸门，更安全。

**第二层，输出格式可控。**

你直接跑 bash grep，输出就是一坨纯文本。但 Grep 工具可以提供结构化输出：行号、上下文行、按文件分组、甚至支持「只返回匹配文件名」、「只返回匹配数量」三种粒度。模型按需选择，token 浪费少很多。

**第三层，性能。**

Claude Code 的 Grep 底层用的是 ripgrep，不是传统 grep。ripgrep 是 Rust 写的，多线程并行、自动尊重 .gitignore（不去搜 node_modules 这种垃圾目录），性能甩老牌 grep 几条街。

设计意图都讲完了，咱们看下源码 `src/tools/GrepTool/prompt.ts` 里的工具描述（这段会出现在模型的 system prompt 里，直接告诉模型怎么用）：

```
A powerful search tool built on ripgrep

- ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash 
  command. The Grep tool has been optimized for correct permissions and access.
- Output modes: "content" shows matching lines, "files_with_matches" shows 
  only file paths (default), "count" shows match counts
- Use Agent tool for open-ended searches requiring multiple rounds
```

中文意思大致是：

```
基于 ripgrep 打造的强力搜索工具

- 搜索任务请永远使用 Grep。绝对不要用 Bash 命令调用 `grep` 或 `rg`。
  Grep 工具已经针对权限和访问做过优化。
- 输出模式："content" 返回匹配的具体行，"files_with_matches" 只返回
  文件路径（默认），"count" 只返回匹配数量
- 开放式、需要多轮迭代的搜索，请用 Agent 工具
```

注意第二行那句话，「ALWAYS use Grep ... NEVER invoke grep or rg as a Bash command」，语气特别强硬对不对？

这就是 Anthropic 在用 system prompt 强制模型走专用工具，不许用 bash 抄近路。

最后一行也埋了个伏笔，「open-ended 多轮搜索请用 Agent 工具」，这个咱们第六节细聊。

> 配图意见：Grep 三种输出模式示意，content 模式给行号+内容，files 模式只给文件名列表，count 模式只给数字

<figure>

<figcaption>Grep 三种输出模式示意图</figcaption>
</figure>

#### 5.2 Glob：按文件名找，按修改时间排序

Grep 是按内容找，那 Glob 是干啥的？是按文件名找。

举个场景：你想看下项目里所有的 `.tsx` 文件有哪些。用 Grep 是不行的，因为你没有内容关键字，你只知道扩展名。这时候 Glob 就上场了，它支持 `**/*.tsx` 这种 pattern。

Glob 工具还有两个小巧思。

**第一，结果按修改时间倒序排列。** 也就是说，最近改过的文件排在前面。为啥这样？因为大部分时候，「最近改过的」就是「跟当前任务最相关的」。这是个很朴素但很有效的启发式规则。

**第二，结果有 100 文件硬上限。** 超出会截断，避免输出爆炸把上下文塞满。模型如果还想看更多，可以收紧 pattern 再搜一次。

是不是很像你平时用 IDE 的「最近打开文件」列表？设计哲学是相通的。

<figure>

<figcaption>Glob pattern 示意图</figcaption>
</figure>

#### 5.3 Read：按需读取，绝不贪心

找到文件了，下一步就是看内容。Read 工具就是干这个的。

但 Read 的设计有个反直觉的地方：它**默认只读 2000 行**，超出会截断。

你可能要问：那要是文件 5000 行咋办？

很简单，模型可以指定 `offset` 和 `limit` 参数，分段读取。比如先读 1 到 2000 行看看大概结构，确定要看的具体位置，再 `offset=3500, limit=500` 精准读那一段。

这套设计的核心思想就一句话：**模型应该按需读取，不要贪心**。

源码 `src/tools/FileReadTool/FileReadTool.ts` 的工具描述里有这么一段原文：

```
By default, it reads up to 2000 lines starting from the beginning of the file.
When you already know which part of the file you need, only read that part.
This can be important for larger files.
```

中文意思大致是：

```
默认从文件开头读取，最多读 2000 行。
如果你已经知道需要文件的哪一部分，就只读那一部分。
对大文件来说，这一点特别重要。
```

「需要哪部分就只读哪部分」，这就是 Anthropic 在引导模型形成节约 token 的习惯。

还有一个非常关键的细节：Read 工具每次都直接 stat 磁盘文件、读取最新内容，**不缓存、不索引、不预处理**。

这意味着什么？意味着只要你刚改了文件，下一次 Read 立刻能看到新内容。这就是 Claude Code 实时性的来源。没有索引层，就没有索引滞后。

<figure>

<figcaption>磁盘实时读取 vs 索引缓存对比图</figcaption>
</figure>

#### 5.4 三件套的组合用法

讲完单个工具，最关键的是看它们怎么组合。

我举一个真实场景：你跟 Claude Code 说「这个项目登录功能在哪实现的？」

它的检索过程大概是这样：

第一步，先用 Glob 找候选文件，比如 `**/*login*.{ts,tsx,js}`，可能拉回来 5 个候选文件。

第二步，用 Grep 在这些文件里搜关键字，比如 `passport|auth|login`，定位到具体的几个命中行。

第三步，用 Read 读命中文件的相关行段，看具体实现。

整个过程是模型一步步推进的：每一步看到上一步的结果，决定下一步搜什么、读什么。

<figure>

<figcaption>Glob Grep Read 三件套组合工作流</figcaption>
</figure>

注意没有，这里没有「一次性召回所有相关代码」的步骤，而是「每一步都基于上一步的结果调整方向」。这是和 RAG 范式最大的不同点，我们第七节再细讲。

但是有个问题：如果是更复杂的任务呢？比如「调研一下整个项目的认证模块流程」，这种活儿三件套循环几次就能搞定吗？

下一节揭晓。

------------------------------------------------------------------------

### 六、当三件套不够用：派子 agent 去探索

来想象这么一个场景：你跟 Claude Code 说「调研一下这个项目的认证模块整体流程」。

这种「调研」类任务有什么特点？需要看的东西多，要 grep 好几个关键词、读好几个文件、来回比对、最后总结成一段结论。整个过程可能要十几个工具调用。

如果让主 agent 自己一边 grep 一边 read 地干，会发生什么？

主 agent 的上下文很快就会被一堆 grep 输出加文件片段塞满。等它好不容易想清楚「认证流程」、要回头给你写代码的时候，发现真正要解决的问题已经被检索过程的中间结果挤到角落里了，模型注意力被分散，质量直线下降。

这就是大型探索任务的最大敌人：**上下文污染**。

<figure>

<figcaption>主 agent 上下文被污染示意图</figcaption>
</figure>

Claude Code 的解决办法很妙：**派一个子 agent 出去探索**。

类比一下：老板要做战略决策，他要看大量的市场数据。他不会自己一头扎进 Excel 里翻几个小时，而是把这事派给秘书：「你帮我调研一下，明天给我一份精简报告」。秘书看了一堆资料，最后只把结论给老板。老板的注意力（对应主 agent 的上下文）就被保护起来了。

子 agent 的派遣机制具体是这样。

**第一，主 agent 通过 Agent 工具派子 agent。** 子 agent 是一个独立的运行实例，有自己的对话上下文，跟主 agent 完全隔离。

**第二，子 agent 拿到一个精简的工具池。** 通常是只读工具：Grep、Glob、Read、Bash（只读命令），但不能 Edit、不能 Write、不能再派子 agent（防止层层嵌套递归）。这种 agent 在源码里有个名字叫 Explore agent。

**第三，子 agent 在自己的上下文里多轮迭代。** 它可以 grep 几十次、read 几十次，过程中产生的中间结果都留在它自己的上下文里，跟主 agent 无关。

**第四，子 agent 完成后，只把最终结论返回给主 agent。** 主 agent 的上下文里只多了一段「认证流程是这样的：……」的精简结论，所有的搜索过程都被压缩没了。

<figure>

<figcaption>派子 agent 隔离与结论压缩示意图</figcaption>
</figure>

这就是「上下文压缩」的精髓：**用一个隔离的子 agent 把脏活儿干了，主 agent 只接收干净的结论**。

源码 `src/tools/AgentTool/prompt.ts` 里有明确的引导规则：

```
For simple, directed codebase searches use Grep/Glob/Read directly.
For broader codebase exploration and deep research, use the Agent tool 
with subagent_type=Explore. ... use this only when ... your task will 
clearly require more than 3 queries.
```

中文意思大致是：

```
对简单、明确目标的代码搜索，直接用 Grep/Glob/Read。
对范围更广的代码库探索和深度研究，请使用 Agent 工具，并指定
subagent_type=Explore。……只有当你的任务明显需要超过 3 次查询时，
才用这种方式。
```

简单定向搜索（你知道要找啥）就直接用 Grep/Glob/Read；开放式探索（你不太确定要找啥）就派 Explore 子 agent。临界点大概是「预期超过 3 次查询」。

这是个非常实用的工程经验：少于 3 次就别折腾派 agent，多于 3 次就别污染主 agent 上下文。

还有一个加分项：**子 agent 可以并行派多个**。比如你要同时调研「认证模块」、「支付模块」、「订单模块」三块，主 agent 可以一次性派出三个子 agent 各干一块，干完同时回来报告。这种并发探索能极大缩短整体时延。

<figure>

<figcaption>并行派多个 Explore 子 agent 示意图</figcaption>
</figure>

到这里，你应该能看出 Claude Code 检索系统的层次感了：

- **底层**：Grep / Glob / Read 三件套，处理简单定向检索
- **中层**：派 Explore 子 agent，处理开放式探索和上下文隔离
- **上层**：主 agent 编排整体任务

每一层都有自己的职责，不互相干扰。

------------------------------------------------------------------------

### 七、再深一层：LLM-driven 的多轮迭代循环

到这儿你可能心里还有个疑问：到底是什么让 Claude Code 能「自己探索」？三件套加子 agent 都讲了，但好像还差一层东西没说透。

差的就是「**多轮迭代**」这层。

我用一个对比讲清楚。

**RAG 的范式是「考试发卷子」**：用户提问 → 系统一次性召回 Top-K 个片段 → 模型基于这些片段一次性生成答案。中间没有循环，没有反悔，没有「等等再看看」。这是一锤子买卖。

**Claude Code 的范式是「现场探案」**：用户提问 → 模型说「我先 Grep 一下」→ 系统执行 Grep 返回结果 → 模型看到结果说「嗯，看起来 UserService 比较像，我 Read 一下」→ 系统执行 Read 返回内容 → 模型说「找到了，逻辑是这样的：……」。这是循环、是边查边推理。

<figure>

<figcaption>RAG 一锤子 vs Agent 多轮循环对比图</figcaption>
</figure>

这个循环本质上就是 query 主流程里的一个 while 死循环，源码 `src/query.ts` 里的核心逻辑大概长这样：

```
while (true) {
  const response = await callLLM(messages)
  if (没有 tool_use) break  // 模型不再调工具了，循环结束
  for (const toolUse of response.toolUses) {
    const result = await executeTool(toolUse)
    messages.push(result)  // 把结果回填到对话历史
  }
}
```

每一轮：模型说话 → 可能带 tool_use → 执行工具 → 把结果回填到对话历史 → 模型继续说话。直到模型自己说「我搞定了」，循环才停下来。

<figure>

<figcaption>query 多轮循环示意图</figcaption>
</figure>

这个看似简单的循环，其实是 Agent 范式的灵魂：**它给了模型在每一步根据上一步结果调整方向的能力**。

你看到 Grep 结果是空的？那就改个关键字再搜。

你看到 Read 出来的代码逻辑不像你以为的？那就再 Grep 几个相关函数看看。

你发现这个文件引用了另一个文件？那就跟过去看下那个文件。

这种「走一步看一步」的能力，是 RAG 的「一次召回」给不了的。RAG 召回错了就是错了，模型只能将错就错。Agent 召回错了，下一轮自己就调整了。

所以你看，Claude Code 用看起来很原始的 grep + read，能做出 RAG 都做不到的事，根本原因就在这层 LLM-driven 的多轮迭代上。

grep 本身不强，但「让 LLM 自己决定每一轮 grep 什么」就强了。

------------------------------------------------------------------------

### 八、回到原题：到底为什么 Claude Code 不用 RAG？

讲了这么多，咱们把答案串起来。Claude Code 不用 RAG 主要有六个原因。

第一，**冷启动**。grep 是毫秒级响应，开箱即用；RAG 要先建索引，分钟级冷启动，劝退一半用户。

第二，**实时性**。grep 每次现读磁盘最新版本；RAG 索引会滞后，文件改了得重建。

第三，**精确性**。grep 是确定性的字符正则匹配，要找 `getUserById` 就只有它；RAG 是向量近似匹配，会把一堆相似函数糊在一起。

第四，**Token 经济**。grep 加 Read 按需读取，模型只看真正需要的几行；RAG 一上来就要给整个代码库做 embedding，存储和计算成本都不小。

第五，**可解释性**。grep 每一步检索过程都对用户透明可审计；RAG 的 Top-K 召回是黑盒，出 bug 没法 debug。

第六，**决策权**。grep 让 LLM 自己决定每一轮搜什么、读什么，多轮迭代逐步逼近答案；RAG 是一次性把材料丢给模型，模型只能将错就错。

<figure>

<figcaption>Claude Code grep 与 RAG 六大原因对比表</figcaption>
</figure>

但如果再升一层，我觉得这背后还有更根本的东西：**两种方案代表了两种不同的设计哲学**。

RAG 派的潜台词是：**LLM 不够强，所以我们要用工程手段帮它把材料准备好**。chunking、embedding、向量召回，本质都是「替模型做决定」。

Claude Code 派的潜台词是：**LLM 已经足够强，工程的角色是给它准备好工具，把决策权还给它**。grep 不替模型做任何决定，它只是个工具。用还是不用、什么时候用、怎么用，全是模型说了算。

Anthropic 押注的是「模型会越来越强」，所以他们选择信任模型的判断能力。这是个长期主义的选择。

<figure>

<figcaption>RAG 与 Claude Code 设计哲学对比图</figcaption>
</figure>

------------------------------------------------------------------------

### 九、那 RAG 是不是该被淘汰了？

讲到这儿，可能有朋友要给我发消息了：「林哥你这是黑 RAG 啊？我们项目还在用 RAG 呢！」

别急，我没说 RAG 该淘汰。RAG 仍然有它的舞台，只是不在 Claude Code 这种场景里。

什么场景适合 RAG？

第一种，**巨型代码库加跨仓库检索**。比如一些大公司有几十个 monorepo、上千万行代码，靠 grep 在整个公司代码库里搜，性能扛不住，这时候建好索引的 RAG 就有用武之地。

第二种，**纯语义查询**。比如「找一下处理用户认证相关的代码」这种描述性、模糊性的问题，用关键字 grep 反而不好搜，向量召回这时候反而有优势。

第三种，**多人协作的知识库类查询**。代码加文档加 Wiki 全部混合检索，这种场景 RAG 是合适的。

而 Claude Code 这套方案，最适合的是**单项目、探索式开发、需要精确性、要求实时性**的场景，恰好是大部分 AI 编程工具的主战场。

工具是为场景服务的，没有银弹。

最后留一个开放问题给你思考：如果未来 LLM 的上下文窗口能到 1 亿 token，整个代码库都能塞进去，RAG 还有意义吗？grep 还有意义吗？我自己也没想得特别清楚，欢迎你在评论区跟我聊聊。

<figure>

<figcaption>grep 与 RAG 场景适配对比图</figcaption>
</figure>

------------------------------------------------------------------------

### 结尾：回到面试题

文章开头那个面试题，现在你应该能漂亮地答出来了。我给你一个三句话的精炼版：

> Claude Code 不用 RAG 是基于三层考虑。  
> 第一，代码场景下 RAG 有切片破坏结构、向量近似不准、索引滞后等本质问题；  
> 第二，Claude Code 用 Grep 加 Glob 加 Read 三件套加上派子 agent 探索的设计，本质上是把检索决策权还给 LLM 自己，配合多轮迭代循环实现精准定位；  
> 第三，更深层是 Anthropic 信任 LLM 的能力，押注模型会越来越强，所以选择「不替模型做决定」的设计哲学。

这道题表面是问技术选型，实际上问的是「你对 Agent 设计哲学的理解」。

Anthropic 这套思路其实在告诉我们一件事：**Agent 不是带工具的聊天机器人，而是会自己做决策的执行体**。工程师的职责是给它一套好用的工具，而不是替它做决策。

<figure>

</figure>

如果你觉得这篇文章有收获，记得点个赞、转发给身边做 Agent 的朋友。

我们下篇见。

## Claude Code 记忆机制图解：为什么不用向量数据库？

> Source: https://xiaolinnote.com/claudecode/source/cc_memory.html

大家好，我是小林。

我在公众号已经陆陆续续写过好几篇<a href="https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzUxODAzNDg4NQ==&amp;action=getalbum&amp;album_id=4404340926102421504#wechat_redirect" target="_blank" rel="noopener noreferrer">图解 Claude Code 原理</a>的文章，基本每篇都是万字长文，配上几十张图解。

<figure>

</figure>

说实话，开写第一篇之前我心里是打鼓的。这种硬核到要啃源码的技术文，真的会有人愿意读吗？

结果挺出乎我意料。这么硬核的图解原理文，竟然有好几篇阅读量冲到了 5w+，最高的一篇甚至直逼10w+了。

<figure>

</figure>

不是，林友们，你们一个个都这么硬核的吗？

既然大家爱看，那就接着写。

最近陆续收到不少读者留言，好几位读者点名想看一篇深度讲 **Claude Code 记忆机制的文章**。理由出奇地一致：面试被问太多次了。

这事我太能理解了。

agent 的记忆机制，如今已经是个不折不扣的面试热点。

只要你简历上挂着一个 agent 项目，面试官大概率会追着问一句：「你这个 agent 的记忆机制，到底是怎么做的？」

<figure>

</figure>

而大多数人能端出来的，往往只有一个标准答案：「上向量数据库，把对话存成 embedding，每次新会话做相似度检索。」这答案不能算错，但只要你知道 Claude Code 偏偏不这么做，就会发现它平平无奇。

那这一篇，我们就把 Claude Code 的记忆机制，从源码层面翻个底朝天。

读完你会撞见一件挺反直觉的事：它走的路子跟业界主流的「向量检索」完全不一样，土到没用一个向量数据库，但用起来反而比向量检索好用得多。

照例，开写之前先把几个问题摆出来，这篇文章就是带你一个一个把它们想明白：

- LLM 明明是「无状态」的，那它跟你聊到第 50 轮还像记得你，记忆到底存在哪？
- 滑动窗口、对话摘要、向量检索这些主流方案，听上去都挺合理，到底差在哪？
- Claude Code 不上向量数据库，那它究竟拿什么来存记忆？
- 为什么一个 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 不够用，非得拆成六个层级？
- Claude Code 怎么做到让 agent「自己学、自己写、自己用」记忆这一整个闭环？
- 这套设计里，到底有哪几条原则能直接抄进你自己的 agent 项目？
- 面试官再问起记忆机制，怎么答才能让对方眼前一亮？

七个问题逐个想透，这道面试题你就稳了。

------------------------------------------------------------------------

### 一、先聊聊「LLM 其实没记忆」

聊 Claude Code 怎么做之前，我们先退一步，把「LLM 没记忆」这件事彻底说清楚。如果你已经很熟悉，可以直接跳到第二节。

#### LLM 的「金鱼记忆」是怎么回事

我们先做一个小实验。

你跟 ChatGPT 聊到第 50 轮的时候，它好像还记得你前面说过的话。

请你停一下，**先猜一下**：这个「记得」，到底是怎么发生的？

是模型本身把你说的话存进脑子里了？还是别的什么机制？

……

答案可能跟你想的不太一样。**LLM 本身根本「记不住」任何东西**，它是彻头彻尾**无状态**的。

每次你按下回车，对它来说都是「从头看一遍」：把系统提示词（system prompt）、所有历史对话、当前问题，全部塞进去，然后输出一个回复。

<figure>

</figure>

你以为它记得，其实是因为你的客户端（比如 ChatGPT 的网页前端）**偷偷把历史消息又一起发了过去**。所谓「上下文窗口」，就是塞这些消息的最大长度。

打个比方，LLM 就像一只金鱼，记忆只有 7 秒。你跟它说完话，它转个身就忘。如果你想让它记得，得每次开口前先把过往对话再念一遍。

聊天场景下，这套办法还能凑合。毕竟聊天就是「你一句我一句」，每轮也就几百个 token（token 你可以粗略理解成模型眼里的「字」，一个汉字大概算一到两个 token），把历史全念一遍也撑得住。但放到 agent 上，就完全不够用了。

#### agent 真正缺的是哪种记忆

agent 跟聊天最大的不同，是它在**长跑**。

调工具、读文件、调 API、再调工具、再读文件……跑着跑着上下文就爆窗口了。你不能指望它「每次都把昨天的对话全念一遍」，那点窗口光塞历史都不够。

更关键的是，agent 想记住的东西，跟「历史对话」其实不是一回事。

<figure>

</figure>

你昨天可能说过一句「我是十年 Go 后端，第一次接触 React」。今天打开新会话问它一个前端问题，你希望它记住的是「这个用户后端经验丰富但前端是新手」这个**事实本身**，而不是「你昨天说过那句话」。这两者天差地别。

再举个例子。你跟 agent 强调过「测试别用 mock，要打真实数据库」。下次写测试你希望它直接照做，**而不是你重新强调一遍**。这种「规则类」的记忆，跟「对话历史」更是完全两码事。

所以你看，agent 真正想记住的，大概是这几类东西：

- 用户画像：你是谁、擅长什么、知识水平如何
- 行为偏好：你不喜欢什么，喜欢什么
- 项目动态：当前项目要干啥、有什么截止日期
- 外部指针：去哪查什么信息

光靠把对话历史存起来，再搞个 RAG 检索（RAG 即「检索增强生成」，先去资料库里查到相关内容、再拼进上下文给模型参考），**根本解决不了上面这些问题**。

<figure>

</figure>

我觉得一个更贴切的比喻，是把 LLM 想象成一个**得了失忆症的实习生**。

<figure>

</figure>

这个实习生很聪明，写代码、查资料、做调研样样都行。但他每天早上一来上班，**啥都不记得**。你必须在他工位上贴满便签：「你叫某某某」「这是你正在做的项目」「老板不喜欢 PPT 用斜体」……否则他每天都从零开始，干一天还干不出昨天的进度。

那「记忆机制」要解决的核心问题，就变成了：**这些便签贴在哪、谁来贴、什么时候撕掉**。

在盘点之前，还得再说清一件事，免得后面看糊涂：记忆其实分两种。

一种是**短期记忆**，说白了就是上下文窗口本身，装着当前这轮对话；窗口快满了，就把旧消息压缩（compaction）一下腾地方。前面讲的「把历史重新念一遍」，管的就是它。另一种是**长期记忆**，持久化存到磁盘、能跨会话活下来；前面说 agent 真正缺的那四类东西（用户画像、偏好、项目动态、外部指针），全属于这一类。

这篇文章重点拆的 Claude Code 记忆机制，是**长期记忆**这一半。至于短期那套怎么压缩上下文，我之前专门写过一篇（<a href="https://mp.weixin.qq.com/s/NBR6dRr3iO7KCTEChk8QUg" target="_blank" rel="noopener noreferrer">点这里查看</a>），这篇就不展开了。

记好这条坐标轴。业界给 agent 做记忆这件事，实打实研究了十几年，方案五花八门，可惜大部分都不太够看。下面就来盘点一下，你会看到有的方案在管短期、有的在管长期，但没一个真正够用。

------------------------------------------------------------------------

### 二、业界主流的记忆方案为什么不够看？

讲 Claude Code 之前，**我想先让你做一件事**。

假设老板现在拍着桌子让你给一个 agent 加记忆机制，**你会怎么设计**？闭上眼想 30 秒。

我猜你脑子里大概会冒出这几个思路：

- 把最近几轮对话直接存下来？
- 太多了就用 LLM 总结一下？
- 上向量数据库做相似度检索？
- 学操作系统，分热的常驻、冷的归档？

<figure>

</figure>

很正常，你能想到的，业界**全部都试过**。GitHub 上扒一扒开源 agent 框架，记忆机制大概就这四类。

我们一个一个过一遍，过完你会发现一件挺有意思的事：**这些「看上去都很合理」的方案，真用起来全都不太够**。

#### 方案一：滑动窗口 Memory

最直观、最暴力的一种。

原理是把最近 N 轮对话原样保留，超过 N 轮的就丢掉。LangChain 早期那个 buffer window memory 就是这思路（在新版本里这套已经被官方标记弃用了，但思路依然散落在很多框架里）。

<figure>

</figure>

听上去合理对吧？反正窗口有限，丢就丢呗。

但有个问题，**你丢的不一定是没用的**。

举个例子。用户在第 1 轮说「我是后端工程师」，然后跑了 50 轮各种问答，第 51 轮问「这段 useState 怎么用」。滑动窗口一砍，「我是后端工程师」这句早就不在窗口里了。agent 给你按零基础前端讲，你还得再花两轮告诉它「我有十年开发经验」。

**关键信息和无关信息混在一起被丢**，是滑动窗口的硬伤。

#### 方案二：对话摘要 Memory

那再聪明点。不直接砍，而是定期把旧对话**用 LLM 总结一下**，把摘要塞回上下文。对应 LangChain 早期的 summary memory。

<figure>

</figure>

这样原文丢了，但「精华」保留下来了。

听起来还不错？我们看个真实场景。

假设用户在 20 轮前说过一句「我们的 API gateway 用的是 Kong，不是 nginx」。LLM 做摘要的时候觉得这句不太重要，就压成了「讨论了一些技术栈细节」。等下一次你问「我们的 API gateway 怎么排查 502」，agent 完全不知道你用的是 Kong，按 nginx 给你答了一通。

**重要的细节被压糊**，这是摘要 Memory 的硬伤。而且 LLM 做摘要本身要耗 token、有延迟，每隔几轮就摘一次，成本不低。

#### 方案三：向量检索 Memory

这是目前最热的方案。Mem0、Letta（原名 MemGPT）、Zep，甚至大部分自己搓的 agent 都是这思路。

<figure>

</figure>

先说清楚 embedding 是什么。你可以把它理解成，给每段文字在一个高维空间里标一个坐标，意思相近的两段话，坐标也会靠得近。

向量检索就是基于这个坐标做文章。它把每条对话或每条记忆**转成 embedding 向量**（也就是上面说的那个坐标），存进向量数据库。每次新对话来，先把用户当前的问题（query）也转成向量，然后跟数据库里所有记忆比一比坐标远近，召回最相似的前 K 条（业界叫 top-K）塞回上下文。

Mem0 是这条路上最响的选手，按他们 2025 年 4 月那篇论文（arxiv 2504.19413）的数据，在 LoCoMo 这个长对话记忆基准上能打到 91.6 分，落地框架也覆盖了一大票。

听上去这方案是不是无懈可击？我给你泼盆冷水。

**第一个问题，相似不等于相关**。你问「这段代码有没有 bug」，向量检索可能把过去所有讨论 bug 的对话都召回来，但其中只有一两条跟当前代码真正相关，剩下的全是噪音。模型一被噪音淹没，就开始胡言乱语。

**第二个问题，召回不稳定**。embedding 模型换一个，召回结果差别巨大。你今天用 OpenAI 的 embedding 表现好好的，明天换成开源的，记忆系统可能从「能用」变「不能用」。

**第三个问题，维护成本高**。要部署向量数据库（pinecone、qdrant、milvus 任选一个），要选 embedding 模型，要管 chunk 大小，要管索引更新，要管冲突合并……上线一套向量记忆系统，工程量比写 agent 主流程还大。

**第四个问题最致命，用户没法看**。你存进向量数据库的记忆是一堆 768 维浮点数，**人脑根本读不懂**。哪天 agent 给你召回了一条错的记忆，你想去看看是哪条记忆引起的？对不起，先把向量反查回原文，还要 debug 一通索引。

<figure>

</figure>

#### 方案四：分层存储 Memory

MemGPT（现在叫 Letta）那一派的方案。

原理是把记忆分成几层：core memory（常驻上下文）、recall memory（最近对话可召回）、archival memory（远期归档）。LLM 自己当操作系统，**用工具调用主动在不同层之间搬数据**。

<figure>

</figure>

这套设计学术上确实漂亮，但工程上落地的反馈是「概念太多、迁移成本太大」。让 agent 自己管理三层记忆，意味着每个 prompt 都要训练它什么时候搬、搬哪条，复杂度直接翻倍。

而且分层归分层，**搬数据的依据本质上还是「相关性匹配」**，依然要靠 embedding 召回，前面那几个硬伤一个都没躲过去。

#### 这四类方案的共同病根

你把四个方案摆一起，会发现它们其实有几个共同的硬伤。

**第一，自由文本无约束**。存什么、不存什么没规则，结果记忆库迅速膨胀成垃圾堆。

**第二，不区分类型**。「用户画像」「项目动态」「外部指针」全部一锅炖、用同一种方式检索，最后哪个都查不准。

**第三，没有老化机制**。一条记忆存进去就是永久的。今天你跟 agent 说「我们项目用 Kong」，半年后换成了 nginx，旧记忆还在告诉它你用 Kong。这种「权威的错误」比没记忆还糟糕。

**第四，重检索、轻写入**。所有方案都把精力花在「怎么查到」，但「该不该存」「存什么」这一步基本是放任的，导致垃圾进、垃圾出。

<figure>

</figure>

带着这四个病根，我们来看 Claude Code 是怎么治的。

------------------------------------------------------------------------

### 三、Claude Code 的两层记忆架构鸟瞰

那 Claude Code 是怎么避开这些坑的？

**你可能会期待一个特别炫的方案**，比如自研一个混合存储、训一个专门的记忆模型、上一套分布式索引。

但答案恰恰相反。Claude Code 走了一条\*\*「土到反直觉」\*\*的路。

它没用向量数据库，没用 embedding，没用任何复杂的存储引擎。它用的是**磁盘上的 markdown 文件**。

是不是想笑？markdown 文件能管好记忆？

别急。等你看完整套机制，你会明白为什么这个看着「土」的方案，**反而把向量检索那一套比了下去**。

讲细节之前，先给你一张地图，否则一头扎进源码容易迷路。

Claude Code 的记忆机制其实是**两条独立的线**，并行工作：

<figure>

</figure>

**静态层是 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 体系**，本质是「声明式指令」。你写好放那里，agent 启动时全量加载。这一层解决的是「我们怎么协作」「这个项目要遵守什么规则」这种**确定性的事情**。

**动态层是自动记忆系统**，本质是「学习式偏好」。agent 在跟你互动的过程中，把它认为「值得记下来的事」自动写成记忆文件存到磁盘上，下次对话再按需检索。这一层解决的是「我从跟你的互动中学到了什么」这种**不确定的事情**。

打个比方，**静态层就像你工位上的「公司员工手册」**，每个新员工入职都得看一遍；**动态层就像你工位旁那本「自己的工作笔记」**，写的是「老板不喜欢 PPT 用斜体」「张三的需求经常变」这种你慢慢摸索出来的东西。

<figure>

</figure>

两层一起用，才是 Claude Code 记忆机制的完整答案。

接下来两节我们各自展开，先看静态层。

------------------------------------------------------------------------

四、静态层：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md 的六个层级</a>

很多人对 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 的印象停留在「项目根目录放一个 md 文件，写点项目说明」。但你打开 Claude Code 源码就会发现，<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 这套体系**远比你想的复杂**。

为什么不能只有一个 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>

我们先想一个问题，如果只有一个 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>，会发生什么？

你很快会发现，想往里塞的「规则」根本不是同一个来源。公司级的强制策略（比如「禁止 commit secrets」）得全员生效、谁也改不得；你个人的习惯（比如 commit message 用中文）希望跨所有项目通用；项目自己的规则要签入 git、给团队共享。

还有几类更微妙的。本地调试用的约定不想签入 git，只想在自己机器上读到；团队一起摸索出来的经验，希望同步给所有成员；以及 Claude Code 自己从对话里学到的你的偏好，也总得有个地方落下来。

这六种来源，可见范围不同、谁能改也不同，硬塞进一个 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 只会要么打架要么混乱。所以 Claude Code 索性把它拆成了**六个层级**。

#### 六个层级各管一摊

按加载顺序从低到高，长这样：

<figure>

</figure>

每一层的定位都很明确：

- **Managed**：放在系统级路径，只有管理员能改。公司级强制策略走这层
- **User**：放在用户家目录下，你自己的全局偏好走这层，无论在哪个项目都生效
- **Project**：项目根目录的 `CLAUDE.md` 或 `.claude/CLAUDE.md`，项目层规则，签入 git 让团队共享
- **Local**：项目根目录的 `CLAUDE.local.md`，默认不签入 git，你自己用
- **Auto**（源码标识符 `AutoMem`）：项目级的自动记忆目录，Claude Code 自动写入的偏好，下一节单独展开
- **Team**（源码标识符 `TeamMem`）：在 Auto 目录下再开一个 `team/` 子目录，团队共享的 AI 学到的偏好（需要 feature flag 开启）

<figure>

</figure>

你可能注意到了，后两层 Auto 和 Team 存的其实是 `MEMORY.md` 文件，不是 `CLAUDE.md`。这里把六层统称「层级」，是因为它们同属一套加载体系、最后都拼进 system prompt，不必纠结文件名的差别。

六层之间是**叠加关系**不是覆盖关系。Claude Code 启动时把它们**全部拼接**进 system prompt，让模型一起看到。

<figure>

</figure>

@include：让 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md 互相引用</a>

光分层还不够，还有个实际问题。

假设你们公司有一份「通用安全规范」，每个项目都得遵守，难道每个项目的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 都把它**复制一遍**吗？

太蠢了。

Claude Code 给的方案是 `@include` 指令。

你在 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里写一行 `@~/company/security-rules.md`，加载的时候它就**自动把那个文件的内容读进来拼上**，思路跟 C 语言的 `#include` 一模一样。

当然，背后还得防循环引用、防路径遍历这些工程坑，这里就不展开了。

#### 条件规则：编辑 .tsx 才加载前端规范

层级和 @include 都讲完了，再看一个**挺有意思的设计**。

如果你给项目写了一份很长的前端规范：React Hooks 用法、CSS 命名规则、Tailwind 配置原则……几百行。

但你**只在编辑前端代码的时候才需要这套规范**，编辑后端代码也加载，纯粹浪费 token。

Claude Code 在 `.claude/rules/` 这个目录下支持**条件规则**。每条规则是一个独立的 md 文件，文件 frontmatter（md 文件开头用 `---` 框起来的那段元数据）里可以写一个 `paths` 字段，用 glob 通配符（就是 `*.tsx` 这种写法）匹配：

```
---
name: 前端规范
description: React + Tailwind 项目规范
paths: ["**/*.tsx", "**/*.jsx"]
---

# 前端规范
...（规则正文）
```

加载的时候，Claude Code 会**比对当前编辑的文件路径**，只有匹配上的规则才会被拼进 system prompt，匹配不上的就跳过。

这个设计的妙处在于，它让 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> **不再是「一次性全塞」**，而是「按需注入」。一个大项目可以有几十条规则，每条只在它真正需要的时候才占用 token，整体上下文窗口就省下来了。

<figure>

</figure>

#### 截断双保险：防长行索引炸弹

最后聊一个跟安全有关的小设计，挺巧的。

<a href="http://MEMORY.md" target="_blank" rel="noopener noreferrer">MEMORY.md</a> 索引文件（下一节会讲是什么）要塞进 system prompt，所以必须有大小上限，否则索引膨胀就把上下文撑爆了。

按常规思路，限制行数就行了，比如「最多 200 行」。但 Claude Code 团队发现，**有时候一条索引就是一行超长字符串**。代码里有句注释：

> p100 observed: 197KB under 200 lines

什么意思？他们观察到的极端情况是，一个 <a href="http://MEMORY.md" target="_blank" rel="noopener noreferrer">MEMORY.md</a> 只有不到 200 行，但加起来 197KB。光看行数完全察觉不到，但塞进 system prompt 就是个灾难。

所以 Claude Code 用了**双保险**：

```
export const MAX_ENTRYPOINT_LINES = 200
export const MAX_ENTRYPOINT_BYTES = 25_000
```

两个限制**任意一个先触发**，就截断。截断的时候还会主动追加一条警告告诉模型「这个索引被截过了，部分内容没加载进来」，让模型自己有数。

这种「行数 + 字节」的双截断，本质上是在防御「**长行索引炸弹**」这种特殊形态的上下文溢出。在做 agent 项目的时候，凡是用户可控的文本要进 system prompt，都建议参考这种双保险设计。

<figure>

</figure>

静态层讲完了，但**真正有意思的是动态层**，下一节是文章的重头戏。

------------------------------------------------------------------------

### 五、动态层：自动记忆系统的完整闭环

如果说静态层是「框架」，那动态层就是 Claude Code 的真正灵魂。

#### 为什么还需要动态记忆

<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 这套体系再完善，也有一个根本局限：**得你主动写**。

你昨天跟 Claude 抱怨「这个 mock 测试又骗过 CI 了」，今天换会话再写测试，它依然得你重新强调「别用 mock」。<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 又不会自己长出新内容。

理想的状态应该是：**Claude 在跟你聊天的过程中，自动把它学到的东西记下来**，下次你不用再说一遍。

这就是动态层要解决的问题，让 agent **自己学、自己写、自己用**。

<figure>

</figure>

听起来像 RAG 那一套对吧？但 Claude Code 做得很不一样。它没有用向量数据库、没有用 embedding，而是用了一套**结构化的文件系统 + LLM 选择**。

我们一步步拆。

#### 四种类型为什么这么分

要让 agent 自己写记忆，第一个问题就是：**让它写什么？**

你可能会想，「记下来对我有用的就行了呗」。

但「对我有用的」是个模糊到没法落地的标准。我们设想一下，**如果你放任 agent 自由发挥**，它一天能写出哪些东西？

它可能会记「用户今天问了 React 的问题」（流水账没用）、「这个函数用了 `useMemo`」（代码里 grep 一下就有）、「上次帮用户改了 5 行代码」（活动日志没用）……

三天下来，记忆库就成了一个**啥都有、啥都查不准**的垃圾桶。

那怎么办？

Claude Code 的答案干脆利落：**只允许四种类型，其他一律不许写**。

```
export const MEMORY_TYPES = [
  'user',
  'feedback',
  'project',
  'reference',
] as const
```

短短四个词，背后的设计哲学非常重，我们来拆。

<figure>

</figure>

**user 用户画像**，记的是「你是谁」。比如「这位用户写了十年 Go，刚接触 React」。这类记忆让 agent 的回答**因人而异**，对老兵和新手用完全不同的解释方式。

**feedback 行为偏好**，记的是「你不喜欢什么 / 你确认有效的做法」。比如「这位用户不希望每次回复后做总结，diff 就够了」。这类记忆是 Claude Code 最看重的一种，因为它直接决定了 agent **下一次行为的对不对**。

`feedback` 类型有一个**强制结构**，正文必须包含三段：

```
规则本身

**Why:** 用户为什么这么要求（往往是踩过的坑）
**How to apply:** 什么情况下生效
```

<figure>

</figure>

为什么这么严？因为只记规则不记原因，遇到边界情况就抓瞎。比如「不要用 mock 测试」，单纯一句话不够，加上「上季度 mock 测试通过但 prod 迁移挂了」这个 Why，agent 在边界情况下就能**自己判断这个 case 该不该破例**。

**project 项目动态**，记的是「项目正在发生什么」。比如「移动端 3 月 5 号开始合并冻结」。

这类记忆其实跟 feedback 一样吃**同一套强制结构**：开头是事实/决定，然后 Why（这事为什么发生，是哪条约束或截止日期推动的）、How to apply（这件事应该怎么影响 agent 的建议）。Claude Code 把这套约束同时套在 feedback 和 project 上，是因为这两类记忆最容易过期、最需要让 agent 自己判断「现在还该不该信」。

除此之外，project 还有个怪要求：**必须把相对日期转成绝对日期**。用户说「周四之前冻结」，存进去要变成「2026-03-05 之前冻结」。原因很简单，「周四」过几天就过期了，「2026-03-05」永远准确。

**reference 外部指针**，记的是「去哪查什么」。比如「pipeline bug 都在 Linear 的 INGEST 项目里追踪」。agent 不需要知道外部系统的具体内容，**只需要知道去哪里找**。

四种类型限定死了，记忆系统的「信息形态」就有了纪律。每次想存一条记忆，agent 必须先想清楚「这属于哪一类」，而不是一股脑往里塞。

#### 该存什么 vs 不该存什么

跟「该存什么」同样重要的，是「**不该存什么**」。Claude Code 在系统提示词里明确列了一份禁令清单。

<figure>

</figure>

不该存的有这些：

- 代码模式、架构、文件路径、项目结构（用 `grep` / `CLAUDE.md` 就能得到，存了反而和实际状态不一致）
- Git 历史和最近改动（`git log` / `git blame` 是权威，记忆只会落后于真相）
- 调试方案和修复方法（fix 已经在代码里，commit 已经记录了上下文）
- <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里已经写过的内容
- 临时任务状态和当前对话上下文

为什么这条「不该存」清单这么重要？

因为**代码是「活的」，记忆是「死的」**。代码随时在变，但一条记忆存进去就定格了。如果记忆说「`AuthService` 在某个具体路径第 42 行」，但代码已经重构了，这条记忆就变成了一个\*\*「权威的错误」\*\*，比没有记忆还糟糕。

<figure>

</figure>

所以记忆系统的纪律是：**只记代码推不出来的东西**。这个原则贯穿了整套设计。

#### 存储设计：单文件 + 索引

类型定下来了，下一个问题来了。

假设你已经积攒了 50 条记忆，**它们放哪里，怎么让模型知道**？

最直觉的做法，全部塞进 system prompt。但你算一下，50 条每条 200 字，就是 1 万字、几千 token，system prompt **直接被记忆撑爆**。

那反过来呢，完全不塞？那模型根本不知道有这些记忆可用，写了等于白写。

两难。

我们先看 Claude Code 怎么平衡。每条记忆是一个独立的 `.md` 文件，文件头有一段 YAML frontmatter：

```
---
name: 不要用 mock 数据库
description: 集成测试必须连真实数据库
type: feedback
---

集成测试必须连真实数据库，不要用 mock。

**Why:** 上季度 mock 测试通过了但 prod 迁移挂了
**How to apply:** 所有标了「集成测试」的 case 都适用
```

frontmatter 里的三个字段是「身份证」：`name` 是标识，`description` 是「这条记忆是啥」（**这一句话非常重要，决定了它能不能被检索到**），`type` 是四种类型之一。

所有记忆文件放在一个目录下，目录里还有一个**特殊文件 `MEMORY.md`**，是所有记忆的索引清单。

<figure>

</figure>

这里有一个**很关键的设计**，回到刚才那个两难问题，Claude Code 的答案是：**只塞「目录」，不塞「正文」**。

```
MEMORY.md 索引 → 始终加载进 system prompt
独立记忆文件 → 按需加载
```

是不是有点像**翻一本厚厚的工具书**？你不需要把整本书背下来塞脑子里，但**至少得知道目录里都有哪几章**，需要某章再翻到那一页。

落到 Claude Code 上就是：

- agent 看到索引就知道**有哪些记忆可选**（看 name 和 description 就够）
- 真正需要某条的时候，再把完整内容加载进来

这个设计还顺带解决了上一节提到的「截断双保险」问题：索引始终常驻，所以索引的大小必须严格控制，行 + 字节双限制就是这么来的。

<figure>

</figure>

#### 写入：Extract Memories 代理

光设计好类型和存储格式还不够。**记忆是怎么写进去的？**

最朴素的做法你应该想到了，**让主对话自己写**呗。每轮结束让模型「想想这次有啥该记的」，然后自己写文件。

听上去合理。但你再仔细想想，会不会有坑？

至少**两个坑很明显**：

第一个，**模型分心**。主任务都做不好，还要分一脑子去想「这条要不要记、属于哪类、放哪里」，回复质量会打折。

第二个，**token 浪费**。每轮都得在 system prompt 里塞一段「记一下偏好啊」的指令，每次都进、每次都算钱，全是冗余。

那 Claude Code 怎么躲开这两个坑？

它的方案是：**每轮对话结束后，后台单独跑一个代理来抽取记忆**。

这个代理叫 `extractMemories`，触发时机是每轮 query loop 完整结束（也就是模型给出最终回复、没有任何 tool call 了），通过一个 `stopHook` 钩子触发。

<figure>

</figure>

源码注释把它的精髓讲得很清楚：

```
// Uses the forked agent pattern (runForkedAgent) — a perfect fork of the main
// conversation that shares the parent's prompt cache.
```

这里有个**很妙的小心思**。Extract Memories 代理不是从零启动一个新对话，而是「**完美 fork（复刻一份）主对话**」，复用主对话的 prompt cache（把已经算过的 system prompt 缓存下来，下次不用重算）。

这意味着什么？意味着它不用重新加载几千 token 的 system prompt，**只需要看着对话历史，决定「这次有没有值得记的东西」就行**。整个抽取过程多花的钱很少。

<figure>

</figure>

抽取代理的逻辑大致是：

- 扫一遍这一轮对话里用户的反馈、纠正、信息
- 跟现有记忆比对，看有没有重复
- 如果有新的值得记的内容，按四种类型分类，写一个新文件

注意「跟现有记忆比对」这步。代理会主动检测 `hasMemoryWritesSince`，**过滤掉它刚刚写过的内容**，避免对同一件事反复写记忆。

还记得第二章那四个病根吗？其中一个是「重检索、轻写入」，所有方案都在琢磨「怎么查到」，却放任「该不该存、存什么」。Claude Code 专门派一个后台代理来干「写入」这件事，本身就是在治这个病根。

#### 检索：用 Sonnet 选 top-5

存储讲完了，下一个问题来了。

下次对话来的时候，假设你有 100 条记忆，**怎么挑出最相关的 5 条塞给主模型**？

你脑子里第一反应，大概是**向量检索**对吧？给每条记忆做个 embedding，给 query 也做个 embedding，算相似度、召回 top-5，干脆利落。

合情合理。**但 Claude Code 偏偏不这么干**。

它的做法反直觉到你可能根本没往这上面想过，**让一个小模型来挑**。

具体怎么挑？逻辑写在 `findRelevantMemories` 里。

**第一步，扫描所有记忆文件的「头部信息」**。它只读每个文件**前 30 行**，足够提取 frontmatter，不会读取记忆完整内容。这样即使有 200 个记忆文件，扫描开销也很小。

**第二步，把所有记忆的「标题清单」拼成一段文本**，发给 Sonnet：

```
Query: 用户当前的问题

Available memories:
- user_role.md — 后端工程师，新接触 React
- feedback_no_mock.md — 测试不要用 mock
- project_freeze.md — 3 月 5 号开始合并冻结
...
```

**第三步，让 Sonnet 用 JSON schema 返回 top-5 文件名**。系统提示词写得非常严苛：

```
Only include memories that you are certain will be
helpful based on their name and description.
Be selective and discerning.
```

意思就是「不确定的就别选」。宁可少选，不可错选。

为什么是 Sonnet 不是 Haiku？Haiku 更便宜啊。

我的理解是：**Sonnet 比 Haiku 准很多**，记忆相关性判错的代价远大于多花的那点钱。一旦把错的记忆塞进上下文，整个回复都会被污染。这道选择题的容错率非常低，所以宁可贵一点也要选准。

<figure>

</figure>

还有几个细节值得讲。

- **`alreadySurfaced` 过滤**：上一轮对话已经露过脸的记忆，这次直接排除。让 Sonnet 把 5 个名额花在新候选上，而不是反复挑同样的。

- **`recentTools` 过滤**：最近用过的工具的「用法参考文档」不要选。因为 agent 当前对话里已经在实际使用这些工具了，再塞一份文档纯粹是噪音。**但工具的「警告、坑点、已知问题」记忆要保留**，「正在用」的时候恰好是这些警告最该出现的时候。

这两个细节看着小，但反映的是一个非常成熟的产品思维：**检索不只是匹配，还要看「当前上下文已经有了什么」**。

<figure>

</figure>

#### 注入：system-reminder 包裹 + 老化警告

记忆被选出来之后，怎么塞进对话？

直接拼到 user message 里？不行，模型可能把它当成「用户刚说的话」。

Claude Code 的做法是用 `<system-reminder>` 标签包裹后注入：

```
<system-reminder>
This memory was saved 5 days ago. Verify it's still accurate before acting on it.

[记忆内容]
</system-reminder>
```

<figure>

</figure>

注意上面那句「This memory was saved 5 days ago」。这就是**记忆老化**的设计：

```
今天 / 昨天 → 不警告
2 天以前 → 主动加警告，提醒模型「这是过去的快照」
```

为什么 2 天就警告？

因为软件开发节奏快，**两天前的「项目正在做 X」可能今天就已经改了**。让模型见到记忆的时候，自动带着「这是历史，不是现状」的心态去用。

这一条警告解决了向量检索方案最致命的问题，「**权威的错误**」。当一条记忆过时了，agent 不是闭着眼睛信，而是会主动去验证（比如 grep 一下、读一下当前文件状态），发现冲突就更新或忽略它。

Claude Code 还有专门一段「Before recommending from memory」的提示词，明确告诉模型：

- 如果记忆里写了文件路径，先检查文件是否存在
- 如果记忆里写了函数名或 flag，先 grep 一下
- 如果用户要照你的建议动手了，先验证再说

「记忆说 X 存在」不等于「X 现在存在」，这句话写得非常重。

#### 完整闭环图

到这里，自动记忆系统的完整闭环就讲完了。我们把它串成一张图：

<figure>

</figure>

抽取在右边、检索在左边，一写一读，记忆系统转起来。

而最妙的是，整套机制没有一行 embedding、没有一个向量数据库，**全是结构化文件 + 一个小模型当选择器**。简单，但**比向量检索好用得多**。

------------------------------------------------------------------------

### 六、几个值得借鉴的设计选择

读到这里，**你不妨再停一下**。

把上面 Claude Code 的整套机制盖住，闭眼回想一下：**你能从里头抽出几条「可以抄到自己 agent 项目」的原则吗**？

试着自己列一列，再往下看。

我自己看完源码，抽出了四条。你可以对照着看，跟你想的一不一样。

#### 第一条：结构化优于自由文本

Claude Code 不让记忆是「自由文本流」，而是强制四种类型 + 强制 frontmatter。

为什么这么重要？因为**自由文本无约束 = 垃圾堆**。三个月后回头看自由文本的记忆库，你会发现里面什么都有，什么都查不准。

强制类型的本质是**逼 agent 在写之前先做一次「分类决策」**：这条信息属于「用户画像」还是「行为偏好」？是「项目动态」还是「外部指针」？想清楚才能写，写下来的东西就有用。

<figure>

</figure>

迁移到自己的项目，这条原则可以变成：**给记忆定一个 schema，强制每条都填齐**。哪怕只是 4 个字段，比啥都不要强一百倍。

#### 第二条：索引常驻 + 内容按需

这是我最喜欢的一个设计。

「全塞进 system prompt」会爆窗口，「完全不塞」agent 又不知道有什么。中间这道平衡，Claude Code 用一个 <a href="http://MEMORY.md" target="_blank" rel="noopener noreferrer">MEMORY.md</a> 索引解决了：

- 索引始终在 system prompt 里（让 agent 知道**有什么**）
- 完整内容按需加载（不浪费 token）

这个思路不只适用于记忆系统。**任何「内容总量大但只有少数需要展开」的场景**都可以套：长 RAG 文档、知识库、tool 列表、历史 PR……都可以做成索引 + 按需展开。

<figure>

</figure>

#### 第三条：廉价模型做选择题

向量检索流派把检索当成「数学题」（相似度计算）来做。Claude Code 把检索当成「选择题」来做，让小模型挑。

这是一个**非常重要的思维转换**。

向量相似度是个**数值**，它告诉你「这条记忆跟 query 有 0.87 的相似度」。但 0.87 是相关吗？模型不知道，得靠阈值。阈值难定，错召回了你都不知道是哪一步出错了。

让 Sonnet 当选择器，它给你的是**自然语言判断**：「这条该选，那条不该选」。判错了也是模型可以解释的错，调起来比调阈值容易得多。

成本上，Sonnet 一次选择几百 token，比维护一个向量数据库的人力 + 算力便宜得多。

迁移到自己的项目：**只要候选集合不大（几百以内），用小模型选择 \> 向量检索**。

<figure>

</figure>

#### 第四条：时间感知 + 主动验证

最后一条，是给「权威的错误」上锁。

记忆系统最大的风险，不是「召回不准」，而是「召回了一条**已经过时的记忆**，agent 闭着眼睛信」。这种错误特别隐蔽，因为模型把记忆当 ground truth，错也不知道是错。

Claude Code 的两道防线是：

- **时间感知**：2 天前的记忆主动加 stale 警告
- **主动验证**：记忆里写了文件路径 / 函数 / flag，使用前先 grep 一下

这两道防线背后是同一个心法：**记忆不是真理，是历史快照**。模型对待记忆的姿态应该像对待 git log，「这是过去发生的事」，不是「这是现在的状态」。

<figure>

</figure>

------------------------------------------------------------------------

### 七、这道面试题该怎么答？

回到开头那个面试问题：

> 你的 agent 项目记忆机制是怎么做的？

**先别急着往下看**。请你把书合上，自己心里默答一遍。

如果你只能答出「上向量数据库存 embedding 做相似度检索」，那是 60 分答案，跟开头那位被问懵的读者一个水平。

如果你想答 95 分，可以这样组织：

**第一步，先指出 LLM 是无状态的**。记忆机制的本质是「在工位上贴便签」，所以核心问题是「贴在哪、谁来贴、什么时候撕」。

**第二步，指出业界主流方案都有共同短板**。滑动窗口、对话摘要、向量检索、分层存储这四类方案的共同病根，是自由文本无约束、不区分类型、没有老化机制、重检索轻写入。

**第三步，举 Claude Code 作为反例**。它没用向量数据库，而是用了「结构化文件 + LLM 选择」的设计。

展开来说：

- **两层架构**：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 六层级（声明式）+ 自动记忆系统（学习式）
- **四种类型**：user / feedback / project / reference，强制分类决策
- **索引常驻**：<a href="http://MEMORY.md" target="_blank" rel="noopener noreferrer">MEMORY.md</a> 索引始终加载、内容按需加载
- **小模型选择**：用 Sonnet 当选择器选 top-5，过滤已露脸 + 过滤工具文档
- **老化警告**：2 天前的记忆主动加 stale 提醒，逼模型主动验证

**第四步，给出可迁移的设计原则**：结构化优于自由文本、索引常驻 + 内容按需、廉价模型做选择题、时间感知 + 主动验证。

到这里，**面试官不光会觉得你看过 Claude Code 源码，还会觉得你能把这套思路用在自己的 agent 项目里**。

如果你能答到这个程度，下次再被问到记忆机制这道题，大概率能拿下 offer。

<figure>

</figure>

------------------------------------------------------------------------

### 最后

最后说一句，记忆机制这套设计的精髓。

其实跟 Claude Code 的整体哲学高度一致：**不堆复杂度，把已经成熟的简单组件（文件系统 + LLM）组合出比花哨方案更好用的东西**。

如果这篇文章对你有帮助，记得点个赞、在看、转发三连，感谢林友们的支持！

我们下篇见啦。

## Claude Code 多 Agent 图解：SubAgent 实现机制怎么做？

> Source: https://xiaolinnote.com/claudecode/source/cc_multi_agent.html

大家好，我是小林。

最近不少朋友跟我反馈，说 AI Agent 岗的面试越来越多，十有八九都要问 Multi-Agent，什么「多 agent 之间怎么通信」「一个 agent 搞不定的任务怎么拆」「并发 agent 怎么调度」。

所以这篇文章我就想带你从源码视角，把 Claude Code 的多 Agent 机制彻底讲明白，目标是让你看完能同时 get 三个问题：

- 第一，**架构是怎么设计的**，多个 agent 之间怎么隔离、各自跑各自的。

- 第二，**协同机制**是怎么跑起来的，父子 agent 怎么分工，多个 agent 怎么并发。

- 第三，**通信方式**是怎么设计的，agent 之间是直接调函数，还是有别的巧妙设计。

Claude Code 里跟「多 agent」沾边的代码其实有三套不同的机制：**常规 Subagent、Fork Subagent、Coordinator 协调者模式**。

后面我会按由浅入深的顺序，一个个讲清楚。

<figure>

</figure>

------------------------------------------------------------------------

### 一、先搞明白 Multi-Agent 到底是个啥

在扒源码之前，我想先花一点篇幅，把 Multi-Agent 这个词的底层逻辑讲清楚。因为我发现很多人连「为啥要有多 agent」都没想明白，光盯着代码看是看不懂的。

#### 为什么一个 agent 不够用？

我们先回到最朴素的 agent 模型：一个 LLM + 一堆工具 + 一个循环。你给它一个任务，它自己决定调什么工具、调几次，直到做完。这就是经典的 agentic loop。

看起来挺强的是吧？但一到真实项目里，问题就出来了。

想象你让一个 agent 去做这么一件事：「**调研下 React 18 的新特性，然后在我的项目里实现一个 useTransition 的例子，最后帮我把代码评审一遍**」。

这一套下来有三个麻烦：

**第一，上下文会爆炸**。调研阶段要看大量文档和 StackOverflow 链接，实现阶段要读项目代码，评审阶段又要重新读实现。三个阶段的内容全塞到一个 agent 的上下文里，token 蹭蹭往上涨，后面直接塞不下。

**第二，职责混乱**。一个 agent 既当研究员又当程序员又当评审员，它自己都不知道现在是什么角色，容易跑偏。比如调研到一半就开始写代码了，代码写到一半又去查文档。

**第三，没法并发**。一个 agent 一次只能做一件事，它在查文档的时候，项目代码就在那干等着。

<figure>

<figcaption>单 agent 硬扛三件事</figcaption>
</figure>

#### 老板派活的思路

这时候 Multi-Agent 的思路就来了。说白了，就像一个老板带团队：

老板不自己一头扎进代码里，而是把任务拆成几块，派给不同的「专家」。研究员去调研，工程师去写代码，评审员去挑错。老板自己只负责看大方向、收结果、做决策。

这样一来：每个专家的上下文是干净的（只装自己领域的信息）；职责也清楚（研究员就好好查资料别去写代码）；多个专家还能同时开工。

这就是 Multi-Agent 的核心思想：**把一个大任务拆给多个职责清晰的 agent 去做，它们之间通过某种方式通信和协作**。

#### Multi-Agent 的三种常见形态

绕开花哨的术语，Multi-Agent 系统在工业界落地时，一般就三种形态。

<figure>

</figure>

**第一种，父子型**。主 agent 处理整个任务，遇到某个子问题时派一个 subagent 出去搞定，拿结果回来接着干。这是最常见的，Claude Code 里的 Task 工具就是这种。

**第二种，平级协作型**。几个 agent 职责对等，通过共享状态或者消息互相协作。不过这种在工程上比较难落地，状态同步很麻烦。

**第三种，主从型（Coordinator-Worker）**。有一个专门的「协调者 agent」，它自己不干活，只负责派 worker、收结果、做合成。worker 之间互不通信，全靠协调者调度。这种是高并发场景的标配。

Claude Code 源码里，**常规 Subagent** 对应父子型，**Coordinator 模式**对应主从型，**Fork Subagent** 是父子型的一个特殊优化版本（跟 cache 有关，后面讲）。

<figure>

</figure>

#### subagent 在 Claude Code 里到底长啥样？

讲到这儿可能还有朋友有点虚：「subagent 听起来挺抽象，它在 Claude Code 里到底长啥样，看得见吗？」

我举个真实能感知的场景你就懂了。

你跟 Claude Code 说「调研一下这个项目的认证模块」，它自己判断一下：这活得派个「侦察兵」去干，而不是我亲自扎进去。于是它在内部调了一个叫 **Agent 的工具**（对，这个工具的名字就叫 Agent），把任务交给一个叫 **Explore** 的内置 subagent 去跑。

Explore 带着一套精简的工具池（只有读文件、搜代码这些只读工具），带着一份独立的上下文，跑完调研把结果打包回来交给主 agent。主 agent 收到结果后，该改代码改代码、该回答回答。

所以 subagent 不是什么玄学，说白了就是「主 agent 通过一个特定工具派出去的另一个独立 agent 实例」。每一个 subagent 都是一个真实存在的执行单元，有自己的工具池、上下文、生命周期。

<figure>

</figure>

明白了这些，咱们就可以进入 Claude Code 的源码了。

------------------------------------------------------------------------

### 二、Subagent 的隔离机制

在讲通信、讲并发之前，我想先从 Claude Code 多 agent 设计里**最关键的一环**讲起：**隔离机制**。

为什么隔离最关键？你想想，多 agent 系统本质就是「一堆 agent 共处一个进程、共享一个底层运行时」。如果隔离做得不好，一个 subagent 偷偷污染了父 agent 的状态、或者调了不该调的工具，整个系统就会乱成一锅粥。

Claude Code 在 subagent 启动时，把隔离做到了**两个维度**：**工具隔离**（不给子 agent 它不该有的工具）和 **上下文隔离**（不让子 agent 搅乱父 agent 的运行时状态）。咱们一个一个看。

<figure>

</figure>

#### 第一维度：给子 agent 发一个定制工具箱

先说工具隔离。这是 Claude Code 多 agent 设计里**最容易被忽略，但又很重要**的一环。

什么意思呢？主 agent 拥有一大堆工具（读文件、写文件、执行命令、派 subagent、问用户问题等等几十个），但你不能把这堆工具原封不动地丢给 subagent。为啥？

你想想，如果 subagent 也能调派新 subagent 的工具，那它就能派子子 agent，子子 agent 又派子子子 agent，层层嵌套没完没了，token 消耗直接起飞。

再比如主 agent 用来管理任务列表的工具，是给主 agent 的大脑用的，subagent 跟着瞎写会污染主 agent 的待办状态。

所以 Claude Code 给 subagent 发工具的思路是「**按 agent 身份走三道准入门**」：

**第一道门是「所有 subagent 通用黑名单」**。这道门里被禁的工具有几类：

- **能派新 subagent 的工具**：防止子再派孙、孙再派重孙的递归嵌套
- **能主动问用户问题的工具**：子 agent 不该抢主 agent 的对话权，用户是跟主 agent 说话的
- **能切换规划模式的工具**：规划模式是主 agent 用来跟用户对齐方案的，子 agent 没资格切
- **能停止其他任务的工具**：任务管理是主线程的专属权力，子 agent 乱停会天下大乱

**第二道门是「自定义 agent 多套一层黑名单」**。用户自己写的 agent（比如在项目里自己配的那种 Markdown agent）比内置 agent 要再严一点，因为用户写的没经过官方审核，多防一道更安全。

**第三道门反过来，是「后台异步 agent 走白名单」**。这类 agent 是完全后台跑的，没法跟用户交互，所以只准用事先圈定好的一小批工具（读文件、搜代码、执行命令、编辑文件这些）。白名单的哲学是「默认不准用，明确列出来的才能用」，比黑名单更保险。

三道门走下来，每个 subagent 拿到的都是一份**量身定制**的工具池，既够它干活，又不会越权。

<figure>

</figure>

这个机制在源码里其实就是一个过滤函数：

```
// src/tools/AgentTool/agentToolUtils.ts:70
export function filterToolsForAgent({ tools, isBuiltIn, isAsync, permissionMode }): Tools {
  return tools.filter(tool => {
    if (tool.name.startsWith('mcp__')) return true  // MCP 工具全放行
    if (ALL_AGENT_DISALLOWED_TOOLS.has(tool.name)) return false
    if (!isBuiltIn && CUSTOM_AGENT_DISALLOWED_TOOLS.has(tool.name)) return false
    if (isAsync && !ASYNC_AGENT_ALLOWED_TOOLS.has(tool.name)) {
      return false
    }
    return true
  })
}
```

可以看到就是顺着「全局黑名单 → 自定义 agent 加严 → 异步白名单」这三道条件依次判定。最后留下来的，才是这个 subagent 能用的工具。

<figure>

</figure>

这个设计看着简单，其实挺有工程智慧的。我在设计自己的多 agent 系统时，就学到了一条原则：**不要假设所有 agent 都能用所有工具，按 agent 类型做细粒度的权限控制**。

#### 第二维度：搭一个隔离的运行环境

说完工具，再来聊第二维度：**上下文隔离**。这块是 Claude Code 多 agent 设计里**最精髓的一块**，我觉得全篇文章最值得细读的就是这一节。

先说问题。父 agent 跑起来后有一个庞大的**运行时上下文**，里面装着很多东西：已经读过哪些文件、每个文件读到第几行、全局的 UI 状态、中止信号、权限状态、任务注册表等等。

现在轮到你做设计。要派一个 subagent，这份庞大上下文怎么传给它？

你脑子里很可能蹦出两个直觉方案：**A 完全共享**（父那份直接给子用）、或者 **B 完全新建**（给子一份全新空的）。先别看下面，自己想想哪个对？

…

先说 A 不行，举个具体场景你就懂：父 agent 已经读过 file.ts 的前 100 行，子 agent 拿过去接着读到 200 行。这下父 agent 那边「文件读到哪了」的缓存被刷成 200 了，下次它要读这文件就以为自己已经读过 200 行了，直接跳过。**子的一次操作，把父的视图污染了**。

再说 B 也不行：用户按 Ctrl+C 想中止整个任务，主线程把中止信号广播出去，结果子 agent 因为是全新上下文收不到这个信号，对外面发生啥一无所知，自顾自继续跑。**子 agent 跟世界完全脱节了**。

发现了吧，**两个极端都走不通**。那 Claude Code 怎么办？答案是一个很巧妙的折中思路：**不按「整体」决策，而是按「字段」决策。每一项状态单独判断该克隆、该共享、该屏蔽，还是该新建**。

我把 Claude Code 在这件事上的**四个关键决策**挑出来，用大白话讲一遍：

**决策一：「读文件的缓存」要复制一份给子 agent**

这个缓存存的是「这个文件读过没、读到第几行」。如果父子共享，子 agent 读了某个文件，父 agent 会误以为自己也读过，下次跳过不读，数据就错了。所以要复制一份独立的给子 agent，子怎么折腾都不影响父的文件视图。

**决策二：「改全局状态」这件事对子 agent 直接关闭**

全局 UI 状态是主线程用 React 在管的。如果异步 subagent 也能改，就会出现「两边同时改同一份状态、抢起来对不上」的问题，界面就花了。所以 Claude Code 干脆把 subagent 的「写全局状态」这个权力**完全关闭掉**，改成空操作，一了百了。

**决策三：但「注册后台任务」这条通路得保留**

这里有个小细节值得讲。既然子 agent 的写权力关掉了，那它自己起的后台进程（比如在后台跑一条 bash 命令）怎么登记到全局任务表？

Claude Code 专门开了一个**小口子**：其他写全局的口都堵死，唯独「注册/结束后台任务」这条路留着。不然子 agent 起的后台进程就变成「没爹的孤儿进程」，永远在后台跑没人回收。

**决策四：给每个 subagent 发独立 ID、深度代代 +1**

每派一个 subagent，都给它一个独立的 ID，并且在父 agent 的深度基础上 +1。这样系统能随时知道「当前这个 agent 处于嵌套的第几层」。深度超过阈值（比如 5 层）就报警甚至强制停止，防止意外嵌套失控。

这四个决策其实回答了四类问题：**信息怎么传、状态怎么写、通路怎么留、身份怎么追踪**。

<figure>

</figure>

对应到源码里，就是一个叫 `createSubagentContext` 的函数，我把最能说明上面四个决策的部分精简出来：

```
// src/utils/forkedAgent.ts:345
export function createSubagentContext(parentContext, overrides): ToolUseContext {
  return {
    // 决策一：文件读缓存克隆一份
    readFileState: cloneFileStateCache(parentContext.readFileState),
    // 决策二：写全局状态直接设为空操作
    setAppState: () => {},
    // 决策三：但任务注册的通路例外保留
    setAppStateForTasks: parentContext.setAppStateForTasks ?? parentContext.setAppState,
    // 决策四：独立 ID + 深度 +1
    agentId: overrides?.agentId ?? createAgentId(),
    queryTracking: {
      chainId: randomUUID(),
      depth: (parentContext.queryTracking?.depth ?? -1) + 1,
    },
    // ...其他字段略
  }
}
```

你看这几行代码，一一对应上面讲的四个决策：克隆缓存、关掉写权限、保留任务通路、发独立 ID。

看完这块，我的感受是：**所谓上下文隔离，不是一刀切地「全隔离」或者「不隔离」，而是按每个状态的语义单独决策**。这个细腻劲儿，正是 Claude Code 这种工业级产品稳定跑的根基。

<figure>

</figure>

走完「工具隔离」和「上下文隔离」这两道门，一个 subagent 就拿到了干净的工具池 + 干净的运行环境，可以独立跑起来了。那父 agent 和这个跑起来的 subagent，又是怎么互相说话的呢？下一章见真章。

------------------------------------------------------------------------

### 三、父子 Agent 是怎么通信的

隔离机制搞定了，但隔离只是开始，真正决定一个多 agent 系统好不好用的，是**它们之间怎么通信**。

这一章我来讲 Claude Code 的通信方式。但开讲之前，得先立一个分水岭，不然很容易把人带沟里去：**父子之间能怎么通信，取决于你开没开「团队（agent-teams）模式」**。

默认形态和团队形态，是两条很不一样的线：

- **默认形态**：subagent 更像一次「重型工具调用」，父 agent 派它出去、它跑完把结果交回来。这条线里，父 agent 没法中途给在跑的子 agent 插话，消息基本是**子→父 单向通知**。
- **团队（agent-teams）模式**：开启之后才升级成完整的**双向消息驱动**，父 agent 能往子 agent 的信箱里扔字条，子 agent 也能回话，真正的双向对讲。

下面两条线分开讲，你就不会糊涂了。

> 配图意见：父子通信「两条线」对照图。左右两栏，左边「默认形态」画一根从子指向父的单向箭头（标注「只有子→父 完成通知」），右边「团队模式」画父子之间两根来回箭头（标注「父→子 扔字条 + 子→父 通知」）。重点用箭头数量和方向直观表达「默认单向 vs 团队双向」，配色上左栏冷色（克制）、右栏暖色（双向激活）。

#### 默认形态：派出去，跑完把结果交回来

先看默认这条线。这里先停一下，问你一个问题：父 agent 派一个子 agent 出去调研，在子 agent 埋头跑的这段时间里，它俩还能不能说上话？父能不能临时补一句「顺便也看看权限模块」？

凭直觉你可能觉得「应该可以吧」。但默认形态下的答案是：**不能**。

你跟 Claude Code 说「调研一下认证模块」，它派一个 subagent 出去，这个 subagent 带着独立上下文自己跑，跑完把结果作为一次工具调用的返回值（tool_result）原样交回给父 agent。

注意这条线的关键特征：**父 agent 对正在跑的子 agent 是「只能等」的**，没法中途塞新指令给它。就是一次「派出去，等结果」，跟你平时调一个普通工具没啥两样。

那又冒出一个新问题：如果子任务跑得特别久，父 agent 干等着不就被卡死了吗？Claude Code 这里有个补丁，叫 **auto-background**。

**如果 subagent 很快跑完（比如 30 秒内），父 agent 就在前台阻塞等**，像一次普通工具调用，完事就拿结果继续。**但如果 subagent 跑超过 2 分钟还没完，Claude Code 会自动把它转到后台**，让父 agent 可以先继续干别的。子任务真完成时，再回头通知父 agent。

这个设计本质上是**把同步工具调用自动降级成异步通知**的优化。没有它，长任务会一直占着父 agent 的执行权，用户也没法跟父 agent 继续对话。

<figure>

</figure>

源码里这个「2 分钟阈值」就是一个常量开关，而且它本身也带着 feature 门控，不是无条件开的：

```
// src/tools/AgentTool/AgentTool.tsx:72
function getAutoBackgroundMs(): number {
  if (isEnvTruthy(process.env.CLAUDE_AUTO_BACKGROUND_TASKS) 
      || getFeatureValue_CACHED_MAY_BE_STALE('tengu_auto_background_agents', false)) {
    return 120_000;  // 2 分钟
  }
  return 0;
}
```

那子任务转后台之后，完成时怎么告诉父 agent「我干完了」？

最直觉的做法是：给主线程发一个「工具返回结果」事件。但 Claude Code 玩得更骚气，它的设计是：**把完成通知拼成一段 XML，伪装成一条用户消息，塞给父 agent 的对话历史**。

父 agent 那边看到的就像用户发了一条新消息过来，长这样：

```
<task-notification>
<task-id>agent-a1b</task-id>
<output-file>/tmp/xxx.txt</output-file>
<status>completed</status>
<summary>Agent "Investigate auth bug" completed</summary>
<result>Found null pointer in src/auth/validate.ts:42...</result>
<usage>
  <total_tokens>12345</total_tokens>
  <tool_uses>8</tool_uses>
  <duration_ms>34567</duration_ms>
</usage>
</task-notification>
```

> 📌 配图建议：task-notification XML 渲染示意，高亮各个 tag 的含义

<figure>

</figure>

**为啥要搞 XML 不用结构化对象？** 这个设计有它的巧妙之处，我特意想明白过。

**第一**，LLM 对 XML 非常友好。Anthropic 训练 Claude 的时候就强调了 XML 的结构化表达。你把 XML 塞到 prompt 里，LLM 能很自然地解析出语义，不用额外教它。

**第二**，XML 是纯文本，可以直接塞进对话历史。如果是结构化对象，还得额外走个「工具结果」的字段结构，流程更复杂。

**第三**，它伪装成用户消息，**天然地复用了 agentic loop 的处理逻辑**。父 agent 不需要额外的状态机去「等通知」，它就像收到一条新的用户输入一样处理。

这种「把系统事件伪装成对话」的设计思路，在 LLM 应用里是非常值得学的一招。

<figure>

</figure>

对应到源码里，生成这段 XML 的代码就是在拼字符串：

```
// src/tasks/LocalAgentTask/LocalAgentTask.tsx:197
const message = `<${TASK_NOTIFICATION_TAG}>
<${TASK_ID_TAG}>${taskId}</${TASK_ID_TAG}>
<${OUTPUT_FILE_TAG}>${outputPath}</${OUTPUT_FILE_TAG}>
<${STATUS_TAG}>${status}</${STATUS_TAG}>
<${SUMMARY_TAG}>${summary}</${SUMMARY_TAG}>${resultSection}${usageSection}
</${TASK_NOTIFICATION_TAG}>`;
enqueuePendingNotification({ value: message, mode: 'task-notification' });
```

拼完就扔到主 agent 的待处理消息队列里，等主 agent 下一轮循环时当作一条用户消息来处理。

讲到这里，把默认这条线小结一下：父 agent 派出去、等结果，长任务转后台后子 agent 回头发个完成通知。**这条线里有「消息」的，基本只有子→父这一个方向的通知**，父 agent 没法主动给在跑的子 agent 发新指令。如果你只看默认形态，说「subagent 的消息队列只是用来通知父 agent 的、是单向的」，这个判断是站得住的。

#### 团队（agent-teams）模式：父子之间才真正双向对讲

那「父→子 也能发消息」的双向对讲是什么时候出现的？答案是：**开了团队（agent-teams）模式之后**。这是 Claude Code 的一个实验特性，外部用户要显式开启（比如带上 `--agent-teams` 启动），内部默认就开着。

为什么要专门为这个场景设计一套双向通信？我建议你先停个 10 秒想想：如果让你来设计「父 agent 中途给子 agent 发指令」这套通信，你会怎么写？

大概率你脑子里第一反应是「父 agent 调个函数，等 subagent 跑完返回」对吧？这跟我们平时写 RPC 调远程服务的思路一模一样，太自然了。

但我接着追问你两个问题，你看你能不能答上来：

**第一个追问**：如果 subagent 是个跑 5 分钟的代码评审任务，那这 5 分钟里，父 agent 想临时改个要求，怎么递进去？

**第二个追问**：如果父 agent 想同时指挥 5 个 subagent 并行干活、随时给它们各自补充指令，你这个「调函数等返回」的方案要怎么改？

是不是有点卡了？「调函数等返回」这种同步思路，根本没法支持「一边跑一边对讲」。Claude Code 正是看穿了这一点，才在团队模式里铺了一套完全不一样的底座：**消息驱动**。

想象每个 subagent 是公司里一个带「信箱」的独立员工。父 agent 要给它布置新活，就往它信箱里扔一张字条走人，不站在那里等。subagent 自己干完活了，通过另一条信道把结果送回主 agent 的案头。

这个「信箱 + 字条」的模型，本质上就是**消息队列 + 异步通知**。没有直接的函数返回，没有主线程阻塞，所有沟通都是消息。

> 配图意见：沿用原「函数调用 vs 消息驱动」对比图，无需重画。只需把图注/标题里若有「Claude Code 的通信用消息驱动」这类绝对说法，收窄成「团队模式选了消息驱动」，避免读者误以为这是默认形态的通用机制。

<figure>

</figure>

**先看每个 subagent 的「员工档案」**。Claude Code 给每个 subagent 建了一份档案：一个对象，里面记着这个 subagent 的 ID、当前状态（等待中/跑步中/已完成/失败/被停了）、它的信箱（待处理消息数组）、已经产生的结果、进度信息等等。

要说明的是，这份档案本身是通用的，不是团队模式才有，默认形态下那套「转后台、发完成通知」也是靠它来追踪每个子任务的。我们这里要重点盯住的，是里面那个**信箱字段**，它才是团队模式真正用起来、支撑起「父→子 扔字条」的部分。

所有跟 subagent 有关的读写（父要发消息，子要改状态），都通过全局的 task 表里这份档案来进行。

对应到源码里的类型定义大致长这样：

```
// src/tasks/LocalAgentTask/LocalAgentTask.tsx:116
export type LocalAgentTaskState = TaskStateBase & {
  type: 'local_agent';
  agentId: string;               // 子 agent 唯一 ID
  prompt: string;                // 初始任务
  agentType: string;
  status: TaskStatus;            // pending/running/completed/failed/killed
  result?: AgentToolResult;      // 完成后的结果
  progress?: AgentProgress;      // 进度
  isBackgrounded: boolean;       // 是否已转后台
  pendingMessages: string[];     // 信箱：父 agent 扔进来的待处理消息
  messages?: Message[];
};
```

重点关注的是 `pendingMessages` 数组，它就是我们说的「信箱」，父 agent 往里扔字条，子 agent 自己来捡。

<figure>

</figure>

**再看父 → 子怎么扔字条**。父 agent 要给跑着的 subagent 发指令的流程，拆开看就是两步：

**第一步：父往信箱扔字条**。父 agent 在自己的 agentic loop 里调用一个叫 SendMessage 的工具，工具内部做的事情很简单：**往目标 subagent 档案的信箱末尾追加一条消息，然后立刻返回**。父 agent 扔完走人，不等子 agent 看。

这里要点一句关键的：**SendMessage 这个工具本身就是团队模式才启用的**。它的 isEnabled 判断挂的就是「有没有开 agent-teams」这个开关，没开团队模式，主 agent 的工具箱里压根没有 SendMessage，父→子 这条路自然就不存在。这也是前面说「默认形态是单向」的根本原因。

**第二步：子在循环边界自己捡字条**。subagent 自己的 agentic loop 在每一轮工具调用结束后，都会去瞄一眼自己的信箱。如果有新字条，就**把这些字条作为「用户消息」注入自己的对话历史**，然后带着新消息进入下一轮 LLM 调用。

这里还有个细节设计特别巧：**如果子 agent 已经干完活停下来了（completed 或者被手动停了），父 agent 发 SendMessage 会怎样？**

Claude Code 的做法是：**自动把它唤醒**。从磁盘上那份已经保存的对话 transcript 里，把子 agent 的完整对话历史恢复出来，拼上新消息，重新跑起来。这个唤醒机制很妙，意味着 subagent 即使完成了也不是「死了」，父 agent 随时可以叫醒它继续干。

> 配图意见：沿用原父子通信时序图，无需重画。但标题/图注要补上「团队模式下」这个前提（比如「团队模式下的父子双向通信时序」），别让读者误以为这套父→子 + 子→父 的完整来回是默认形态的通用流程。

<figure>

</figure>

对应到源码，SendMessage 工具里的核心逻辑长这样：

```
// src/tools/SendMessageTool/SendMessageTool.ts:800
const task = appState.tasks[agentId]
if (isLocalAgentTask(task) && !isMainSessionTask(task)) {
  if (task.status === 'running') {
    queuePendingMessage(agentId, input.message, context.setAppStateForTasks)
    return { data: { success: true, message: 'Message queued...' } }
  }
  // 任务已停止，自动唤醒从 transcript 里恢复
  const result = await resumeAgentBackground({ agentId, prompt: input.message, ... })
}
```

可以看到就是两个分支：正在跑就扔信箱，已经停了就唤醒。

「扔信箱」这个动作本身的实现就 4 行：

```
// src/tasks/LocalAgentTask/LocalAgentTask.tsx:162
export function queuePendingMessage(taskId, msg, setAppState): void {
  updateTaskState<LocalAgentTaskState>(taskId, setAppState, task => ({
    ...task,
    pendingMessages: [...task.pendingMessages, msg]
  }));
}
```

纯纯的「追加到数组末尾」。子→父 那条信道，复用的就是默认形态里讲过的那套 task-notification（伪装成用户消息）。两个方向凑齐，团队模式下的父子通信才算真正双向。

#### 回头看通信设计的全貌

到这里我们把两条线都讲清楚了，串起来看：

- **默认形态**：父 agent 派子 agent 出去、等结果，长任务转后台后子 agent 回头发完成通知。消息通路基本是**子→父 单向通知**，父 agent 不能中途插话。
- **团队（agent-teams）模式**：在上面基础上，补齐**父→子**这条路（SendMessage 往信箱扔字条），凑成完整的双向消息驱动。

团队模式那套双向消息体系，落到底就两个关键字：**异步** + **消息**。没有直接函数调用，没有锁，没有回调地狱，全靠读写共享的任务状态和消息队列。

> 配图意见：这张「通信全貌」图建议重画。原图若只画了一套双向通信，现在撑不起「两条线」这个新框架了。改成上下或左右两栏：一栏「默认形态」画「父派出 → 子跑 →（超 2 分钟转后台）→ 子发完成通知」的单向链路，一栏「团队模式」在其上叠加「父→子 SendMessage 扔字条」的回路。重点让读者一眼看出：团队模式是在默认形态之上「补了父→子这一条」，而不是另起炉灶。

<figure>

</figure>

而且不管走哪条线，这套「不阻塞」的底子都带来一个特别大的好处：**天然支持多 subagent 并发**。只要父 agent 不傻等着某个子跑完，它就能同时派 5 个 subagent 出去，谁先完成谁先给它发通知，父 agent 按到达顺序处理就行。并发不是团队模式的专利，默认形态配合 auto-background 也能并发，后面要讲的 Coordinator 模式更是把并发拉到极致。

<figure>

</figure>

下一章，我们再讲一个特别精妙的优化：**Fork Subagent**。

------------------------------------------------------------------------

### 四、Fork Subagent：省钱又省延迟的隐藏大招

前面讲的常规 subagent 已经是主流玩法了，但 Claude Code 还有一个更精妙的机制，叫 **Fork Subagent**。这个机制有点隐蔽，用起来是透明的，但对成本和延迟的优化非常显著。

我先抛两个问题让你估算下，**先别往下翻看答案**：

第一，Claude Code 的 system prompt 大概有多长？是几百 token、几千 token，还是上万 token？  
第二，每派一个 subagent，如果它有自己独立的 system prompt，LLM API 那边对这段 prompt 是从头算一遍，还是有办法复用？

#### subagent 的隐藏成本

公布答案：Claude Code 的 system prompt 长度是**上万 token**，里面塞了大量的工具说明、规范约定、用户上下文。

而每派一个 subagent，如果它有独立的 system prompt（内置的 Explore、Plan 这些都有独立的），LLM API 那边就得**对这一万多 token 重新从头算一遍**，就跟没见过似的。

这有两个代价：**钱**（input token 重新算钱）和**延迟**（首 token 等更久）。在生产环境里，subagent 派得越频繁，这个开销线性放大，是个很可怕的成本黑洞。

Anthropic 有个 **prompt 缓存**机制可以缓解这事。简单说：**API 请求里如果前缀跟之前某次请求一样，这段前缀可以不重新算，直接走缓存，价钱只要原来的 10%，延迟也大幅降低**。

到这儿我再问你一个关键的：**prompt 缓存命中的条件是「内容大致相同」就行，还是「字符级别相同」，还是「字节级别完全相同」**？再猜一下。

公布：是**最严格的那个，字节级别完全相同**。系统 prompt 一个字不一样、工具列表顺序不一样、甚至空格位置不一样，都会直接没命中缓存。

是不是比你想的严格多了？

那既然这么严，能不能设计一种 subagent，它的 system prompt 和工具池跟父 agent **完全一样**，这样就能复用父的缓存了？这就是 Fork Subagent 的起点。

<figure>

</figure>

#### Fork 的核心思路：派一个「字节级相同」的分身

Fork Subagent 的直觉是这样的：**派一个子 agent 出去干活，但这个子 agent 的 API 请求前缀跟父 agent 一模一样，让 Anthropic 那边一看：「哦这个前缀我认识」，走缓存**。

这里的「一模一样」要做到什么程度？**字节级**。一个字节不对都不行。

具体要对齐哪些东西呢？有五样必须跟父 agent 完全一致：

1.  **系统 prompt 的内容**（最核心的，对齐第一位）
2.  **用户上下文**（拼在消息前的那部分动态内容，比如当前项目的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 内容）
3.  **系统上下文**（拼在 system prompt 后的环境信息）
4.  **工具池的顺序和定义**（工具的字段结构会被序列化进 API 请求，顺序都不能变）
5.  **对话历史的前缀**（决定了 user/assistant 消息序列中「从哪里开始分叉」）

这五样只要有一样跟父 agent 字节不一致，缓存就直接没了。

对应到源码里，Claude Code 专门定义了一个类型（CacheSafeParams），把这五项打包：

```
// src/utils/forkedAgent.ts:57
export type CacheSafeParams = {
  /** System prompt - 必须跟父完全一致 */
  systemPrompt: SystemPrompt
  /** User context - 拼接在消息前，影响缓存 */
  userContext: { [k: string]: string }
  /** System context - 拼接在 system prompt 后，影响缓存 */
  systemContext: { [k: string]: string }
  /** 工具池、模型等所在的上下文 */
  toolUseContext: ToolUseContext
  /** 父 agent 的消息前缀，用于缓存共享 */
  forkContextMessages: Message[]
}
```

你看这个类型的意思很明显：**凡是会影响缓存命中的字段，我全列在这儿，你 Fork 的时候严格按这份清单跟父 agent 对齐**。

<figure>

</figure>

#### 一个有意思的细节：system prompt 不重新生成

Fork Subagent 的合成定义里有个有意思的细节，值得单独说。

正常一个 subagent 有个生成 system prompt 的函数，跑的时候现生成一段 prompt 文本。但 Fork 机制用的那个 subagent 的生成函数**直接返回空字符串**：

```
// src/tools/AgentTool/forkSubagent.ts:60
export const FORK_AGENT = {
  agentType: FORK_SUBAGENT_TYPE,
  tools: ['*'],             // 用父的完整工具池
  maxTurns: 200,
  model: 'inherit',          // 继承父的模型
  permissionMode: 'bubble',  // 权限弹窗浮到父终端
  source: 'built-in',
  getSystemPrompt: () => '', // 返回空串！
} satisfies BuiltInAgentDefinition
```

这不是偷懒，而是精心设计的。

为啥要返回空串？因为 Fork subagent 的 system prompt **根本不走这个函数生成**，而是**直接用父 agent 已经渲染好的那份字节**。

原因很简单：如果重新调一次生成函数，里面可能有些小差异（比如某个功能开关的缓存状态变了、某个动态字段的值变了），生成出来的 prompt 跟父 agent 就可能差一个字符，缓存就没了。

最稳的办法是：**把父 agent 那边已经渲染出来的 prompt，作为字节原样拿过来用，一个字节都不动**。

这个细节非常工业级，普通人写 agent 系统根本想不到。

<figure>

</figure>

#### 什么时候用 Fork，什么时候用常规 subagent？

Fork 机制不是万能的，它的**适用场景很特定**：你希望子 agent 完全继承父 agent 的整个上下文（对话历史、system prompt、工具池），只是「派个分身去试试另一条路」。

比如「Ctrl+F 生成 PR 描述」「运行 /btw 命令做 post-turn 总结」，这些任务需要父 agent 的完整上下文，但又不希望污染父 agent 的主循环。

相反，如果你的任务有明确的专业分工（比如派一个专门搜代码的 agent、派一个专门做规划的 agent），那就用常规 subagent，它们的 system prompt 是定制的，Fork 机制反而不适用。

还有一个关键点：**Fork 机制和 Coordinator 模式是互斥的**。Coordinator 模式下主 agent 已经是个纯协调者了，它派的 worker 本来就是异步的，不需要 Fork 这种「轻量分身」机制。两个机制职责重叠，就只留一个：

```
// src/tools/AgentTool/forkSubagent.ts:32
export function isForkSubagentEnabled(): boolean {
  if (feature('FORK_SUBAGENT')) {
    if (isCoordinatorMode()) return false  // 互斥！
    if (getIsNonInteractiveSession()) return false
    return true
  }
  return false
}
```

<figure>

</figure>

#### Fork 的工程启示

Fork 机制我想单独说下它对我们的启示。

很多人做 agent 系统只关心「能不能跑起来」，不关心「跑起来要花多少钱」。但在生产环境，这两个是一回事。Claude Code 靠 Fork 机制，在缓存友好的场景下能把 subagent 的成本降到原来的 **10%** 左右。

这意味着什么？意味着你的 subagent 可以调得**更频繁**。原本成本考虑不敢派的活，现在都能派了，这反过来又让整个 agent 系统的能力边界扩大了。

所以**成本优化本身就是能力的一部分**。这个思路我觉得对自建 agent 系统的朋友特别重要。

<figure>

</figure>

好了，讲完 Fork，下面进入整篇文章最「多 agent」的一章：Coordinator 模式。

------------------------------------------------------------------------

### 五、Coordinator 模式：真正的多 Agent 并行协作

前面讲的 subagent（不管是常规的还是 Fork 的），本质都是**父子结构**：父 agent 派一个子，自己该干啥干啥，子完成了通知一声。

但如果你的任务量很大，需要**一堆 agent 同时开工**呢？比如一个大的代码迁移，要并行调研 10 个模块。这时候父子结构就显得单薄了。

Claude Code 为此设计了一个专门的模式：**Coordinator 模式**。这是 Claude Code 多 agent 设计里最「多 agent」的部分，也是最能打的地方。

#### Coordinator 模式的启用

这个模式不是默认开的，要显式打开。需要同时满足两个条件：**编译时的功能开关**和**运行时的环境变量 `CLAUDE_CODE_COORDINATOR_MODE=1`**。

```
// src/coordinator/coordinatorMode.ts:36
export function isCoordinatorMode(): boolean {
  if (feature('COORDINATOR_MODE')) {
    return isEnvTruthy(process.env.CLAUDE_CODE_COORDINATOR_MODE)
  }
  return false
}
```

开启之后，主 agent 的行为模式会发生根本性变化。

#### 核心设计：主 agent 退化成「纯协调者」

常规模式下，主 agent 是「全能型选手」：它读代码、写代码、跑测试、做规划全都干，只在需要时才派 subagent 帮一把。

Coordinator 模式下，主 agent 不干实际工作了，它只做三件事：**派 worker、收结果、合成答案**。

这个角色转换是通过主 agent 的 system prompt 强制约束出来的。打开源码里那段 prompt，开头就写得很明白：

```
You are Claude Code, an AI assistant that orchestrates software engineering 
tasks across multiple workers.

## 1. Your Role
You are a **coordinator**. Your job is to:
- Help the user achieve their goal
- Direct workers to research, implement and verify code changes
- Synthesize results and communicate with the user
- Answer questions directly when possible, don't delegate work 
  that you can handle without tools
```

翻译一下：**你的身份是协调者，你的工作是指挥 worker 去做研究、实现、验证，然后自己合成结果跟用户交流。能自己回答的问题不要派人去做**。

<figure>

</figure>

#### 三大内部工具

既然主 agent 要协调，就得有专门的协调工具。Coordinator 模式下，主 agent 多了一套「团队管理」工具箱：

- **派 worker 的工具**：派一个新 worker 出去干某件具体的活，派完立刻返回 worker 的 ID。
- **创建/解散团队的工具**：批量管理 worker 组。
- **给 worker 发消息的工具**：给已经派出去的 worker 发后续指令（也就是前面讲的 SendMessage），因为 worker 的上下文还在，续命比重新派一个更省钱。
- **合成最终输出的工具**：协调者合成完答案后，通过这个工具把最终回复交给用户。
- **停止 worker 的工具**：当协调者意识到某个 worker 跑错方向时，把它停掉省 token。

这套工具放在一起，协调者就有了一整套指挥团队的 API。

> 📌 配图建议：协调者工具箱图，把五个工具画成五个按钮，标注每个按钮的作用

<figure>

</figure>

对应到源码里，有这么一组常量把这几样工具圈在一起：

```
// src/coordinator/coordinatorMode.ts:29
const INTERNAL_WORKER_TOOLS = new Set([
  TEAM_CREATE_TOOL_NAME,       // 创建 worker 团队
  TEAM_DELETE_TOOL_NAME,       // 解散团队
  SEND_MESSAGE_TOOL_NAME,      // 给 worker 发消息
  SYNTHETIC_OUTPUT_TOOL_NAME,  // 合成最终输出给用户
])
```

这里得说清楚它的真实用途，免得误会：这组常量其实是一张**给 worker 用的「黑名单」**。Coordinator 模式下，系统会把这几样工具从 worker 的工具池里**摘掉**，让 worker 只管干活、没法反过来去创建团队、给别人发消息、调度别人。所以它不是「只有协调者才有这些工具」，而是「这些协调专用的工具，不发给 worker」。特别是 SendMessage，它本身并不是 Coordinator 模式专属的东西，前面第三章讲的团队（agent-teams）模式里，父 agent 用的就是它。

顺带厘清一个容易混的点：**agent-teams（团队/队友）模式和 Coordinator 模式是两个独立的开关**。前者管的是「父子之间双向发消息、派队友」，后者是更进一步的「主 agent 退化成纯协调者」的编排模式，别把两者当成一回事。

#### 并行才是真本事

Coordinator 模式的 prompt 里有一句我特别喜欢：

> Parallelism is your superpower. Workers are async. Launch independent workers concurrently whenever possible, don't serialize work that can run simultaneously and look for opportunities to fan out.

翻译一下：**并行是你的超能力，worker 全是异步的，能并行的绝不串行，多找机会一口气派一堆出去**。

这句话背后是一个很关键的工程事实：Claude Code 的派 worker 工具调用**可以在同一条 assistant 消息里出现多次**，底层会一起并发执行，不是一个跑完再跑下一个。

所以协调者要做的就是在一次 LLM 回合里，一口气生成多个派 worker 的工具调用：

```
派 worker 调研 auth 模块
派 worker 调研 session 模块
派 worker 调研 token 模块
```

这三个调用同时启动，三个 worker 同时干活，协调者等通知一条条返回。

<figure>

</figure>

对比一下：

- **串行**：派 worker1 → 等 → 结果 → 派 worker2 → 等 → 结果 → 派 worker3... 用户等十分钟
- **并行**：同时派三个 worker → 三份结果陆续到 → 用户等三分钟多一点

这就是「并行是超能力」的真正含义。工业级多 agent 系统，没有并行就没有可用性。

<figure>

</figure>

#### 协调者的「任务流水线」

Coordinator 模式下，一个典型的任务流程被切成四个阶段：

| 阶段 | 谁来做          | 目的                           |
|------|-----------------|--------------------------------|
| 调研 | Workers（并行） | 调查代码库、找文件、理解问题   |
| 合成 | **协调者本人**  | 读完发现、理解问题、写实现规格 |
| 实现 | Workers         | 按规格做具体修改、提交         |
| 验证 | Workers         | 测试改动是否真的工作           |

注意中间的「合成」阶段是协调者**亲自**做，这是协调者存在的意义：**理解全局，做决策**。prompt 里反复强调：不要偷懒让 worker「based on your findings, implement the fix」，而是自己把 findings 读懂、写成具体的规格再派下去。

<figure>

</figure>

这是一个非常重要的 multi-agent 设计哲学：**协调者必须「理解」而不能「转发」**。如果协调者只是转发，它就没有存在价值，worker 直接跟用户对话就行了。

<figure>

</figure>

#### Continue vs Spawn：老 worker 还是新 worker？

协调者要持续派活，遇到一个新任务，是**给老 worker 发消息续命**，还是**派个新 worker 从头开始**？这是个有经验才能做好的决策。

Claude Code 的 prompt 里给出了一张决策表，我总结一下核心逻辑：

- 如果新任务跟 worker 现有上下文**高度相关**（比如刚查的文件现在要改），**续命老 worker**，因为它已经「知道」那些文件了。
- 如果新任务跟 worker 现有上下文**没关系**，或者之前 worker 的工作走偏了，**派新 worker**，避免旧上下文干扰判断。
- **验证**这种需要「新鲜眼光」的工作，永远派新 worker，不能让刚写完代码的 worker 自己验自己。

这个设计其实也挺反映人类团队合作的直觉：有的活就该让懂上下文的人接着干（沟通成本低），有的活就该换个人做（避免认知偏差）。

<figure>

</figure>

#### Worker 的工具限制

Coordinator 模式下，worker 拿到的工具有什么不同？关键在于：**协调者专属的那套内部工具（创建团队、发消息、合成输出等等），不给 worker 用**。worker 不需要再去协调别人，它的活是干事情。

这其实是一个**递归防护**：如果 worker 也能派 worker，整个系统就变成递归树了，没完没了。通过工具白名单把 worker 的「派人权」收回，让系统结构保持「一个协调者 + 一堆 worker」的扁平形态。

<figure>

</figure>

#### 跟常规 subagent 对比

讲完这些我们对比一下 Coordinator 模式和常规 subagent：

| 维度          | 常规 subagent            | Coordinator 模式       |
|---------------|--------------------------|------------------------|
| 主 agent 角色 | 全能选手                 | 纯协调者               |
| subagent 执行 | 同步（2 分钟后才转后台） | 默认异步               |
| 并发程度      | 偶尔并发                 | 最大化并发             |
| 适合场景      | 单个任务 + 临时帮手      | 大任务 + 高并发拆解    |
| 系统形态      | 父子树                   | 协调者 + worker 扁平层 |

<figure>

</figure>

#### Coordinator 模式的工程启示

讲完 Coordinator，我想提炼几条值得学的设计思想。

**第一，角色分离**。协调和干活是两件事，不要让同一个 agent 身兼二职。角色清晰的系统更稳定。

**第二，并发优先**。异步 + 消息队列是并发的基础，有了这套基础，多 agent 才能真正发挥威力。

**第三，合成不转发**。协调者要理解中间结果，不能把它当传话筒。这是 Multi-Agent 系统里最容易踩坑的一点。

**第四，扁平不递归**。通过工具权限把层级限制在两层（协调者 + worker），避免失控的递归嵌套。

<figure>

</figure>

------------------------------------------------------------------------

### 六、5 条 Multi-Agent 设计原则

Claude Code 的源码扒得差不多了。我把前面讲的所有东西浓缩一下，沉淀成 5 条可以直接用到自己项目、也可以直接用到面试答案里的设计原则。

<figure>

</figure>

#### 原则 1：上下文隔离要按字段粒度做

这是我最想强调的一条。很多 agent 框架的「隔离」就是粗暴地给 subagent 一个空 context，结果缺这缺那一堆 bug。

Claude Code 的做法是：**每个状态单独决策**。读文件缓存克隆（避免污染），写全局状态关掉（避免两边抢），任务注册通路保留（不然孤儿进程没人回收），深度计数 +1（可追踪，防失控嵌套）。

做多 agent 系统时，对着父 agent 的每项状态问一句：「子 agent 拿这个状态干啥？会不会影响父？」，就能避开大部分坑。

#### 原则 2：通信走消息，不走函数调用

**父 → 子**：写入子 agent 的消息队列，子 agent 下一轮循环自己读取。

**子 → 父**：把完成通知包装成 XML 消息，伪装成用户消息注入父 agent 对话。

这套模型的好处：天然异步、天然支持并发、天然兼容 agentic loop、天然持久化（消息都能落盘）。

要补一句严谨的：上面这套完整的**双向**消息驱动，是 Claude Code 团队（agent-teams）模式打开后的形态。默认的常规 subagent 更接近「同步派发 + 子→父 完成通知」，父→子 这条主动发消息的路是团队模式才接通的。面试时把这个边界讲清楚，比笼统说「双向」更显你真读过源码。

如果你问面试官「你们的多 agent 之间怎么通信」，把这套答出来，基本就到位了。

#### 原则 3：工具权限要分级管控

**全局黑名单**（防递归、防乱问用户），**类型黑名单**（自定义 agent 更严），**异步白名单**（后台 agent 只能用子集）。

每种 agent 按自己的场景配工具，不要一刀切。

#### 原则 4：缓存友好是一种架构能力

API 成本和延迟对生产环境 agent 来说是**能力的一部分**。设计 subagent 的时候，考虑它的 prompt 前缀能不能复用父 agent 的缓存，能省 80-90% 的成本。

Claude Code 那套「严格锁定缓存前缀 + 复用父 agent 已渲染字节」的思路，是这方面的教科书式实现。

#### 原则 5：并行优先 + 协调者合成

真正的多 agent 系统威力在**并发**。通过异步消息和消息队列做基础，通过协调者做合成，避免「大 agent 大循环什么都自己扛」的窘境。

并且协调者要**亲自合成**，不能当传话筒。

这 5 条原则背后，其实都能看到 Claude Code 源码里的清晰落点。我建议你别光记这些原则，**下次看到 Multi-Agent 相关的东西，都拿这 5 条去对照**，会迅速看出对方系统的深浅。

------------------------------------------------------------------------

### 最后

写到这里，Claude Code 的多 Agent 机制基本就扒完了。

回过头看，Claude Code 这套系统不是简单的「一个主 agent 嵌几个 subagent」那么朴素。它在架构、通信、并发、成本、隔离每一个维度都做了精致的设计：

- **按字段粒度做的上下文隔离**，既不让 subagent 污染父 agent，又保留了必要的通路。

- **消息队列 + XML 通知**支撑起异步父子通信，让并发成为可能。

- **Fork Subagent 的缓存前缀复用**，把成本打到缓存友好的极致。

- **Coordinator 模式** 把主 agent 彻底解放成纯协调者，让多 worker 真正并行起来。

每一块拆开看都不是啥复杂技术，但组合在一起，就成了一个能支撑 Anthropic 这种级别产品的工业级多 agent 系统。

今天分享都到这里，我们下篇见！
