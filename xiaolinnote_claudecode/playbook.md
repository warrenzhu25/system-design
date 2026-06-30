# Playbook

> Archived from https://xiaolinnote.com/claudecode/ (playbook). Personal study copy.


<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md 指南：Claude Code 的项目记忆该怎么写？</a>

大家好，我是小林。

前阵子，有个林友在群里发牢骚。

他说给 Claude Code 写了一份 1000 多行的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>：整个项目架构文档抄了一份、团队术语表搬了一份、连「我们希望测试覆盖率到 90%」这种愿望也堆上去，自我感觉特别细致。

结果呢？Claude 该忘的还是忘，该违规的还是违规。

是啊，<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 这玩意，很多人用 Claude Code 这么久了，也是想到啥写啥，越写越长。

还真没认真想过：写得多，到底是好事还是坏事？

我把 Anthropic 官方文档整个翻了一遍，这一翻不要紧，我发现自己之前的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 一半内容压根是负资产。

今天就把这套经验整个分享出来，按这个顺序展开：

- <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 到底是个什么东西？
- 为什么写多了反而废？
- 什么样的规则才真正生效？
- 怎么把 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 分层组织？
- 怎么用 `/init` 和 `/memory` 维护？
- <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 到底该怎么写？

读完这篇，你应该能立刻去改自己的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>，从「写了等于没写」变成「真正能让 Claude 听话」。

01｜<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md 到底是个什么东西？</a>

很多人一上来就开始问「<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 应该写些啥」、「行数多少合适」、「能不能 import 别的文件」。

但我觉得这些问题之前，得先回答一个更基本的问题：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 到底是干啥的？为什么 Claude Code 要专门搞这么个文件？

你想象一个场景。

你刚入职一家公司，主管丢给你一份文档，标题叫「团队约定」。里面写了：我们用 yarn 不用 npm，API 在 `src/api/` 下，生产数据库千万别动，提 PR 之前要跑 `yarn lint`。

这份文档看着不起眼，但效果惊人。新人不用一遍遍问「咱们这边怎么做 X」，老人也不用一遍遍重复回答。一份文档，省下整个团队的沟通成本。

<figure>

</figure>

<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 就是给 Claude 的这份「团队约定」。

说穿了，它本质上就是一个普通的 markdown 文件，文件名固定叫 `CLAUDE.md`，放在你项目的根目录下。

你能用记事本打开，能用 VSCode 编辑，跟你平时写的 README 一样，里面就是写写规则、放放说明，谁都能上手。

但它特别的地方在于：每次你打开 Claude Code 跟它聊天，**Claude 都会自动把这个文件读一遍**，作为整个对话的「ground truth」（基准事实，可以理解为「默认成立的前提」）。你后面提的需求、它做的判断，全都是在这份「团队约定」的基础上推进的。

换句话说：在你输入第一句提问之前，在写任何代码之前，在任何事情发生之前，Claude 都会先读这个文件，并把它当作整段会话的默认前提。它不是「可选的提示」，而是「默认的前提」。

<figure>

</figure>

说概念可能有点虚，咱们来做个最简单的实验你就懂了。

你在项目根目录新建一个 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>，里面就写一行规则：

```
解释任何技术概念时，必须以「打个比方」开头。
```

保存。然后打开 Claude Code 问它一个完全无关的问题：

> 你：什么是 agent？  
> Claude：**打个比方**，agent 就像你雇的一个能自己干活的助理。你跟它说「帮我把上周的数据整理成报表」，它会自己拆任务、找文件、读数据、生成表格，中间用什么工具、分几步走全都它自己决定，最后把成品交给你，不用你一步步指挥……

你看，你压根没在提问里提这条规则，Claude 自己就把它套上了。

更妙的是，下次再开新 session 问别的，比如「什么是 WebSocket」，它还是会以「打个比方」开头。规则一旦写进 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>，相当于给 Claude 装了个稳定的「长期记忆」，每个新对话都默认带着。

<figure>

</figure>

讲到这儿，你可能会冒出一个新疑问：那为啥要专门搞一个 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>？

我直接把规则写在 README 里不行吗？反正它俩都是 markdown 文件。

不行。两个文件长得像，但定位完全不一样。

Anthropic 官方文档里有句话点醒了我：「README 是写给人看的，<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 是写给 agent 看的，两个读者群体不一样，密度也不一样。」

啥意思？README 是给开发者翻的，写项目介绍、快速上手、贡献指南，长一点没事，散一点没事，反正人会跳读。

**而且 Claude 默认不会主动去读 README**，你不告诉它去看，它就当不存在。

<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 才是那个被自动加载、每次都吃 token 的「默认配置」。

（顺带说一句，token 简单理解就是模型读写时的「字符单元」，大致 1 个中文字 ≈ 1 个 token；每次请求消耗的 token 既算钱也占用上下文窗口，所以「省 token」全文会反复出现。）

<figure>

</figure>

那它具体是怎么被加载的？这里顺便贴一小段 Claude Code 源码让你看看背后的机制：

```
const dirs: string[] = []
const originalCwd = getOriginalCwd()
let currentDir = originalCwd

while (currentDir !== parse(currentDir).root) {
  dirs.push(currentDir)
  currentDir = dirname(currentDir)
}
```

这段代码出自 `src/utils/claudemd.ts`。逻辑很朴素：从你当前所在的目录一路往上爬到文件系统根目录，每爬一层就把目录名记下来。爬完之后再反向遍历，从根目录往下读每一层的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 和 `.claude/CLAUDE.md`，全部合并喂给模型。

所以一个项目可能同时有好几份 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 在生效，这一点我们第四节会展开。

<figure>

</figure>

到这里你应该能感觉到 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 的特殊之处了：**它不是文档，是配置**。是你给 Claude 配的「这个项目的预设」。

<figure>

</figure>

理解了这一层，接下来就要扒一个让所有人意外的事实：写得越多，效果反而越差。

### 02｜写多了反而废？

> Source: https://xiaolinnote.com/claudecode/playbook/cc_claude_md.html

我刚开始用 Claude Code 的时候，是这么想的：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 嘛，多写点总没坏处，规则越细，Claude 越知道我要啥。

后来看到一组数据，我直接把自己 400 多行的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 删了一半。

这个数据来自一个叫 SFEIR Institute 的技术博客。他们做了一组实测：把所有规则塞在一个 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里，**控制在 200 行以内的时候，规则遵守率大概 92%**。但写到 400 行往上，遵守率就肉眼可见地往下掉。

<figure>

</figure>

更有意思的是，如果你把 200 行拆成 5 个 30 行的模块化文件，丢到 `.claude/rules/` 目录里，**遵守率反而能涨到 96%**。

写得多反而不听，写少了拆开反而听了。这跟我朴素的直觉完全是反的。

为啥会这样？两个原因。

第一个原因，**token 经济**。<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 每次启动都会被完整加载进上下文窗口。你写 400 行，每次请求就消耗几千 token，挤压你的对话、Claude 的思考、工具调用结果的位置。

打个比方，会议桌上摆 50 张便签，重点一目了然。换成 400 张，整张桌子都被淹没，谁也找不着重点。

<figure>

</figure>

第二个原因，**注意力稀释**。模型的注意力不是无限的，规则一多，每条规则在模型脑子里的权重就被摊薄了。社区里不少重度用户都聊过这个体感：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 超过 300 行之后，「记不住」就变成常态。

<figure>

</figure>

讲到这儿你可能想，那只要控制在 200 行就行了？也不全是。光控制行数还不够，**得知道哪些东西根本就不该写进 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>**。

肯定有不少人的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里塞着大量负资产。最典型的三类反例：

**第一类，复述型。** 把整个项目架构文档复制粘贴进 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>，一写写 100 行。问题是项目架构会变，今天 React，半年后可能就 Vue 了，<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里的 100 行还停留在 React 时代。正确做法是一行话指过去：「项目架构详见 docs/architecture.md」，Claude 真要看自己会去 read。

**第二类，愿望型。** 「我们希望测试覆盖率达到 90%」、「我们的目标是 0 bug」。这种话听着政治正确，但 Claude 没法判断「希望」和「实际」的差距，可能为了「满足愿望」给你乱补一堆没意义的测试。<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里只写当下实际执行的规则，「PR 提交前必须跑 npm test」是规则，「我们希望大家多写测试」是 PUA。

**第三类，术语表型。** 把团队术语表往 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里搬。「Repo 指 repository、PR 指 pull request……」Claude 是个 LLM，这些通用术语它都懂。你真正需要解释的是团队特有的黑话（比如「我们说『小灰』指的是预发布环境」），但也建议放 `docs/glossary.md` 里。

<figure>

</figure>

把这三类垃圾清掉，你的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 可能直接从 400 行瘦到 80 行。Claude 的表现，下一次开 session 就能感觉到。

### 03｜什么样的规则才真正「有效」？

清完垃圾，问题就变成：剩下那些该写的规则，到底怎么写才有用？

我先抛个问题给你猜：同样讲缩进，下面哪种写法 Claude 听得更好？

- A：「所有 TypeScript 文件用 2 个空格缩进」
- B：「代码要按规范格式化」

如果你猜 A，恭喜，答对了。但你能说清楚为什么吗？

关键差异在一个词：**可验证**。

A 是具体的，Claude 写完代码自己就能数：是不是 2 个空格？是不是 TypeScript 文件？这两个问题都有明确答案，它能自检。

B 是模糊的，什么叫「按规范格式化」？这个判断需要外部标准，Claude 只能猜你的偏好，猜得对就对，猜得错就错。

<figure>

</figure>

关于「啥样的规则才有效」，可以浓缩成一句话四个原则：

> 短、具体、告诉为什么、持续更新。

这四点挨个聊聊。

**短，** 上一节聊过了，呼应 200 行的黄金线。

**具体，** 就是上面 A 和 B 的差异。再举几个例子你感受一下：

| 模糊写法（无效） | 具体写法（有效） |
|----|----|
| 测试一下你的修改 | 提交前跑 `npm test` |
| 保持目录整洁 | API 处理函数放在 `src/api/handlers/` 目录下 |
| 别把构建搞挂了 | 推代码前跑 `npm run typecheck` 检查类型 |
| 用好的命名 | 组件文件用 PascalCase（大驼峰），工具函数用 kebab-case（短横线小写） |

「具体」其实就是把抽象意图翻译成可执行命令、可定位路径、可验证规则。Claude 不是你团队里磨合三年的老同事，它没法靠默契理解你的意思。

<figure>

</figure>

**告诉为什么，** 这条乍一看像废话。规则就是规则，Claude 照做就行，还要告诉它为啥？

要的。**而且这是四条里最关键的一条**。

比如你写「不要在测试里写入生产数据库」，Claude 知道不能写生产库就完了。

但你加一句「因为去年有次测试不小心把 users 表清空了，出过事故」，Claude 不光知道这条规则，还知道**规则的边界**。

啥意思？以后你跑预发布环境（staging）测试，问它能不能写预发布数据库，它会基于「规则的本质是防生产事故」做出正确判断，而不是机械地说「规则说了不能写数据库」。

**告诉「为什么」不是废话，是给 Claude 留判断空间。**

<figure>

</figure>

**持续更新，** 就是把 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 当活文档维护。

Claude 在哪儿犯错了两次以上，你就加一条防御规则。但同样重要的是：**老规则要删**。

<a href="http://claudeguide.io" target="_blank" rel="noopener noreferrer">claudeguide.io</a> 上有句话特别戳：「错误的规则比没有规则更糟。」

想想也是，规则在那儿摆着，Claude 就会试图遵守，但规则本身已经过时了，结果就是你在花 token 买一份混乱。

<figure>

</figure>

讲到这儿，想多说一句：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 其实没有标准模板，每个项目都该有自己的样子。

Claude Code 当初的设计理念就是「让用户随便用、随便改、随便魔改」，根本没有所谓「正确」的用法。

所以别迷信任何人的「最佳实践」，包括我这篇。**把原则吃透，按你项目的实际情况裁剪。**

<figure>

</figure>

04｜<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md 不只是一个文件</a>

讲到这里，你可能默认 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 就是项目根目录下那一个文件。

但实际上，**<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 是分层的**。

一个项目可能有好几份 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 同时在生效。

回想一下第一节那段源码做的事：从你的工作目录一路往上爬，每一层都尝试读 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 和 `.claude/CLAUDE.md`，全部合并喂给模型。

所以一份完整的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 生态长这样：

- **项目根的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>**：写整个项目的通用约定（技术栈、目录、命令、硬约束），每次启动都加载，是大头。
- **子目录的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>**：比如前端 `frontend/CLAUDE.md` 写组件约定。这层按需加载，Claude 工作到该目录才生效，不污染整个项目上下文。
- **`~/.claude/CLAUDE.md` 全局**：跨项目的个人偏好（比如「永远用中文回复」、「我喜欢 4 空格缩进」），相当于给所有 Claude 打了同一份补丁。

<figure>

</figure>

文字可能还有点抽象，咱们拿一个典型的前后端分离项目举例，目录结构大概长这样：

```
~/.claude/
└── CLAUDE.md          # 全局：用中文回复我、commit message 写中文

my-project/
├── CLAUDE.md          # 项目根：技术栈、目录结构、命令、硬约束
├── frontend/
│   ├── CLAUDE.md      # 前端模块：组件用函数式、状态管理用 Zustand
│   └── src/
└── backend/
    ├── CLAUDE.md      # 后端模块：API 用 RESTful 风格、错误统一抛 AppError
    └── src/
```

启动 Claude Code 的时候，根目录的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 和 `~/.claude/CLAUDE.md` 会自动合并加载。等你让 Claude 改 `frontend/` 里的代码，它才会顺手把 `frontend/CLAUDE.md` 也读进来。改后端代码时，前端那份规则压根不会进上下文，节省 token。

理解了这三层，你会发现玩法一下打开了。**项目通用规则放项目根、模块特有规则放子目录、个人偏好放全局**，各管各的，互不污染。

但还有一层更进阶的玩法：`.claude/rules/` 目录。

这是 Claude Code 提供的「模块化 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>」机制。你不在 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里堆所有规则，而是在 `.claude/rules/` 目录下每个主题一个文件。

举个例子，你的 `.claude/rules/` 目录可能长这样：

```
.claude/
└── rules/
    ├── testing.md       # 测试规则
    ├── api-design.md    # 接口设计规则
    ├── security.md      # 安全规则
    └── ui-components.md # UI 组件约定
```

每个文件聚焦一个主题，控制在 30 行以内，结构清爽好维护。

最妙的是，每个 rules 文件可以加一段 YAML frontmatter（写在文件最顶部、用 `---` 包起来的一段元信息），标注「这规则只在改某类文件的时候加载」。比如 `testing.md` 长这样：

```
---
paths: ["**/*.test.ts", "**/*.spec.ts"]
---
# 测试规则
- 用 describe / it，不用 test()
- mock 外部依赖必须用 vi.mock
- 每个测试只写一个断言
- 别用 expect.anything()，断言要精确
```

frontmatter 里的 `paths` 告诉 Claude：「这条规则只在改测试文件时才加载」，业务代码改起来你压根看不到这份规则。

同理，`api-design.md` 顶部可以写 `paths: ["src/api/**/*.ts"]`，Claude 只在改接口代码时才加载：

```
---
paths: ["src/api/**/*.ts"]
---
# 接口设计规则
- 所有接口走 RESTful 命名（GET / POST / PUT / DELETE）
- 返回值统一用 { data, error } 格式
- 错误码用 4 位数字（如 1001、1002），别用字符串
```

这就叫 **path-scoped rules**（路径作用域规则）。Claude 只在工作到匹配路径的文件时才把这份规则加载进上下文。改业务代码的时候根本看不到测试规则，改接口的时候也不会看到 UI 组件约定，省下来的 token 全留给真正有用的对话。

<figure>

</figure>

打个比方，公司有总公司手册、各部门有部门手册、每个岗位有岗位手册。你不会让每个新人都把所有手册随身带着，对应业务的时候才翻对应的手册。

这种模块化拆分的好处就是上一节那个 96% 数据：少加载、按需加载，效果反而比一坨更好。

<figure>

</figure>

我之前看到有海外开发者的方案把这套生态推到了极致：

> <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 起步；长了拆 `rules/`；高频工作流写到 `commands/`；可复用能力封装成 `skills/`。

<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 只是入口，后面还有 commands（自定义命令）和 skills（可复用能力包）两套机制。

<figure>

</figure>

讲到这儿，可能有读者要问了：

> 我用的不是 Claude Code，而是 OpenAI 的 Codex，前面这一通是不是跟我没关系？

也不是。

Codex 那边也有一份自己的「团队约定」，只不过文件名不叫 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>，叫 **<a href="http://AGENTS.md" target="_blank" rel="noopener noreferrer">AGENTS.md</a>**。

它的作用、写法、加载机制跟 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 几乎一模一样。你前面学的所有原则，200 行黄金线、具体可验证、告诉 why、持续更新，一条都不用扔，照搬到 <a href="http://AGENTS.md" target="_blank" rel="noopener noreferrer">AGENTS.md</a> 里就行。

那要是你的项目同时用 Claude Code 和 Codex 呢？两份文件维护成两套，规则一改要改两遍，妥妥的负担。

这里有个特别巧的做法：**把所有规则写在 <a href="http://AGENTS.md" target="_blank" rel="noopener noreferrer">AGENTS.md</a> 里，<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里只留一行**：

```
@AGENTS.md
```

<figure>

</figure>

<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里的 `@文件名` 是个引用指令。Claude Code 启动加载 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 时，看到 `@AGENTS.md`，会顺着这条引用把 <a href="http://AGENTS.md" target="_blank" rel="noopener noreferrer">AGENTS.md</a> 的内容也读进来；Codex 那边本来就直接读 <a href="http://AGENTS.md" target="_blank" rel="noopener noreferrer">AGENTS.md</a>，规则自然也拿到。

一份文件、两个工具、零重复维护。

<figure>

</figure>

讲完跨工具这茬，最后提一个容易被忽略的坑：**规则之间会打架**。

官方文档原话是：「如果两条规则互相矛盾，Claude 可能会随便挑一条。」模型又不是律师，没法判断哪条优先级更高。所以分层之后，得定期 review，把过时的、冲突的规则清掉。我自己的习惯是每 1 到 2 周扫一次。

<figure>

</figure>

### 05｜/init 起步、/memory 维护

讲完 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 怎么写、怎么分层，最后一个绕不开的话题是：**怎么把它跑起来**。

<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 并不是自动创建的，而是需要我们自己手动创建的。

如果你项目里压根还没 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>，第一步是什么？答案是 `/init`。

在 Claude Code 里输入 `/init`，Claude 会自动扫一遍你的代码库，把分析出来的技术栈、目录结构、常用命令起个草稿。Anthropic 官方文档里有句话：「五分钟时间，永久受益。」

<figure>

</figure>

我实测过，`/init` 起的草稿质量出乎意料地好。当然不完美，你得 review 一遍删掉不准的、补上漏掉的，但起点已经比从空文件开始高出几个台阶。

<figure>

</figure>

项目跑起来之后，规则怎么补充？最经典的工作流是这样：**Claude 在哪儿犯错了，就加一条防御规则。**

但你不需要手动打开 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 编辑，Claude Code 提供了 `/memory` 命令。

<figure>

</figure>

session 中途想加规则，直接输入 `/memory` 会弹出 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 让你直接改。或者你跟 Claude 说一句「记一下这条规则」，它会自动追加到合适的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 文件里去。

<figure>

</figure>

<a href="http://claudeguide.io" target="_blank" rel="noopener noreferrer">claudeguide.io</a> 给了一个特别实用的规则触发标准：**Claude 错两次以上，就加一条新规则。** 一次可能是偶发，两次说明规则有缺。再不写就会一直被它坑。

<figure>

</figure>

还有一个常被忽略的命令配合：**Plan Mode**。

复杂任务的时候，按 Shift+Tab 两次切到 Plan Mode，Claude 不直接动手写代码，而是先出一份计划给你看，确认了再执行。

<figure>

</figure>

为啥这玩意儿要跟 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 配合讲？因为 Plan Mode 出计划的时候，会把你 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 里的规则全考虑进去。**一份好的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 直接决定了计划的质量，计划出得好不好，决定了最终代码写得对不对。**

Claude Code 官方一直在推 Plan Mode 的用法，社区里也基本形成了共识：动手写代码之前先切 Plan Mode，尤其是改动跨多个文件的时候。

我自己实测下来，对 3 个文件以上的改动，Plan Mode 配合 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 这套组合质量提升肉眼可见。

<figure>

</figure>

### 06｜可以参考的模板

讲了这么多原则和反例，最后给你一份可以参考的模板：80 行以内、6 段式结构、每段都加了点注释。

这套结构参考了 <a href="http://claude-codex.fr" target="_blank" rel="noopener noreferrer">claude-codex.fr</a> 技术博客里的 6 段式建议，然后稍微做了精简：

```
# CLAUDE.md

## 1. Project Overview
（2-3 行讲清这是个啥项目，技术栈 + 定位）
- 这是一个面向 B 端的订单管理系统
- 技术栈：TypeScript + Next.js 14 + PostgreSQL
- 部署：Vercel + Supabase

## 2. Commands
（最常用的几个命令，Claude 会直接执行）
- 安装依赖：`pnpm install`
- 启动开发：`pnpm dev`
- 跑测试：`pnpm test`
- 类型检查：`pnpm typecheck`
- Lint：`pnpm lint`

## 3. Architecture
（三句话讲完架构，不要展开）
- 前端页面在 app/（App Router）
- API 路由在 app/api/
- 数据库 schema 在 prisma/schema.prisma
- 详细架构见 docs/architecture.md

## 4. Conventions
（团队真实在用的约定）
- 组件文件用 PascalCase（UserCard.tsx）
- 工具函数用 kebab-case（format-date.ts）
- API 返回统一用 { data, error } 格式
- 错误处理用 Result type，不要 throw

## 5. Hard Constraints
（这部分要严，Claude 越界一次就要补）
- 不要写入 production 数据库（去年事故）
- 不要修改 prisma/migrations/ 下已经合入的 migration
- 不要把 .env 文件加入 git
- 所有 API 路由必须过 requireAuth() middleware

## 6. Gotchas
（每个新人都踩过的坑）
- 跑 dev 之前要先 pnpm db:push 同步 schema
- macOS 上 Prisma 偶发崩溃，重启 dev server 就好
- Vercel 部署日志在 dashboard 里看，不在终端
```

<figure>

</figure>

这份模板里有几个细节值得留意。

第一，**总行数 50 行左右**，远低于 200 黄金线，给后续加规则留了空间。

第二，**Architecture 段故意写得短**，只指住址不复述详情，避开第二节讲的「复述型」陷阱。

第三，**Hard Constraints 写了 why**（「去年事故」），呼应第三节的「告诉为什么」原则。

第四，**Gotchas 部分价值最高**，因为这些坑都是踩出来的经验，Claude 没法从代码里推断。

你照这份模板改，**别从头复制**。先抄结构，再填你项目的内容。复制完整内容只会让你换了个壳，规则还是不准。

<figure>

</figure>

### 收尾：3 句话精华

文章讲了一堆，咱们最后做个总结。

如果你只能从这篇文章带走三句话，那就是这三句：

第一，**<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 是给 Agent 的入职手册，不是给人的 README**。写之前先问自己：这句话是给人看的，还是给 Claude 看的？给人看的留给 README。

第二，**200 行是黄金线，每行都吃 token，多写不如不写**。复述型、愿望型、术语表型这三类内容直接删，瘦下来 Claude 反而更听话。

第三，**具体可验证、告诉 why、持续更新，三条铁律压过一切技巧**。哪条规则都别忘了这三个核心。

<figure>

</figure>

如果你面试被问到对 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 的理解，可以这么答：

> 「<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 每次启动都会被完整加载进上下文，规则一多反而稀释模型注意力。社区实测数据是 200 行 92% 遵守率，400 行掉到 70%。我的做法是项目根 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 控制在 80 行以内，按模块拆到 `.claude/rules/` 下用 path-scoped 加载，配合 `/init` 起步和 `/memory` 维护，规则遵守率明显上来了。」

如果这篇文章对你有帮助，记得点个赞、在看、转发三连，感谢林友们的支持！

我们下篇见啦。

#### 参考资料

- Anthropic 官方文档：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 使用指南，<a href="https://docs.anthropic.com/en/docs/claude-code/claude-md" target="_blank" rel="noopener noreferrer">https://docs.anthropic.com/en/docs/claude-code/claude-md</a>

- Anthropic Help Center：Give Claude context with <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> and better prompts，<a href="https://support.claude.com/en/articles/14553240" target="_blank" rel="noopener noreferrer">https://support.claude.com/en/articles/14553240</a>

- <a href="http://claudeguide.io" target="_blank" rel="noopener noreferrer">claudeguide.io</a>：How to Write Effective <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> Files (With 12 Real Examples)，<a href="https://claudeguide.io/claude-md-effective-patterns" target="_blank" rel="noopener noreferrer">https://claudeguide.io/claude-md-effective-patterns</a>

- <a href="http://claude-codex.fr" target="_blank" rel="noopener noreferrer">claude-codex.fr</a>：Mastering <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>（6 段式结构推荐来源），<a href="https://claude-codex.fr/en/prompting/claude-md/" target="_blank" rel="noopener noreferrer">https://claude-codex.fr/en/prompting/claude-md/</a>

- SFEIR Institute：The <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> Memory System Deep Dive（200 行 92%、模块化 96% 实测数据来源），<a href="https://institute.sfeir.com/en/claude-code/claude-code-memory-system-claude-md/deep-dive/" target="_blank" rel="noopener noreferrer">https://institute.sfeir.com/en/claude-code/claude-code-memory-system-claude-md/deep-dive/</a>

## Claude Code 大型代码库实战：百万行代码怎么扛得住？

> Source: https://xiaolinnote.com/claudecode/playbook/cc_large_codebase.html

大家好，我是小林。

平时用 Claude Code 都是在自己的小项目上跑，舒坦得很。

可一旦放到「公司百万行级别的大代码库」这个场景下，所有问题立刻浮出来。

而这些问题，恰恰是 Anthropic 自己每天在解决的。

为了搞清楚官方到底是怎么应对的，我把 Anthropic 上周刚发的那篇专门讲大代码库实践经验的博客整篇扒了一遍，又翻了 Claude Code 创始人 Boris Cherny 分享过过的一些经验。

把这些一手信源串完之后，我把在大代码库里最容易踩的 7 个坑总结了出来：

- Q1：大代码库下 context 老爆，是不是模型太小了？
- Q2：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 到底写多长合适？写了 1000 行 Claude 反而变笨？
- Q3：大代码库里让 Claude 找一个函数，总找错文件，怎么办？
- Q4：跨几十个文件的改动，Claude 总是改一半就崩，怎么救？
- Q5：团队里只有我一个人会用 Claude Code，怎么推广？
- Q6：Claude Code 创始人平时怎么用 Claude Code？
- Q7：什么样的项目其实不适合用 Claude Code？

我们一个一个来说。

------------------------------------------------------------------------

### Q1：大代码库下 context 老爆，是不是模型太小了？

不少人的第一反应是这个：「context 不够，那是不是该换更大模型？」

**Anthropic 官方答案是：换模型没用，问题不在模型，在 Claude Code 怎么找代码。**

你想啊，Opus 4.7 已经支持 1M token 了，换算下来两百多万字。但一个像样点的项目动辄几百万行代码，再算上依赖库就更夸张了。再大的窗口也塞不下整个代码库，这是物理上的事。

<figure>

</figure>

那 Claude Code 在大代码库下怎么解决「精准找到要改的那几行代码」的？

业内的主流答案是 RAG：把代码切片、做 embedding、塞向量数据库，要查的时候用相似度召回。Cursor、Copilot、Windsurf 走的都是这条路。

但 Claude Code 偏偏不走。它连 embedding 和向量数据库的影子都没有，就靠 grep、读文件、看目录这种最朴素的方式。

<figure>

</figure>

Anthropic 给这套办法起了个名字，叫 agentic search，翻译过来就是「让 Claude 像 agent 一样去搜」。

Claude 像一个真人工程师一样：先 `ls` 看根目录、再进 `auth/` 看里面有啥、grep 一下「login」找到相关函数、再读 `middleware.ts` 和 `session.ts`，读一个文件决定下一步读什么，循环往复。

```
flowchart LR
    A[收到任务] --> B[看目录]
    B --> C[grep 找关键字]
    C --> D[读相关文件]
    D --> E{够了吗?}
    E -->|不够| C
    E -->|够了| F[动手改]
```

<figure>

</figure>

为什么 Anthropic 选这个反主流的路线？官方博客给了三个理由。

第一，**索引会过期**。千人团队每天提交几百个 commit，embedding pipeline 根本跟不上。等你查的时候，索引里返回的可能是两周前已经被重命名的函数。Claude 拿着过期信息推理，代码自然就崩了。agentic search 每次都基于当下的代码，没有这个问题。

第二，**冷启动几乎为零**。RAG 在百万行代码库上建一次索引要十几分钟，Claude Code 是「打开就能用」。

第三，**精确匹配向量干不了**。你说「帮我看下 getUserById」，向量召回会返回 getUserByName、getUserByEmail、fetchUserInfo 一堆「相关」函数。代码很多时候要的就是精确，不是相似。

那 agentic search 的代价是什么？

Anthropic 在博客里有一句关键的原话：**它严重依赖一个好的起点 context**。如果你不给它清晰的起点，它就会乱翻，等摸清楚结构 context 已经被烧得差不多。

所以 context 爆不是模型小，是你没给 Claude 一个好的起点。下面 6 个问题，就是在解决这件事。

但在拆这 6 个问题之前，得先建立一个核心概念，因为它是后面所有答案的总纲。

这个概念叫 **harness**。

很多人讨论 Claude Code 强不强的时候，第一反应是看模型：「我用 Sonnet 4.6 还是 Opus 4.7？」「benchmark 哪个分高？」「要不要升 Max 套餐？」

但 Anthropic 在博客里抛了一个挺反直觉的论点，原话叫「**The harness matters as much as the model**」，翻译过来就是 **harness 跟模型一样重要**。

什么意思？

Anthropic 说，大家评估 Claude Code 时都盯着 benchmark 看模型表现，但**在实际生产中，围绕模型搭的那套外壳对最终效果的影响，比模型本身还大**。

打个比方。你请了个米其林三星大厨到家里给你做饭，他厉不厉害是模型能力；但你家里有没有趁手的灶台、菜刀、调料架、抽油烟机，这才是 harness。灶台不行，再牛的厨师也炒不出锅气。

<figure>

</figure>

Anthropic 的 harness 一共七层，每层都建立在前一层基础上：**<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> → Hooks → Skills → Plugins → MCP**，再加两个增强 **LSP 和子 agent**。

<figure>

</figure>

听着多？其实下面几个 Q 就是按官方顺序一层一层把它们拆透：

- Q2 拆 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 怎么写（含 Hooks 怎么挂）
- Q3 拆 LSP 和子目录启动
- Q4 拆子 agent 怎么和主 agent 协作
- Q5 拆 Skill、Plugin、MCP 怎么打包分发给团队
- Q6 看创始人 Boris 怎么把这七样东西组合起来用

读完你就明白，**用好 Claude Code 不是搞定模型选型，而是把这套 harness 一层一层搭起来**。

**context 爆不是模型小，是你的 harness 没搭好。**

------------------------------------------------------------------------

Q2：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md 到底写多长合适？写了 1000 行 Claude 反而变笨？</a>

那我们就从 harness 第一层开始拆，也就是 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>。

这一层是大代码库下踩坑最多的一个。

**Anthropic 官方答案非常具体：单文件控制在 200 行以内**。

听起来是不是有点吃惊？毕竟一个项目的规范规则随便列列就上千行了。

官方的逻辑也很简单：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 每次启动都被整个塞进 context，写太长就等于在跟自己抢空间。超过 200 行之后，Claude 开始忽略指令的概率会肉眼可见上升。

那大代码库下规则确实多怎么办？关键词是**分层**。

Anthropic 在博客里有句原话挺狠的：「根目录的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 应该只放指针和关键的坑，其他细节都会变成噪音。」

正确做法是 root 文件只放跨包通用约定（比如「生产数据库千万别动」「提 PR 前要跑 lint」），每个子目录再放自己的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 写模块细节。Claude 会自动从当前目录往上走树把沿途每个 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 都加载进来。

<figure>

</figure>

但这还不够。Claude Code 创始人 Boris 还为 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 维护放过一句口号当 slogan：「Ruthlessly edit your <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> over time」，翻译过来就是**对你的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 下狠手，毫不留情地删**。

怎么判断 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 该不该留某一行？有个特别实用的检查法：对每一行你都问自己「如果删掉这行，Claude 还会按这条规则做事吗？」答案是「会」（常识或代码已经体现），就该删；答案是「不会」才值得留。

<figure>

</figure>

任何时候你发现 Claude 还在反复犯某个错，**先别急着加新规则，先去看看 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 是不是已经太长把规则淹没了**。

<figure>

</figure>

Boris 还分享过 Anthropic 内部团队怎么维护这份文件：整个 Claude Code 团队共享一份 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 提交到 git，**一旦发现 Claude 做错了什么就立刻加进 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>**。这份文件在他们那里不是「写一次放着」的文档，而是持续打磨的活文件。

还有一条 Anthropic 官方建议特别容易被忽略：**每 3-6 个月对你的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 做一次完整审查**。

为什么？因为模型在进化。

你三个月前为了约束 Claude 写的「每次重构只改一个文件」，可能在新模型上反而变成了枷锁，新模型已经能做跨文件协调编辑了，旧规则反而把它捆住了。同样，为了弥补旧模型某个弱点写的 Hook、Skill，模型升级之后可能直接成多余负担。

说白了，模型都已经往前跑了，你的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 可能还停在三个月前。

如果你感觉 Claude Code 最近用得怎么都上不去一个台阶，**先别怀疑模型，先回去看你的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 是不是过期了**。

总结一下 Q2 的官方答案：单文件 200 行以内、分层加载、持续狠删、每 3-6 个月审查一次。

不过你可能会问：「我哪有时间天天盯着 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 改？」

官方对这个问题也有解法，叫 **Hooks**。

Hooks 是 Claude Code 的事件钩子机制，在「编辑完文件之后」「会话开始之前」「工具调用之前」这些时间点上挂脚本做事。

大多数人对 hook 的认知停留在「防止 Claude 做错事」，比如挂一个 hook 自动跑 lint、自动 format。

这没毛病，但官方点出来一个反直觉的洞察：**hook 真正的价值不是阻止 Claude 做错事，而是让你的整套设置自我进化**。

举个例子。

挂一个 Stop hook，在每次会话结束时让它自动反思「这次 Claude 有没有什么常犯的错误？要不要写进 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>？」然后 hook 自己改 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>。

或者挂一个 Start hook，根据你当前所在子目录动态加载这个模块特有的 context，今天在 `payments/` 下就自动拉支付 skill，明天换到 `auth/` 下就换成认证相关。

这样一来，**你的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 是被 Claude 自己持续打磨的，不再需要你手动维护**。Boris 自己挂了一个 PostToolUse hook 给 Claude 写完的代码自动跑格式化，把偶尔遗漏的 10% 格式问题直接抹平。

<figure>

</figure>

**<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 不是写一次的文档，是一份持续打磨的活文件。**

------------------------------------------------------------------------

### Q3：大代码库里让 Claude 找一个函数，总找错文件，怎么办？

<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 这层搞定之后，Claude 知道了「这个项目长啥样」。但接下来还有个更细节的问题：让它找一个具体函数，它老是找错文件。

这个问题在多语言大代码库（C/C++/Java/PHP 这种符号歧义高的语言）里特别突出。

**Anthropic 官方答案是两件事：装 LSP + 在子目录里启动 Claude。**

先说 LSP。

LSP 全称叫 Language Server Protocol。听着挺唬人，但其实你天天都在用：平时你在 VS Code 里点「go to definition」「find references」，背后跑的就是它。

Claude Code 接上 LSP 之后，搜代码就不再是按字符串 grep，而是按**符号**搜。

举个例子。你在大代码库里 grep 一个 `getUser` 函数，可能返回三千个匹配，前端有、后端有、测试也有。Claude 得一个个读文件判断哪个是你真要改的，光这个过程就能把 context 烧光。

但有 LSP 之后，Claude 直接问 LSP：「找跟 `auth/login.ts` 那个 getUser 同源的所有引用」。LSP 一口气返回精确的三个，过滤工作在 Claude 读文件之前就完成了。

<figure>

</figure>

Anthropic 官方博客直接把 LSP 称作多语言大代码库下「one of the highest-value investments」，并讲了一个真实案例：有家做企业软件的公司，在全公司铺 Claude Code 之前专门先把 LSP 集成在组织级别铺开，就是为了让 C 和 C++ 这种符号歧义高得离谱的语言能跟 Claude 配合得动。

装 LSP 怎么操作？在 Claude Code 的 `/plugin` 里搜「lsp」，找到对应语言的 code intelligence plugin（`typescript-lsp` / `pyright-lsp` / `rust-analyzer-lsp` 等等）装上，再装对应的语言服务器二进制（pip 装 pyright、npm 装 typescript-language-server 之类）。整个过程不超过两分钟。

<figure>

</figure>

再说子目录启动这件事，这是反直觉但官方博客被反复强调的一条。

大多数人第一次用 Claude Code，习惯都是 `cd` 到项目根目录然后 `claude`。在小项目没毛病，但在大代码库里，这会让 Claude 一上来就把根目录那个超大的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 全部加载进 context，前端后端 infra 所有微服务的规则全来一遍。

官方博客原话叫「Initializing in subdirectories, not at the repo root」。

正确做法是直接在你要改的子目录启动。比如要改支付服务，就 `cd services/payments` 然后 `claude`。Claude 会自动往上走树把根目录的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 也加载进来，通用规则不丢；但优先加载 `payments/` 子目录的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>，context 立刻聚焦到「支付」一个领域。

<figure>

</figure>

除了 LSP 和子目录启动，官方博客还提了三个小细节，配合起来效果更好：

第一，**测试和 lint 命令按子目录写进 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>**。Claude 改了支付服务里一个文件，结果它跑整个项目的测试套件，几十分钟才出结果，context 也跟着烧光。每个子目录的 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 应该明确写「这块用什么命令测，怎么 lint」，让 Claude 只跑该跑的那一部分。

第二，**用 `.ignore` 规则把生成文件、构建产物、第三方代码排除掉**。把 `permissions.deny` 规则提交到 `.claude/settings.json`，整个团队就能自动共享这些排除规则，不用每个人手动配。

第三，**目录结构不直观时，在根目录放一张「代码库地图」**。一份简单的 markdown 文件，列出每个顶层文件夹的一句话说明就够。Claude 在动手探索之前先扫一眼这张地图，比让它瞎翻一通要快得多。

**让 Claude 按符号搜代码、按子目录工作，准确率立刻翻倍。**

------------------------------------------------------------------------

### Q4：跨几十个文件的改动，Claude 总是改一半就崩，怎么救？

Claude 知道了项目结构、也能找准代码了。这下总能干大活了吧？

还真不一定。重构、迁移、跨服务联动这种「大动作」上，Claude 经常前半段还在状态，后半段就开始忘前面、漏改、改错。这是大代码库下另一个高频翻车点。

**Anthropic 官方答案：跨大量文件的改动，正确解法是把任务拆成多个会话 + 用 subagent，不是写更长的 prompt**。

很多人第一反应是改 prompt、改 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>、加更多规则。

但 Anthropic 在博客里明说过：跨大量文件的改动，**正确的解法是把任务拆成多个会话**。

Boris 还把这句话翻译得更直白：「Pour your effort into the plan so Claude can one-shot the implementation」，意思是与其用一个超长 prompt 让 Claude 一次搞定所有事，不如先单独花一轮把方案敲定，再分多个会话去实现。

具体怎么做？

**第一步：派 subagent 出去探索，主 agent 留着干净的 context。**

大代码库下「读懂这个系统怎么工作」本身就要烧掉好几万 token。让 Claude 一边读代码一边改代码，相当于让一个人一边查资料一边写论文。

Subagent 的思路特别简单：派一个小弟去探索，让他写一份 findings 报告回来，主 agent 看完报告再动手。小弟在独立的 context 窗口里跑，读了几十个文件烧的是自己的 context，跟主 agent 没关系。他最后只把几百字摘要给主 agent。

```
flowchart TB
    A[主 Agent 接到任务] --> B[派 Subagent 去探索]
    B --> C[Subagent 读 50 个文件烧自己的 context]
    C --> D[Subagent 写 200 字 findings 报告]
    D --> E[主 Agent 拿到摘要]
    E --> F[主 Agent 在干净 context 下动手改]
```

<figure>

</figure>

最简单的操作就是直接跟 Claude 说：「先用 subagent 调查一下我们项目里 X 是怎么实现的，写成 findings 文件，再回来动手改。」

**第二步：会话拆分。**

会话 1 只做探索写 plan 不动代码；会话 2 加载 plan 实现一个模块跑通测试；会话 3 实现下一个模块。每个会话都从干净 context 开始，plan 文件做桥梁串联。

<figure>

</figure>

**第三步：跑大型迁移用 `/batch`。**

如果你的改动是「整个项目从一个框架迁到另一个」「把几十个文件的某种调用全部替换」这种大规模迁移，Claude Code 已经直接内置了一个专门工具叫 `/batch`。

用法是这样的：先用对话方式把迁移方案敲定，然后它一次性派出几十个并行 subagent，每个在独立 git worktree 里跑、自测、开 PR。

你不用守屏幕，跑完直接给你一堆 PR 等 review。

<figure>

</figure>

这就是创始人 Boris 本人正在用的工作流，以前要自己手撸的多 agent 编排，现在一行命令就搞定。

**跨大文件改动救不回来的不是 prompt，是会话边界。**

------------------------------------------------------------------------

### Q5：团队里只有我一个人会用 Claude Code，怎么推广？

前面 3 个 Q 解决的都是「你自己一个人怎么把 Claude Code 用顺」。但接下来这个问题就升级了：你用得飞起，旁边的同事还在用 demo 版，怎么办？

这是个组织层面的问题，也是 Anthropic 官方博客花了不少篇幅讲的一块。

**Anthropic 官方答案：先把好实践做成 skill，再用 plugin 打包分发出去，再用 MCP 把团队内部系统接进来，最后得有人维护这套东西。**

听着有点多？我们一步一步来。

#### 第一步：先把高频操作做成 skill

什么是 skill？你可以理解成「针对某个具体任务的 SOP」。比如「这个项目的数据库迁移怎么做」「这个微服务上线的标准流程」，这些都是 skill 该干的事。

Skill 跟 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 最大的区别在一个词：**按需加载**。

<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 每次会话都全文加载，跟你这次任务有没有关系都加载；skill 不是，它只在 Claude 判断「当前任务需要」的时候才加载，平时静静躺在仓库里不占 context。官方有个专门的词叫 progressive disclosure（渐进式披露），讲的就是这个机制。

<figure>

</figure>

Boris 还说过一句话特别值得记下来：「如果一件事你一天做超过一次，就把它做成 skill。」一个大项目里高频操作就那么几十种，每个都做成 skill 全队共享，效率立刻是几何级提升。

skill 还可以绑定到特定路径。「支付服务部署 skill」绑定到 `services/payments/`，只有 Claude 在这个目录下工作时才加载，避免「改前端代码结果支付 skill 也来凑热闹」这种 context 污染。

#### 第二步：用 plugin 把好实践打包分发

但 skill 本身还在每个人的本地，没法共享。这就引出了 plugin。

大公司里有个经典问题：好的工具配置永远只在小圈子里流传。某个高级工程师本机配置了三十个 skill、十几个 hook、五个 MCP server，他用 Claude Code 爽得飞起。但旁边的实习生啥都没配，体验就跟用了个 demo 版差不多。

Plugin 就是解决这个问题的。它本质上是一个安装包，把 skill、hook、MCP、LSP 配置打包在一起。新人入职第一天 install 一下，立刻和团队所有人有一样的 Claude Code 能力。

官方博客讲过一个特别接地气的案例：一家大型零售公司搭了个 skill 让 Claude 连内部数据分析平台，业务分析师不用切工具就能拉销售数据。这个 skill 起初只是少数人的本地配置，后来打包成 plugin 全公司铺开，整个公司业务分析效率被拉高一个档次。

<figure>

</figure>

公司还可以建自己的 plugin marketplace。谁有更好的实践就更新到 marketplace 里，全公司一起受益。

#### 第三步：用 MCP 把团队内部系统接进来

光有 skill 和 plugin 还不够。

大代码库下的工作往往不是孤立的，得跟团队的 Slack、Jira、内部 wiki、数据库、监控系统都联动。

这个连接的桥梁叫 **MCP server**（Model Context Protocol）。

装一个 Slack MCP，Claude 就能搜公司 Slack 消息；装一个 BigQuery MCP，它就能跑数据查询；装一个 Sentry MCP，它就能拉线上错误日志。

听着很强，但官方在这块特别提醒了一个反直觉的点：**别太早上 MCP**。

很多团队 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 都还没写好，hook 也没挂，就着急忙慌接各种 MCP，结果反而把 context 搞得更乱。MCP 是 harness 里最后才该上的一层，前面的基础没搭好，MCP 接进来的数据就是噪音。

正确的顺序是：先把 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 和 skill 打磨好 → 再用 plugin 打包分发 → 最后才上 MCP 把外部世界接进来。

<figure>

</figure>

#### 第四步：得有人负责维护

但官方还点出来一个更关键的事：**光把工具堆起来不够，得有人负责维护**。

Anthropic 观察到，推广最顺的组织都有一个共同点：在大面积铺开之前，会先安排一小队人（甚至一两个人）把整套基础设施搭好，然后才放开访问。

开发者第一次摸 Claude Code 就能跑通，**第一印象如果是「这东西不好使」，后面要翻盘就太难了**。

官方博客里点出了一个正在浮现的新角色，叫 **Agent Manager**，半 PM 半工程师，专门负责 plugin 分发、<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 规范、skill 审批这些事。

规模小一些的团队没条件设这个岗位也没关系，至少要有一个 DRI（直接责任人）把 Claude Code 的配置维护起来，有拍板权决定哪些 skill / plugin 上、哪些不上。

没有人盯着这件事，再好的 plugin 也会变成「张三两年前搭的，没人会改」的部落知识。

**好实践不再是个人玩具，而是组织资产。**

------------------------------------------------------------------------

### Q6：Boris 自己平时怎么用 Claude Code？

前面 4 个 Q 把官方答案讲完了，你可能会好奇：那 Claude Code 的创始人自己平时是怎么用的？

这一节其实是个彩蛋，但读完你会发现，里面藏着创始人对 Claude Code 用法的全部理解。

Boris Cherny 是 Claude Code 的创始人，他分享过一段让我看完直接破防的话：

「我同时在终端里跑 5 个 Claude，再加 5 到 10 个跑在 <a href="http://claude.ai/code" target="_blank" rel="noopener noreferrer">claude.ai/code</a> 上，并行处理不同任务。」

<figure>

</figure>

听着是不是有点不可思议？

但他的这套 setup 其实很值得拆解，里面藏着创始人对 Claude Code 用法的全部理解：

**第一，他不用 `--dangerously-skip-permissions`**。他明确说过自己用 `/permissions` 命令把常用的安全命令预先加白名单，避免一遍遍点确认，但又不放弃权限审计。

**第二，他几乎所有复杂任务都从 Plan Mode 开始**。先跟 Claude 把方案敲定，再切到 auto-accept 模式让它一发命中地把代码写出来。

**第三，他挂了一个 PostToolUse hook 给 Claude 写完的代码自动跑格式化**，把 Claude 偶尔遗漏的 10% 格式问题直接抹平，避免后面 CI 挂掉。

**第四，他把每天做超过一次的事都做成了 slash command 或 skill**。Boris 有句名言：「如果一件事你一天做超过一次，就把它做成 skill。」他自己有个 `/commit-push-pr` 命令，一天用几十次，避免重复 prompt。

**第五，他给整个 Claude Code 团队共享一份 <a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a>，提交到 git**。一旦发现 Claude 做错了什么就立刻加进去，是一份持续打磨的活文件。

把这 5 件事串起来看你会发现：创始人对 Claude Code 的态度不是「装上就用」，而是**把它当成一个会进化的工作伙伴，每天都在喂它新规则、新工具、新工作流**。

这才是大代码库下用好 Claude Code 的底层心态。

**创始人对 Claude Code 的态度，不是「装上就用」，而是「每天打磨它」。**

------------------------------------------------------------------------

### Q7：什么样的项目其实不适合用 Claude Code？

讲了这么多 Claude Code 在大代码库里有多能打，最后还得给你泼一盆冷水：**它也不是万能药**。

这是最后一个问题，也是 Anthropic 官方博客说得最坦诚的一块。

官方原话是这样的：「Claude Code 是围绕传统软件工程环境设计的，假设工程师是代码库的主要贡献者，仓库用 Git，代码遵循标准目录结构。」

也就是说，下面这几种场景 Claude Code 用起来会比较吃力：

- **游戏引擎那种大量二进制资源的项目**：Claude 没法读你的 3D 模型、贴图、音频
- **用非常规版本控制系统的项目**：比如老牌的 Perforce / Subversion / 自研 VCS，需要额外配置才能跑顺
- **非工程师为主贡献的代码库**：比如产品经理改产品文档、设计师改 Figma 配置文件，这些场景 Claude Code 的 harness 不太对得上

官方在博客结尾建议这种非常规场景需要更多定制化配置，他们的 Applied AI 团队会专门跟客户对接。换句话说，**Claude Code 当下最擅长的还是「Git + 工程师 + 标准目录」这个最大公约数**。

如果你的项目正好踩在这几个非常规场景上，别死磕，找官方支持渠道才是正解。

<figure>

</figure>

**Claude Code 不是万能药，最擅长的是「Git + 工程师 + 标准目录」这个最大公约数。**

------------------------------------------------------------------------

### 最后

到这里，7 个问题的官方答案就说完了，我把这 7 个答案浓缩成 3 句话送你：

- 第一，Claude Code 在大代码库不是「装上就能用」，是要在 harness（外围基建）上花一次性功夫的。

- 第二，最高 ROI 的三个动作是：<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 砍到 200 行以内 + 在子目录启动 Claude + 装 LSP。这三件事做完，体验立刻不一样。

- 第三，跨大文件改动、团队推广、<a href="http://CLAUDE.md" target="_blank" rel="noopener noreferrer">CLAUDE.md</a> 维护这些大代码库下的硬骨头，官方都给了具体答案，Boris 自己也在用，你照抄就行。

现在你可以打开你公司的项目，对照这 7 个问题逐一过一遍，看看哪几个你已经做对了，哪几个还差一截。

------------------------------------------------------------------------

### 参考资料

- Anthropic 博客《How Claude Code works in large codebases: Best practices and where to start》：<a href="https://claude.com/blog/how-claude-code-works-in-large-codebases-best-practices-and-where-to-start" target="_blank" rel="noopener noreferrer">https://claude.com/blog/how-claude-code-works-in-large-codebases-best-practices-and-where-to-start</a>

## Claude Code Skill 揭秘：Skill 真的只是一份 markdown 吗？

> Source: https://xiaolinnote.com/claudecode/playbook/cc_skills.html

大家好，我是小林。

不知道你有没有这种体验：兴冲冲给 Claude Code 装了一堆 skill，或者自己照着教程写了几个，结果用了一个月，Claude 压根没主动调用过几次。

skill 就这么安静地躺在目录里，像极了你收藏夹里那些「以后一定看」的文章。

问题出在哪？是 skill 这个机制不行吗？

还真不是。

Anthropic 内部光是活跃使用的 skill 就有几百个，他们前几天刚发了一篇博客，把这几百个 skill 沉淀下来的经验一次性全交代了：什么样的 skill 值得做、怎么写 Claude 才会用、团队怎么共享、甚至怎么给 skill 做数据埋点。

<figure>

</figure>

我把这篇博客整篇扒了一遍，整理成 7 个问题：

- Q1：「skill 不就是一份 markdown 吗？」这可能是最大的误解
- Q2：Anthropic 内部几百个 skill，最后只归成了 9 类
- Q3：为什么你写的 skill Claude 从来不触发？
- Q4：一个 skill 里含金量最高的部分，是「坑点清单」
- Q5：skill 还能有记忆、带脚本、挂临时 hook？
- Q6：skill 怎么从你的本地走向全团队？
- Q7：怎么知道一个 skill 到底有没有人用？

我们一个一个来说。

------------------------------------------------------------------------

### Q1：「skill 不就是一份 markdown 吗？」这可能是最大的误解

先抛个问题：如果让你现在描述一下什么是 skill，你会怎么说？

我猜不少人的答案是：「就是一个写了操作步骤的 markdown 文件，Claude 需要的时候会去读。」

Anthropic 在博客里点名说了，这是他们听到的关于 skill 最常见的误解。

那 skill 到底是什么？官方的定义是：**一个文件夹**。里面除了那份 <a href="http://SKILL.md" target="_blank" rel="noopener noreferrer">SKILL.md</a>，还可以放脚本、参考资料、数据文件、输出模板，Claude 能自己发现、探索和使用这些东西。

一个五脏俱全的 skill 长什么样？拿一个部署服务的 skill 举例，典型的目录结构是这样的：

```
deploy-service/
├── SKILL.md               # 唯一必需：何时用我 + 操作指引 + 坑点清单
├── references/            # 参考资料，正文放不下的细节放这里
│   ├── api.md             # 部署平台 API 的详细参数和示例
│   └── troubleshooting.md # 部署失败时的排查手册
├── scripts/               # 现成的可执行脚本
│   ├── smoke_test.sh      # 冒烟测试
│   └── rollback.sh        # 一键回滚
└── assets/                # 输出模板
    └── release_note.md    # 发布报告的固定格式
```

整个文件夹里只有 <a href="http://SKILL.md" target="_blank" rel="noopener noreferrer">SKILL.md</a> 是必需的：文件开头一段 frontmatter 写名字和 description，正文写操作指引。references/、scripts/、assets/ 都是可选的，连名字都不是强制的，按你的需要随便加。

更妙的是，这些子文件**不会一股脑塞给 Claude**，而是它干活干到哪一步、需要什么材料，才自己去文件夹里翻什么。这个机制 Q3 会专门拆。

<figure>

<figcaption>skill 文件夹结构示意图</figcaption>
</figure>

打个比方。markdown 文件版的 skill，相当于你给新同事发了一条微信：「部署流程是先这样再那样」。而文件夹版的 skill，相当于你给他一个工位：桌上有操作手册，抽屉里有现成的工具，墙上还贴着前任留下的「这台打印机会卡纸，要先按两下」的便利贴。

哪个更能让新同事快速干活，不用我多说了吧。

而且在 Claude Code 里，skill 还有一堆配置项可以玩，比如绑定特定的触发条件、注册只在 skill 运行期间生效的动态 hook（这个 Q5 细说）。

官方观察下来，**内部效果最好的那批 skill，恰恰都是把文件夹结构和配置项用足了的**。只写一份 markdown 的 skill，相当于只用了这个机制十分之一的能力。

<figure>

<figcaption>微信消息 vs 完整工位对比图</figcaption>
</figure>

所以从今天起，请把对 skill 的认知从「一份说明文档」升级成「一个装备齐全的工具箱」。后面 6 个问题，全部建立在这个认知之上。

**skill 是文件夹，不是文件。这是用好它的第一步。**

------------------------------------------------------------------------

### Q2：Anthropic 内部几百个 skill，最后只归成了 9 类

认知摆正了，下一个问题马上就来：到底什么样的事值得做成 skill？

这个问题特别实际。skill 写起来不难，难的是不知道往哪个方向使劲，写了一堆没用的，真正高频的痛点反而没覆盖。

Anthropic 干了一件很值钱的事：他们把内部几百个 skill 全部拉出来做了一次归类，发现这些 skill 自然聚成了 9 类。

| 类别 | 干什么的 | 例子 |
|----|----|----|
| 库和 API 参考 | 教 Claude 正确使用某个内部库或 CLI | 内部计费库的边界情况和坑 |
| 产品验证 | 教 Claude 怎么测试自己写的代码 | 用无头浏览器跑通注册流程并逐步断言 |
| 数据查询分析 | 连接数据和监控系统 | 该 join 哪些表才能看到转化漏斗 |
| 业务流程自动化 | 把重复工作流压成一条命令 | 自动聚合工单和 PR 生成站会日报 |
| 代码脚手架 | 按团队规范生成样板代码 | 新建一个预接好鉴权和日志的内部应用 |
| 代码质量与审查 | 在组织内强制执行代码质量 | 派一个全新视角的子 agent 做对抗式审查 |
| CI/CD 与部署 | 拉取、推送、部署代码 | 盯着 PR 重试不稳定的 CI、解决冲突 |
| Runbook 排障手册 | 从一个报警症状出发做多工具排查 | 给一个请求 ID，把所有系统的相关日志拉齐 |
| 基础设施运维 | 带护栏的例行维护操作 | 清理孤儿资源前先发 Slack 等人工确认 |

<figure>

<figcaption>九大类 skill 全景图</figcaption>
</figure>

这张分类表怎么用？官方给了一个判断标准：**最好的 skill 干干净净落在某一类里；那些想一次干太多事、横跨好几类的 skill，反而会把 agent 搞糊涂**。

你可以拿自己的 skill 库对着这 9 类扫一遍，马上能看出两件事：哪些 skill 越界了该拆，哪些类别还是空白该补。

那如果 9 类只能先做一类，从哪类下手？

官方在这里给出了全文最掷地有声的一个结论：**验证类 skill 是内部实测对 Claude 输出质量提升最明显的一类**。原话甚至说到这个程度：值得专门让一个工程师花一整周，什么都不干，就把验证类 skill 打磨到极致。

<figure>

<figcaption>九大类价值排序示意图</figcaption>
</figure>

为什么是验证类？

你想啊，Claude 写代码的能力已经够强了，真正拉开差距的是它**有没有办法确认自己写的东西是对的**。没有验证手段，它就只能「我觉得应该没问题」；有了验证 skill，它能开个无头浏览器把注册、邮箱验证、引导页一步步跑完，每一步都断言状态，甚至录一段视频给你看它到底测了什么。

一个会自己验收的 Claude 和一个只会交作业的 Claude，干活质量完全是两个物种。

**先别急着写一堆 skill，把「让 Claude 自己验证工作成果」这一件事做好，回报最大。**

------------------------------------------------------------------------

### Q3：为什么你写的 skill Claude 从来不触发？

好，分类清楚了，skill 也写了，新问题来了：为什么 Claude 就是不用？

要回答这个问题，得先搞清楚一件事：Claude 是怎么知道「现在该用哪个 skill」的？

肯定有不少人以为，Claude 每次都会把所有 skill 的全文读一遍，然后挑一个合适的。要真是这样，装 50 个 skill，context 当场就被吃光了。

实际的机制聪明得多。我们先讲原理，再看源码。

会话启动的时候，Claude Code 会把所有可用 skill 收集起来，但**只取每个 skill 的名字和 description**，拼成一张清单注入 context。Claude 平时看到的就只有这张「目录页」。等它判断某个任务匹配上了某个 skill，才会发起调用，这时候 <a href="http://SKILL.md" target="_blank" rel="noopener noreferrer">SKILL.md</a> 的全文才被加载进对话。

这套机制有个专门的名字，叫**渐进式披露（Progressive Disclosure）**：平时只给目录，用到了才给正文。

<figure>

</figure>

<figure>

<figcaption>渐进式披露图书馆类比图</figcaption>
</figure>

明白了这个机制，「为什么不触发」的答案就浮出来了：**Claude 决定用不用你的 skill，唯一的依据就是那一行 description**。

它没读过你的正文，不知道你内容写得多用心。description 没写好，正文写出花来也是白搭。

这就是官方在博客里专门强调的一条：**description 不是写给人看的摘要，是写给模型看的触发条件**。「帮助处理数据库相关工作」这种写法就是典型的人类视角摘要；模型视角的写法是「当用户要写数据库迁移、修改表结构、或者遇到 migration 报错时使用」。

<figure>

<figcaption>description 好坏对比图</figcaption>
</figure>

讲完原理，我去源码里取了证，结果发现真实情况比博客说的还要苛刻。

skill 清单注入 context 是有预算的，而且预算紧得吓人：

```
export const SKILL_BUDGET_CONTEXT_PERCENT = 0.01
export const MAX_LISTING_DESC_CHARS = 250
```

这段代码在 `src/tools/SkillTool/prompt.ts`，两个常量翻译成人话：**整张 skill 清单只允许占用 context 窗口的 1%；单个 skill 在清单里的描述最多 250 个字符**。

250 个字符之后会发生什么？源码里写得明明白白：

```
return desc.length > MAX_LISTING_DESC_CHARS
  ? desc.slice(0, MAX_LISTING_DESC_CHARS - 1) + '…'
  : desc
```

同样在 `src/tools/SkillTool/prompt.ts` 里，超出 250 字符的部分直接被砍掉，换成一个省略号。你在 description 第 300 个字符处写的精妙触发条件，模型从头到尾就没见过。

更狠的还在后面。如果你装的 skill 太多，把那 1% 的预算挤爆了，Claude Code 会先按比例压缩所有 description；要是还装不下，就直接降级成**只显示名字、一个字描述都不留**的模式。

<figure>

<figcaption>skill 清单预算挤压示意图</figcaption>
</figure>

这下「装了一堆 skill 反而都不触发」这个怪现象就完全说通了：装得越多，每个 skill 能留在 Claude 眼前的信息就越少，最后大家一起变成一排只有名字的哑巴。

skill 不是收藏品，贵精不贵多。

那调用之后，全文是怎么加载的？源码里这个加载动作是「懒加载」的，只有 skill 真被调用时才执行，而且会在正文前面拼一行很关键的信息：

```
async getPromptForCommand(args, toolUseContext) {
  let finalContent = baseDir
    ? `Base directory for this skill: ${baseDir}\n\n${markdownContent}`
    : markdownContent
```

这段在 `src/skills/loadSkillsDir.ts`，它在 <a href="http://SKILL.md" target="_blank" rel="noopener noreferrer">SKILL.md</a> 全文的最前面加了一句「这个 skill 的根目录在哪里」。

为什么要加这句？这正好解释了 Q1 说的文件夹机制怎么落地：**references/、scripts/ 这些文件夹里的东西，系统从头到尾都不会自动加载**，是 Claude 拿到根目录地址之后，按 <a href="http://SKILL.md" target="_blank" rel="noopener noreferrer">SKILL.md</a> 里的指引，自己用读文件的工具一个个去取的。

<figure>

<figcaption>skill 全文加载时序图</figcaption>
</figure>

所以渐进式披露其实有三层：平时只有 description，调用时才有 <a href="http://SKILL.md" target="_blank" rel="noopener noreferrer">SKILL.md</a> 全文，正文里提到的参考文件等 Claude 真需要了才会去读。一层比一层深，每一层都只在必要时打开。

<figure>

<figcaption>渐进式披露三层结构图</figcaption>
</figure>

**description 的前 250 个字符，决定了你的 skill 是工具还是摆设。**

------------------------------------------------------------------------

### Q4：一个 skill 里含金量最高的部分，是「坑点清单」

触发问题解决了，Claude 终于肯打开你的 skill 了。下一个问题：正文该写什么？

先做个小测试。下面两条内容，哪条更值得写进 skill？

第一条：「写完代码后要运行测试，确保所有用例通过。」

第二条：「subscriptions 表是只追加不修改的，你要找的那行记录是 version 最大的那条，不是 created_at 最新的那条。」

答案是第二条，而且不是「略好一点」，是一个天上一个地下。

第一条犯了官方点名的大忌：**陈述显而易见的事（don't state the obvious）**。Claude 本来就会写代码、本来就会跑测试，你把它默认就会做的事再写一遍，等于往 context 里灌纯噪音，一点增量信息都没有。

官方给的判断标准很直接：如果你的 skill 主要是传授知识，那就只写**能把 Claude 推离默认思路的信息**。

<figure>

<figcaption>显而易见 vs 增量信息筛子示意图</figcaption>
</figure>

他们自己有个现成的例子：官方那个前端设计 skill，整篇没教 Claude 怎么写 CSS（它会），而是专门列了一堆「不要做」：不要张口就用 Inter 字体，不要动不动紫色渐变。全是冲着 Claude 的默认审美去纠偏的。

那第二条强在哪？它属于官方说的**整个 skill 里信号最强的内容：Gotchas，坑点清单**。

什么样的内容算坑点？除了上面那条 subscriptions 表的例子，博客里还给了两个真实例子，感受一下：

「这个字段在 API 网关里叫 @request_id，在计费服务里叫 trace_id，它们是同一个值。」

「staging 环境就算 Stripe 的回调没真正处理，也会返回 200，真实状态要去 payment_events 表里查。」

发现共同点没有？这些信息有一个共同特征：**Claude 靠读代码永远推断不出来，只有踩过坑的人才知道**。这正是它信号强的原因，每一条都在为 Claude 排掉一个它必然会踩的雷。

<figure>

<figcaption>坑点清单价值示意图</figcaption>
</figure>

而且坑点清单不是一次写完的，官方的玩法是**持续攒**：每次 Claude 用这个 skill 又栽进一个新坑，就回头把这个坑补进去。skill 就这样越用越准。

不过正文也不是写得越细越好，这里有个度要把握。官方专门提醒了一个反方向的坑，叫**别把 Claude 锁死在轨道上（avoid railroading）**。

Claude 对指令的服从度是很高的，你把步骤写得太死，它遇到指令没覆盖的情况就容易僵在轨道上硬开，明明该随机应变的地方也不敢变。正确的姿势是：把它需要的信息给足，把怎么走的自由留给它。

<figure>

<figcaption>铁轨 vs 导航对比图</figcaption>
</figure>

**skill 正文的黄金法则：只写 Claude 推断不出来的，删掉它本来就会的。**

------------------------------------------------------------------------

### Q5：skill 还能有记忆、带脚本、挂临时 hook？

把 Q4 的内功练好，你的 skill 已经能打了。这一节说三个高阶玩法，全部来自 Anthropic 内部的实战，一个比一个超出「skill 就是文档」的想象。

#### 玩法一：给 skill 装记忆

先想一个场景：你做了一个自动写站会日报的 skill，今天跑一次，明天跑一次。问题来了，它怎么知道哪些内容昨天已经汇报过了？

每次会话都是新开的，Claude 不记得上一次执行的任何事。难道每天的日报都从头把所有进展再说一遍？

官方的解法很朴素：**让 skill 把执行结果存在自己的文件夹里**。比如日报 skill 维护一个日志文件，每发一次日报就追加一条记录。下次执行时，Claude 先读自己的历史，自然就知道「只汇报昨天之后的增量」。

简单的场景用追加式的文本日志或 JSON 就够了，复杂的甚至可以塞一个 SQLite 数据库进去。

<figure>

<figcaption>skill 记忆机制示意图</figcaption>
</figure>

那这些数据该存在哪？官方专门为这件事准备了一个稳定的数据目录，在 skill 里通过环境变量 CLAUDE_PLUGIN_DATA 就能拿到。

这个目录最大的特点是持久：**plugin 升级换版本都不会被清掉**，只有彻底卸载时才会删除。也就是说，你的 skill 记忆可以放心地活得比 skill 版本更久。

<figure>

<figcaption>两个数据目录生命周期对比图</figcaption>
</figure>

记忆这个思路还有个变种用法：存配置。

有些 skill 第一次用之前，需要先从用户那里要点信息。还是拿日报 skill 举例，它总得知道日报要发到哪个频道吧？这种信息不该写死在 <a href="http://SKILL.md" target="_blank" rel="noopener noreferrer">SKILL.md</a> 里（不然没法分发给别人用），也不该每次执行都问一遍（烦死人）。

官方给的模式是：把这类信息存进 skill 目录下的一个 config.json。Claude 每次执行先看配置在不在，在就直接用；不在就说明是第一次跑，主动找用户把信息问齐、写进配置，下次就不用再问了。

相当于给 skill 加了一个「首次使用引导」。要是想问得更体面，还可以在 skill 里指明让 Claude 用选择题的形式来收集，用户点一下就配置完了。

<figure>

<figcaption>skill 首次配置流程图</figcaption>
</figure>

#### 玩法二：把脚本喂给 Claude，让它只管编排

官方有一句话我特别认同：你能给 Claude 最有力的工具就是代码。

什么意思？假设你的数据分析 skill 里什么都不放，Claude 每次分析都要现场手写「怎么连数据源、怎么拼查询、怎么算留存」这一整套样板代码，又慢又容易错。

但如果 skill 里预先放好一个函数库，取数、清洗、对比这些底层活全部封装成现成的函数，Claude 的每一个回合就都花在刀刃上：**思考接下来该组合哪几个函数**，而不是重新发明轮子。

你问一句「周二的数据怎么了」，它现场写一段十几行的小脚本，把你的函数库组合起来跑出答案。

<figure>

<figcaption>现场造轮子 vs 组合积木对比图</figcaption>
</figure>

#### 玩法三：挂只在 skill 激活期间生效的 hook

这是三个玩法里最容易被忽略、但想象空间最大的一个：skill 可以自带 hook，而且这种 hook **只在 skill 被调用时注册，会话结束就失效**。

为什么这个设计很妙？想想官方给的两个例子。

一个叫 careful 的 skill：激活后自动阻断 rm -rf、DROP TABLE、强制推送这类危险命令。这种拦截要是常驻开着，开发体验能把人逼疯；但在你明确知道「我现在要碰生产环境」的时刻，手动激活它，就是一道恰到好处的保险。

另一个叫 freeze 的 skill：激活后禁止修改指定目录之外的任何文件。专治排查 bug 时的「我只是想加两行日志，结果 Claude 顺手把无关代码也给修了」。

<figure>

<figcaption>临时 hook 开关示意图</figcaption>
</figure>

用法也很轻：这类 hook 直接在 skill 的 frontmatter 里声明就行，skill 被调用时自动注册，会话结束自动失效，不需要你去碰全局的 hook 配置。

**skill 的上限不是一份好文档，是一个带记忆、带工具、带保险丝的小型工作系统。**

------------------------------------------------------------------------

### Q6：skill 怎么从你的本地走向全团队？

一个人把 skill 玩明白了，价值是 1；让整个团队都用上，价值才是 N。但一到团队层面，马上冒出三个新问题：怎么分发？谁来审批？质量怎么保证？

先说分发。官方给了两条路。

第一条路：**把 skill 直接提交进代码仓库**，放在 .claude/skills 目录下。团队成员拉代码的时候 skill 就跟着到位了，零成本同步。小团队、仓库不多的场景，这条路最省事。

但还记得 Q3 说的那个 1% 预算吗？这条路有个隐性代价：仓库里每多一个 skill，每个人每次会话的清单里就多一行，所有人无差别承担这份 context 开销，不管用不用得上。

所以规模一上来，官方推荐第二条路：**做成 plugin，搭一个团队内部的 plugin marketplace**。skill 打包上架，谁需要谁安装，context 成本回归到「谁用谁付」。新人入职装一遍团队插件，立刻获得和老员工一样的装备。

<figure>

<figcaption>两条分发路线对比图</figcaption>
</figure>

那 marketplace 谁说了算？哪些 skill 能上架？

Anthropic 内部的答案可能跟你想的不一样：**没有一个中心化的团队做审批**。

他们的玩法是完全的自然演化：你写了个 skill 想给大家试试，先扔进 GitHub 上的一个沙盒文件夹，在 Slack 里吆喝一声。用的人多了、口碑起来了（火没火由 skill 作者自己判断），作者再提一个 PR 把它从沙盒挪进正式 marketplace。

<figure>

</figure>

像不像开源社区的运作方式？好东西靠口碑自己长出来，而不是靠委员会评出来。审批环节越重，愿意分享的人越少；门槛低到「扔进沙盒就行」，几百个 skill 才攒得起来。

<figure>

<figcaption>skill 自然演化漏斗图</figcaption>
</figure>

还有一个团队场景下绕不开的小问题：skill 之间能不能互相依赖？比如一个生成 CSV 的 skill，最后一步要调用另一个文件上传 skill。

官方很坦诚：依赖管理目前没有原生支持。但解法意外地简单，**在 skill 正文里直接报另一个 skill 的名字就行**，只要对方装了，模型自己会去调用。毕竟执行者是一个能理解自然语言的 agent，「用 file-upload skill 把结果传上去」这句话本身就是依赖声明。

<figure>

<figcaption>skill 自然语言依赖示意图</figcaption>
</figure>

**好 skill 的团队化路径：沙盒里长出来，口碑里筛出来，marketplace 里沉淀下来。**

------------------------------------------------------------------------

### Q7：怎么知道一个 skill 到底有没有人用？

最后一个问题，也是大多数团队压根没意识到要问的问题。

skill 攒了几十个，marketplace 也搭起来了，然后呢？哪些 skill 天天被调用，哪些写完就成了仓库里的化石？没有数据，你只能靠感觉。

靠感觉的结果通常是：大家继续给没人用的 skill 添砖加瓦，真正高频的 skill 反而没人维护。

Anthropic 的做法是给 skill 做埋点，思路相当巧妙：**用一个 PreToolUse hook 监听 skill 工具的每一次调用，把「谁在什么时候用了哪个 skill」记录下来**，汇总成公司内部的使用统计。

相当于给每个 skill 装了个计数器，数据一拉出来，两类问题立刻现形。

一类是「受欢迎的 skill」：调用量大，值得重点维护、优先打磨，Q2 说的「派工程师花一周打磨验证 skill」这种投入，就该花在这类 skill 上。

另一类更有意思，叫**触发不足（undertriggering）**：你预期它该被高频使用，数据却显示几乎没人碰。这种 skill 八成是 Q3 的病，description 没写对，模型扫一眼清单根本想不起它。数据帮你把「该触发却没触发」的病人筛出来，再回头去修 description，形成闭环。

<figure>

<figcaption>skill 埋点度量闭环图</figcaption>
</figure>

这件事的成本低到没有借口不做：一个 hook、一段日志脚本，官方连示例代码都开源出来了。但它把 skill 建设从「凭感觉做」变成了「看数据做」，这是个质变。

**不被度量的 skill 库，迟早变成没人敢动也没人想用的杂物间。**

### 最后

7 个问题说完了，按惯例浓缩成 3 句话送你：

<figure>

<figcaption>全文总结图</figcaption>
</figure>

- 第一，skill 是文件夹不是文件，把脚本、坑点清单、记忆文件、临时 hook 都用上，它才是一个完整的工作系统。

- 第二，决定 skill 命运的是 description 的前 250 个字符，写成「什么场景下用我」的触发条件，而不是给人看的摘要；装太多 skill 会互相挤占清单预算，贵精不贵多。

- 第三，如果只做一件事，先做验证类 skill，让 Claude 能自己确认工作成果，这是 Anthropic 实测回报最大的投入。

博客的结尾有一句话我很喜欢，也送给准备动手的你：他们内部最好的那批 skill，几乎都是从「几行字加一个坑点」开始的，然后随着 Claude 撞上一个个新的边界情况，被人一点点喂大。

如果这篇文章对你有帮助，记得点个赞、在看、转发三连，感谢林友们的支持！

我们下篇见啦。

------------------------------------------------------------------------

### 参考资料

- Anthropic 博客《Lessons from building Claude Code: How we use skills》：<a href="https://claude.com/blog/lessons-from-building-claude-code-how-we-use-skills" target="_blank" rel="noopener noreferrer">https://claude.com/blog/lessons-from-building-claude-code-how-we-use-skills</a>
- Claude Code skill 官方文档：<a href="https://code.claude.com/docs/en/skills" target="_blank" rel="noopener noreferrer">https://code.claude.com/docs/en/skills</a>
- 官方示例 skill 仓库：<a href="https://github.com/anthropics/skills" target="_blank" rel="noopener noreferrer">https://github.com/anthropics/skills</a>
