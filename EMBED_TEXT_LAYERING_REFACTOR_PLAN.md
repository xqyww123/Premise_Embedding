# 实施计划：record → embedding 文档文本 的单一权威 + 单一写向量咽喉

> 本文是 `EMBED_TEXT_LAYERING_REFACTOR.md`（设计/缺陷分析）的**可执行落地计划**。
> 设计动机、缺陷复现、根因见那份文档；本文只讲"怎么改、改哪里、按什么顺序、怎么验"。
> 跨仓：`Semantic_Embedding` + `Isa-Mini/AoA`，触及准公开接口。撰于 2026-07-11。
> 已经过两轮对抗评审（3 名评审 + 1 名裁判），评审结论并入 §8；批准状态见 §A。

---

## A. 批准状态（⚠️ 读我：区分"用户已批准"与"我的设计，待批准"）

> 用户强调：任何未经明确批准的设计都必须显式标出，不得默认执行。下表是当前状态。

| 决策 | 内容 | 状态 |
|---|---|---|
| D1 | reranker experience `doc_text` 收编到 `document_text_of` | ✅ **用户已明确批准**（"要收编"）|
| D3 | `document_text_of`/`experience_document_text`/`entity_document_text` 做成 `document_text.py` 自由纯函数 | ✅ **用户已明确批准** |
| D6 | `put_experience`/`delete_experience` 做成 `experience_store.py` 自由函数 **+** `Semantic_Vector_Store` 薄方法入口 | ✅ **用户已明确批准** |
| D8 | `write_memory` 经 `await conn.semantic_vector_store()` 顶层接口拿 store | ✅ **用户已明确批准** |
| S1-(a) | `_auto_embed` 豁免"已带 interpretation 的 key（尤其 EXPERIENCE）"于 `auto_interpret_for_embedding` gate | ✅ **用户已明确批准** |
| D2（unicode 轴）| experience 文本在 **unicode** 形式上做 | ✅ 用户已定"肯定在 unicode 上做" |
| D2（模板正文）| experience 文本正文照搬现 `_experience_document_text`（框架句 + `- <unicode pattern>` + 情境句 + desc）| ✅ **用户已明确批准**（"按 write_memory 的模板走"），见 §6 |
| D4 | record→向量 接口 = `Semantic_Vector_Store.embed_records(items:list[(key,rec)], *, force)` | ✅ **用户已明确批准**（命名/签名）|
| D5 | `embed_keys` 退化为薄包装；`_auto_embed`/`embed_entities`/离线全部塌缩到 `embed_records` | ✅ **用户已明确批准** |
| D7 | `embed_records`/`put_experience`/`delete_experience` 不需要 connection（仅守卫式 tracing 用）| ☑️ 事实确认（非设计选择），已与用户核对 |
| D9 | 边界：**展示路径** `query(with_pretty=True)` 语义不变，仅让其 entity 分支复用 `entity_document_text`；只有 embed 路径 + reranker 切 `document_text_of` | ✅ **用户已明确批准** |
| M | §5 迁移方案（删每个 `vector_*.lmdb` 的 EXPERIENCE 向量后重嵌）| ✅ **用户已明确批准** |
| Rt | `embed_records` 返回值 = tokens（非计数）| ✅ **用户已明确批准** |
| Del1 | `delete_experience` purge **所有模型** store（遍历 `_iter_vector_store_envs()`）；`put_experience` 仍只写活跃 store（"惰性嵌入、全局清除"的有原则不对称）；delete 自由函数去掉 `store` 参数 | ✅ **用户已明确批准（删除所有模型）** |
| G1 | `pretty_unicode` 对 `isabelle` CLI 的依赖 | ✅ **用户已自行修复,本计划忽略** |

**未经批准者一律不落地。** 机械/正确性修复（C1、C2、C3、S3、C4）不含设计选择，随对应步骤实现，但仍等你对整体计划放行。

---

## 0. 核心不变量（整份计划的判据）

> **embedding 文档文本 = 已存储 `Record` 的纯函数（给定一致的符号表环境）。**
> 只要文本是"那条 record"的纯函数，"写入时算的文本"与"事后重嵌时算的文本"就在**构造上字节相同**，
> 分裂在物理上无法复发。这比原设计文档"共享一个函数"更强——它把不变量钉在**输入 = record** 上。
> （"一致的符号表"限定见 §1.1；`pretty_unicode` 的 isabelle 依赖由用户在 G1 处修复。）

派生的两条硬约束：
1. 任何写向量的路径，文本**只能**经 `document_text_of(rec)` 产生，不得自行拼装。
2. `write_memory` 写 experience 时，embed 的文本必须来自它**即将存储的那条 rec**，否则 ascii/unicode 轴上会重新分裂（§1.1）。

---

## 1. 已锁定的设计决策（附一句理由；批准状态见 §A）

| # | 决策 | 理由 |
|---|---|---|
| D1 | reranker experience `doc_text`（`lookup` 内，现为裸 `rec.interpretation`）**收编**到 `document_text_of` | cross-encoder 应给"被 embed 的 canonical 文档"打分 |
| D2 | experience 文本在 **unicode** 上；`document_text_of` 从 `rec.expr`（ascii JSON）经 `pretty_unicode` 重建 | 保留"embed unicode 语义形式"意图，改为从 record 可复现派生。**模板正文待拍板** |
| D3 | 文本三函数 → `document_text.py` 自由纯函数 | 存储类不该拥有 embedding 域知识；纯函数可单测 |
| D4 | record→向量 = `Semantic_Vector_Store.embed_records(items, *, force)`，`items: list[(universal_key, Record)]` | 实例死绑 `(model, provider)`；Record 无 key、向量按 key 寻址，故取 (key, rec) 对 |
| D5 | `embed_keys` 退化为 `embed_records` 薄包装；三处写实现塌缩 | 单一写向量咽喉；`document_text_of` 只在 `embed_records` 内调一次 |
| D6 | experience 三存储事务 → `experience_store.py` 自由函数 + store 薄方法入口 | 逻辑可单测，方法是 ergonomic facade |
| D7 | 三函数不要求 connection 非空；仅守卫式 tracing 用 | embedding 走 `store.emb_provider`；两单例与连接无关。离线 `connection=None` 为证 |
| D8 | `write_memory` 经 `conn.semantic_vector_store()` 拿 store，再走 `store.put_experience`/`delete_experience` | store 是（嵌入+检索）半系统顶层 facade |
| D9 | 只 embed 路径 + reranker 切 `document_text_of`；展示路径 `query(with_pretty=True)` 语义不变，仅 entity 分支复用 `entity_document_text` | entity 下 `document_text_of == query(with_pretty=True)`，对 entity 展示零变化；experience 不经 `query_by_name_raw` |
| S1-(a) | `_auto_embed` 豁免已带 interpretation 的 key 于 `auto_interpret_for_embedding` gate | gate 本意是"要不要自动**解释**"；experience 自带解释、零解释成本，不该被连坐（§8-S1）|

### 1.1 ascii/unicode 陷阱（D2 的动机）
`write_memory` 有意地：experience pattern 以 **ascii** 存 `rec.expr`（检索时 ML 内层词法器只吃 ascii），今天 embed 的却是 **unicode**（`pats_uni`）。任何从 record 重嵌的路径只能看到 ascii。若 write 用 `pats_uni`、重嵌用 `rec.expr`，分裂从 kind 轴换到 ascii/unicode 轴复发。
D2 解法：`document_text_of` 统一从 `rec.expr` 经 `pretty_unicode` 重建 unicode，write 与重嵌都作用于同一条 rec → 字节相同。
**注意（评审 C4/S6 澄清）**：对良构 unicode 输入，`pats_uni == pretty_unicode(pats_ascii)`；但若 agent 以 **ascii 记法**输入（如 `\<forall>x. P x`），`IsaTerm.from_agent` 使 `.unicode` = 该 ascii 串，而 `document_text_of` 会产出 `pretty_unicode(pats_ascii)`（unicode 形式），二者**不等**——即本次 embed 的 experience 文本约定确实**从"agent 原样"切换为"pretty_unicode(ascii)"**。核心不变量仍成立（write 与重嵌都用 `document_text_of`），此约定切换由 §5 迁移删+重嵌吸收。§1.1 不再声称二者恒等。

---

## 2. 目标模块/层次布局

```
document_text.py         [新增，纯函数，零 lmdb/连接依赖]
    experience_document_text(patterns: list[str], desc: str) -> str      # 从 AoA 下移
    entity_document_text(rec) -> str                                     # rec.pretty_print + "\n" + rec.interpretation
    document_text_of(rec) -> str | None                                  # ★唯一权威，按 kind 分派
        └ EXPERIENCE 分支: try: patterns=[pretty_unicode(p) for p in json.loads(rec.expr or "[]")]
                           except (json.JSONDecodeError, TypeError): return None   # 评审 S3
                           → experience_document_text(patterns, rec.interpretation)

semantic_embedding.py    [不动]
    Vector_Store.embed(kv_pairs) -> int(tokens)                          # 低层原语

semantics.py             [改]
    Semantic_Vector_Store.embed_records(items, *, force=False) -> int    # ★唯一写向量咽喉
    Semantic_Vector_Store.embed_keys(keys, *, force=False) -> int        # 薄包装 → embed_records
    Semantic_Vector_Store.embed_entities / embed_all_entities_in_theories# 改走 embed_keys
    Semantic_Vector_Store._auto_embed                                    # 末段走 embed_records；补 early-return(C1)；gate 豁免(S1-a)
    Semantic_Vector_Store.lookup                                         # reranker doc_text → document_text_of (D1)
    Semantic_Vector_Store.put_experience / delete_experience            # 薄方法入口 → experience_store
    _Semantic_DB.query(with_pretty=True)                                 # entity 分支复用 entity_document_text (D9)

experience_store.py      [新增，跨三存储事务]
    async put_experience(store, key, rec) -> None
    delete_experience(store, key) -> None

Isa-Mini/AoA/mcp_http_server.py   [改]
    删除 _experience_document_text（下移）
    write_memory: 提前建一次 rec；dedup 用 document_text_of(rec)；
                  _persist → store.put_experience；Case-1 循环保留 `if uk != key` 守卫(C2)后 store.delete_experience

semantics_manage.py      [改]
    离线 embed: iter_entity_records 产出 (key,rec)，按 BATCH=256 分块喂 embed_records(C3)；删 f63b0bb 排除
    迁移入口（§5）
```

---

## 3. 详细实现步骤

### 步骤 1 — 新建 `document_text.py`（无行为变化）
- `experience_document_text(patterns, desc)`：正文=当前 AoA `_experience_document_text` 字节等价搬迁（§6 待拍板后定稿）。
- `entity_document_text(rec)` = `rec.pretty_print + "\n" + rec.interpretation`。
- `document_text_of(rec)`：
  - `rec.interpretation is None → None`。
  - EXPERIENCE：**try** `patterns=[pretty_unicode(p) for p in json.loads(rec.expr or "[]")]` **except (JSONDecodeError, TypeError): return None**（评审 S3，与 `_experience_hits` 的防御一致）→ `experience_document_text(patterns, rec.interpretation)`。
  - 否则 `entity_document_text(rec)`。
- import：仅 `EntityKind`（`universal_key`）、`pretty_unicode`（`unicode`）；**不 import `semantics`**（破循环）。

### 步骤 2 — `embed_records` + `embed_keys`
```python
async def embed_records(self, items, *, force=False) -> int:
    if not force:
        keys = [k for k, _ in items]
        exist = self.contains(keys)
        items = [it for it, ex in zip(items, exist) if not ex]
    kv = [(k, t) for k, r in items if (t := document_text_of(r)) is not None]
    if not kv:
        return 0
    if self.connection is not None:                    # D7 守卫
        await self.connection.tracing(f"[Semantic_Embedding] embedding {len(kv)} records ...")
    return await self.embed(kv)                          # 基类原语，返回 total_tokens

async def embed_keys(self, keys, *, force=False) -> int:
    items = [(k, r) for k in keys if (r := Semantic_DB[k]) is not None]
    return await self.embed_records(items, force=force)
```
- `embed_entities(keys)` → `await self.embed_keys(keys, force=False)`。
- `embed_all_entities_in_theories` 内 `self._embed_keys(...)` → `self.embed_keys(...)`。
- **删除旧 `_embed_keys`**。返回值统一 tokens（Rt；核对：`embed_all_entities_in_theories`/`embed_entities` 及 RPC wrapper 均弃返回值 → 安全）。

### 步骤 3 — `_auto_embed` 改造（走 embed_records + 修 C1 + 加 S1-a）
- 保留前半：抽 theory hash（跳过 xor-prefixed，故 experience 从不进 `theory_hashes`）、`>5` 弹窗、`interpret_theories_by_names`、`>42` 弹窗、`mark_thy_embedded`。
- **S1-(a) gate 豁免**：把顶端 `if not auto_interpret_for_embedding: return []` 从"一刀切"改为"只挡需要解释的 key"。即先把 `missing` 分成：
  - `ready`（已带 interpretation，含所有 EXPERIENCE 及所属理论已解释的实体）→ **无条件可嵌**；
  - `needs_interp`（`document_text_of` 现为 None、且其理论未解释）→ 仅当 gate 开时才走解释流程。
  gate 关时：跳过解释与其 warning 对 `needs_interp` 的部分，但**仍嵌 `ready`**（`ready` 非空时不 return）。
  实现要点：`ready` 的判定 = `Semantic_DB[k]` 存在且 `document_text_of(rec) is not None`。
- **C1 early-return**：在 `mark_thy_embedded` 循环**之前**补 `if not items: return []`，避免空批把 theory 误标 `finished`。
- 末段：`items = ready(+gate 开时新解释出的)` → `tokens = await self.embed_records(items, force=True)`；`>42` 确认仍套在 `items` 上；`for th in theory_hashes: mark_thy_embedded(th, tokens)`；`return [k for k,_ in items]`。
- **注意**：数量成本仍由 `>42`/`>5` 弹窗兜底，故 gate 豁免不会让 gate-关用户被偷偷大批量嵌入。

### 步骤 4 — `lookup` reranker doc_text 收编（D1）
- `doc_text = rec.interpretation if EXPERIENCE else query(with_pretty=True)` → `doc_text = document_text_of(rec)`。

### 步骤 5 — `query(with_pretty=True)` 复用 `entity_document_text`（D9，纯去重）
- `_Semantic_DB.query` 的 `with_pretty` 分支改为对 entity 复用 `entity_document_text(rec)`（字节不变）。**不**让 `query` 对 experience 走框架文本；embed 侧一律不再经 `query`。

### 步骤 6 — `experience_store.py`
```python
async def put_experience(store, key, rec) -> None:
    await store.embed_records([(key, rec)], force=True)          # embed 先行（唯一可远程失败步）
    Semantic_DB[key] = rec
    Experience_Index.add(key, [h for _, h in (rec.theory_constituents or [])])

def delete_experience(key) -> None:                              # Del1：无需 store 参数
    old = Semantic_DB[key]
    Semantic_DB.delete(key)
    for env in _iter_vector_store_envs():                        # Del1：purge 所有模型 store
        with env.begin(write=True) as txn:
            txn.delete(key)
    consts = old.theory_constituents if old is not None else None
    if consts: Experience_Index.remove(key, [h for _, h in consts])
    else:      Experience_Index.remove_scanning(key)
```
- `put_experience` 只写活跃 store（惰性嵌入）；`delete_experience` purge **所有**模型 store（全局清除，消灭覆盖换 key 产生的孤儿向量）——有原则的不对称（Del1）。
- 语义与今天 `_persist`/`_delete_uk` 等价（除 vector 删除从单店扩为全店），文本来源换成 `embed_records`。全程同步无 await，不违反写事务并发不变量。
- `delete_experience` **不内置** same-key 守卫（它无从知道"刚写的 key"）——该守卫留在调用点（C2，见步骤 7）。
- store 薄方法入口（lazy import 破循环）：`store.put_experience(key,rec)`（用 self）/ `store.delete_experience(key)`（委托自由函数 `delete_experience(key)`，self 仅作 ergonomic 句柄）。

### 步骤 7 — AoA `write_memory` 改造（跨仓 Isa-Mini）
- 在 key + constituents 算好后**提前建一次** `rec = SemanticRecord(EntityKind.EXPERIENCE, name, json.dumps(pats_ascii), desc, None, constituents, experience)`。
- dedup 查询文本（原 `_experience_document_text(pats_uni,desc)`）→ `document_text_of(rec)`。
- `_persist` 三步 → `await store.put_experience(key, rec)` + 会话记账（`created[name]=key`、`written_names.append`、`dedup_block=None`）。
- **C2**：Case-1 覆盖循环**必须保留 `if uk != key` 守卫**，写全为
  ```python
  await store.put_experience(key, rec)          # 先 durably 写新
  for uk in targets:
      if uk != key:                             # ★never delete the just-written key
          store.delete_experience(uk)
  ```
- **删除** AoA 的 `_experience_document_text`；改 `from Isabelle_Semantic_Embedding.document_text import document_text_of`（及 `experience_document_text` 若仍需）。

### 步骤 8 — 离线 `semantics_manage.py`（含 C3）
- `_collect_embed_candidates`/`_embed_models`：**直接**用 `Semantic_DB.iter_entity_records()` 产出的 `(key, rec)`，按 **BATCH=256** 分块喂 `store.embed_records(chunk)`——记录已在手，**不**回查 `Semantic_DB[k]`、**不**经 `embed_keys`（避免在游标存活时重开读事务，评审 C3）。
- **删掉 f63b0bb 的 experience 排除**——experience 现在正确嵌入。
- 迁移入口见 §5。

---

## 4. 验证（每条实跑）

- **Unit（`document_text.py`）**：experience rec → `== experience_document_text(patterns,desc)`；entity rec → `== pretty_print+"\n"+interpretation`；`interpretation is None → None`；**畸形 `rec.expr` → None（不抛）**（S3）。
- **Equivalence（entity 不回归）**：一批 entity，`embed_records` 文本与旧 `query(with_pretty=True)` 逐字节相等。
- **Consistency（核心回归守卫）**：同一 experience，`put_experience` 写入文本 与 `embed_keys`/`_auto_embed` 重嵌文本 **字符串相等**。
- **ascii/unicode（C4/S6 修正后）**：断言 `document_text_of(rec)` 与**旧** `_experience_document_text(pats_uni, desc)` 的关系——对 unicode 输入相等、对 ascii 记法输入记录其差异（证明约定确实切换、且切换是一致的），**不再**写成同义反复的 `== pretty_unicode(pats_ascii)`。
- **connection=None（D7）**：`connection=None` 构造 store，`embed_records`/`put_experience` 不抛。
- **_auto_embed gate 豁免（S1-a）**：gate 关 + 一批 missing experience → 仍被嵌入并可检索；空批（C1）→ 不 `mark_thy_embedded`。
- **跨仓端到端 golden（评审 S5，替换原先只列 WriteMemoryGate 的错误）**：跑 `Isa-Mini/IsaMini/AoA/Tests/` 下 `ExperienceMemory.yml`、`DedupOverwriteSameRun.yml`、`DedupRejectThenAdjacent.yml`（其渲染编码 0.7 阈值，对查询向量漂移敏感）、`MemorizeInteractionStages.yml`；`WriteMemoryGate.yml` 只验工具登记、抓不到 embed/dedup 回归。**golden 若需改动，先经用户批准再动**（仓库硬规矩）。

---

## 5. 迁移（既有 store）【待批准 M】

- experience 向量是派生缓存且历史混合约定 → **删每个 `vector_*.lmdb` 的所有 EXPERIENCE 向量**（`_iter_vector_store_envs()` 遍历所有模型 store），再重建。
- entity 向量约定不变（D9）→ 不重嵌 entity。
- 重建走**同步** `embed_records(force=True)`（**不经** `_auto_embed`，故与 gate 无关，是急性期的可靠恢复）。
- 一次性入口：`semantics_manage.py` 加 `embed --kinds experience --force`：遍历 `Semantic_DB` experience key（kind 标签字节判据，`_scan_experiences` 已有），对每个 `vector_*.lmdb` 先 `delete` 后 `embed_records`。

---

## 6. ⚠️ 未决项：experience 文本模板（D2 正文，待用户拍板）

当前 AoA `_experience_document_text` 正文：
```
This is an experience that aims to help prove goals of the following forms:
- <pattern 1>
- <pattern 2>
...
The experience should be used in the following situation:
<goal_description>
```
用户初步倾向"照搬（unicode pattern + 框架句）"，需最终确认。**拍板前不写 `experience_document_text` 正文、不动任何代码。**

---

## 7. 顺序与跨仓提交

1. 步骤 1（`document_text.py`）+ 步骤 5（`query` 去重）：entity 零行为变化，先落。
2. 步骤 2–4（`embed_records`/`embed_keys`/`_auto_embed`(含 C1、S1-a)/reranker）。
3. 步骤 6（`experience_store.py`）+ 步骤 7（AoA `write_memory`，含 C2）。
4. 步骤 8（离线，含 C3）+ §5 迁移 + §4 验证。
- 跨仓：`Semantic_Embedding` 与 `Isa-Mini` 两仓；`Semantic_Embedding` 侧先合。共享工作树，提交信息须完整描述被一并带入的他人改动。
- 纯 Python，无 `.ML` 改动；无需 rebuild heap；AoA 若在跑 REPL，改 `mcp_http_server.py` 后按 Isa-Mini 规则重启 server 再测。

---

## 8. 两轮对抗评审结论（存档）

3 名评审（correctness / codebase-grounding / scope-edge）+ 1 名对抗裁判。**删除 3 条低质量意见**、**8 条存活并入本计划**。

**删除（低质量）**：
- **S2 跨模型向量一致性** — 越界：单模型写是既有架构，"单一咽喉"指文本派生而非向量多店扇出。
- **S4 换模型后 dedup 退化** — 越界+与 S1 重复：`topk→_auto_embed` 未被本计划改动。
- **G3 query 调用点枚举不全** — 琐碎：漏掉两处自证 entity 字节等价、无需动作。

**存活（已并入）**：
- **C1**（→步骤 3）：`_auto_embed` 补空批 early-return，防误标 `finished`。
- **C2**（→步骤 7）：保留 Case-1 `if uk != key` 守卫，防幂等覆盖抹掉刚写记忆。
- **S3**（→步骤 1）：`document_text_of` 的 `json.loads` 加 try/except，防一条坏记录炸整批。
- **S1**（→步骤 3，用户批准 (a)）：`_auto_embed` gate 豁免已带 interpretation 的 key。
- **C3**（→步骤 8）：离线用 `(key,rec)` 直喂 `embed_records` 分块，避免游标内重开读事务 + 保留 BATCH=256。
- **G1**（用户已修）：`pretty_unicode` 的 `isabelle` CLI 依赖 —— 本计划忽略。
- **S5**（→步骤 4/§4）：验证集换成真正端到端的四个 golden。
- **C4/S6**（→§1.1/§4）：ascii/unicode 校验从同义反复改为对比旧 `_experience_document_text(pats_uni)`。
