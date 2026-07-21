# 增量失效机制的已知缺陷

**日期**: 2026-07-20
**状态**: 设计期记录，缺陷均已知且被明确接受，留待后续处理
**关联**: `CHECK_OUTDATE_PLAN.md`（仓库根）

本文集中记录 `check_outdate` 增量失效机制**已知不能覆盖的情况**。这些缺陷是设计权衡的
结果，不是实现 bug。写下来是为了避免日后有人误以为覆盖是完备的，也为将来改进留下起点。

所有结论均在 Isabelle2025-2 / HOL session 下实测，证据附在各条目内。

---

## 缺陷索引

| # | 缺陷 | 严重度 | 影响范围 |
|---|---|---|---|
| 1 | 类型类实例定义对依赖不可见 | **中** | 全部多态算术/序关系引理 |
| 2 | 类参数常量恒不失效 | 低（且部分正确） | 类参数常量 |
| 3 | class 超类传递闭包会被下游污染 | 低 | 317 个 class |
| 4 | method 永不过期 | 低 | 28 个 |
| 5 | theorem collection 永不过期 | 低 | 64 个 |

---

## 1. 类型类实例定义对依赖不可见 ⚠️ 主缺陷

### 现象

改变一个类型类实例的定义（例如 `+` 在 `nat` 上如何计算），**不会**让任何关于该类型
该运算的引理失效。

### 证据

一条纯粹关于 nat 加法的引理，其项中出现的常量是：

```
Nat.add_Suc            →  Groups.plus_class.plus :: nat ⇒ nat ⇒ nat
                          Nat.Suc :: nat ⇒ nat
                          HOL.eq :: nat ⇒ nat ⇒ bool
                          HOL.Trueprop :: bool ⇒ prop

(x::nat) + y = y + x   →  Groups.plus_class.plus :: nat ⇒ nat ⇒ nat
                          HOL.eq :: nat ⇒ nat ⇒ bool
```

出现的是**类参数** `Groups.plus_class.plus`（仅类型被实例化为 `nat ⇒ nat ⇒ nat`），
**没有**实例常量 `Nat.plus_nat_inst.plus_nat`。

而类参数 `plus` 本身的 digest 只含名字与声明类型（见缺陷 2），恒定不变。因此依赖链在此
处断开：

```
Nat.plus_nat_def  改变
      ↓ （无边）
Groups.plus_class.plus  digest 不变
      ↓
Nat.add_Suc 等引理  不失效
```

### 根因

两个设计决定叠加：

- **statement 级依赖**（`CHECK_OUTDATE_PLAN.md` §4.6）：依赖取自定理陈述中出现的
  constant/type。类型类的实例解析发生在项之外，陈述看不见它。
- **下游污染过滤器**（同 §6.0）：`Defs.specifications_of` 对 `plus` 返回 9 条实例定义
  公理，分布在 `Nat`/`Int`/`String` 等下游 theory 中。若纳入则 `Groups` 的实体会依赖
  其后代，既造成大面积过度失效，又使依赖边指向前方、破坏拓扑序假设。故必须过滤掉。

过滤是必需的；断链是过滤的代价。

### 为何接受

本机制检测的是**英文解释是否过时**，而非逻辑健全性。改变一个类型类实例的实现，
几乎不会改变"自然数加法的交换律"这句描述。同时实例定义在实践中极少变动。

### 可能的修复方向（未实施）

对项中每个 `Const (c, T)`，若 `c` 是类参数且 `T` 已实例化，则通过 `Axclass` 的
类型类解析机制找出对应的实例常量，将其补为依赖。

- 拓扑序上是安全的：实例常量（如 `Nat.plus_nat`）位于 `Nat`，而使用它的引理必在 `Nat`
  或更下游，边仍指向后方。
- 代价：需对每个常量出现位置做一次类型类解析，成本与复杂度均显著。
- 建议在有实际证据表明该缺陷造成困扰后再实施。

---

## 2. 类参数常量恒不失效

### 现象

类型类参数（`Groups.plus_class.plus`、`Orderings.ord_class.less_eq` 等）**按明确规则**
不保留任何定义公理，其 digest 只由 `名字 ⊕ 声明类型 ⊕ 所属类` 构成。此后只有改变其声明
类型、或其所属类的公理，才会使它失效。

实现上这不是过滤的副产品，而是一条独立规则：

```sml
if is_some (Axclass.class_of_param thy c) then []   (* 类参数没有自己的定义 *)
```

### 为什么必须如此

仅按"同 theory"过滤是不够的。实测 `Orderings.ord_class.less_eq` 的 16 条定义公理中，
`ord_bool_inst.less_eq_bool_def` 与 `ord_fun_inst.less_eq_fun_def` **就声明在
`Orderings.thy` 自己里**，同 theory 过滤放行它们。

而 `less_eq` 被 **13.3%** 的定理提及。若保留这两条，则编辑 bool 实例会失效八分之一的库，
其中绝大多数是 bool 实例根本影响不到的 nat/int/real 引理。故必须按"是否类参数"判定，
而非按 theory 判定。

### 部分正确性

这个结果**在其自身层面是正确的**：`plus` 在 `Groups` 中本就只被声明、未被定义；那些实例
定义属于 `Nat`、`Int` 各自的实体。新增实例或在下游新增子类，都不应改变"加法，某个 plus
类上的二元运算"这句描述。

缺陷仅在于它与缺陷 1 叠加：类参数恒定，而实例常量又不出现在项里，于是断链无法在此处补上。

---

## 3. class 超类传递闭包会被下游污染

### 现象

`Sign.super_classes` 返回的是**传递闭包**（实测 `Orderings.order` supers=3、
`Groups.semigroup_add` supers=8，均多于其直接超类）。

而 Isabelle 的 `subclass A < B` 命令可以在**下游 theory** 中证明新的类包含关系。一旦如此，
上游某个类的传递超类集合就会增长——与缺陷 1 中 `plus` 的 9 条实例定义完全同构的下游污染。

### 处理

`CHECK_OUTDATE_PLAN.md` §6.0 的过滤器必须同样施加于 class：只保留归属 theory ⊑ 该类
归属 theory 的超类；或更稳妥地只取该类声明处的直接超类。

**未过滤的后果**：下游任何一个 `subclass` 证明都会让上游 class 失效，并使依赖边指向前方。

---

## 4. method 永不过期

`Method` 是注册在 name space 中的 ML 闭包，**无项结构可编码**。

其描述串确实存在于 theory data 中：

```
Method.setup : binding -> ... -> string -> theory -> theory      (* 末位 string 即描述 *)
Data:  {methods: ((Token.src -> Proof.context -> method) * string) Name_Space.table}
                                                                  (Pure/Isar/method.ML:293)
访问器: val get_methods = #methods o Data.get                     (method.ML:310)
```

但 `get_methods` 不在 `METHOD` 签名内，实测不可达：

```
ML error: Value or constructor (get_methods) has not been declared in structure Method
```

`Method` 仅导出 `method_space`（有名字、无描述）。取得描述需修改 Pure 源码，**已明确否决**。

**后果**：改动某 method 的 ML 实现后，其英文描述静默失效且无人察觉。影响 28 个实体。

---

## 5. theorem collection 永不过期

两条独立原因：

- **成员不可用**：成员是 dynamic 的，来自下游 theory（实测 `no_atp` 174 个、
  `ac_simps` 42 个，贡献散布于整个 HOL）。纳入即违反下游污染规则。这与既有决策一致——
  `semantic_store.ML:462-465` 已写明成员只作 `prompt_hint`，
  "members are deliberately kept out of expr/key"。
- **描述不可用**：`Named_Theorems.declare binding descr`
  （`Pure/Tools/named_theorems.ML:88-100`）**不把 descr 存进自身数据**（其 Data 仅
  `thm Item_Net.T Symtab.table`，只有成员），而是加工成
  `"declaration of " ^ (if descr = "" then name ^ " rules" else descr)` 后交给
  `Attrib.local_setup`，落入**属性表**；而 `Attrib.get_attributes`（`attrib.ML:106`）
  同样未导出。且未写描述者存的是 `"declaration of foo rules"`，纯由名字推出，零信息量。

因此 digest = 名字 + 所属 theory，deps = 空，等价于永不过期。影响 64 个实体。

---

## ⚠️ 「覆盖率」有两种，不要混淆

下表统计的是**解析覆盖率**——「该实体能否算出一个 digest」。它**不等于失效能力**——
「内容改了 digest 会不会变」。

第一版实现在解析覆盖率上是 98.6%，确定性测试也全绿，却同时存在 6 个「改了内容 digest
逐位不变」的缺陷（abbreviation 的 rhs、locale 的 `assumes`、typedef 的定义集合、class 的
参数、sort 的类边、超类闭包污染），外加 axiomatization 常量丢失全部公理。

原因是**确定性测试与解析覆盖率测试，对一个几乎不看内容的 digest 同样会全绿**。唯一能发现
这类缺陷的是敏感性测试（「改了 X，digest 必须变」），而第一版没有。

所以：**下表只说明"算得出来"，不说明"测得出变化"。** 失效能力的依据是
`CHECK_OUTDATE_PLAN.md` §11 阶段 1 第 6 项的敏感性测试集。

## 解析覆盖率汇总

| 状态 | kind | 数量 | 占比 |
|---|---|---|---|
| 可算出 digest | thm + 4 rule、constant、type、class、locale | 116,003 | **98.6%** |
| 永不过期 | theorem collection（64）、method（28） | 92 | 0.07% |
| 不适用 | experience（Python 侧数据） | 180 | 0.15% |

数据取自实际语义库（117,611 条记录 / 1,513 个 theory）。

对"永不过期"的那 92 个实体，需保留**手动强制重解释**的入口。
