# PDF 对比细粒度优化方案（可直接替换到你当前组件）

下面是针对你给出的 `pdf compare` 组件的**重点优化点**：

1. 行级 `delete+add` 改 `modify` 的匹配由「贪心相邻」升级为「全局最优匹配（DP）」。
2. 词级差异改为「字符+词混合评分」，减少中文短句误判。
3. PDF 高亮增加归一化与上下文抑噪策略，减少表格重复文本误报。
4. 提供缓存索引，避免 `find/filter` 造成的 O(n²) 卡顿。
5. `v-html` 渲染增加转义，避免内容注入风险。

---

## 1) 先加通用工具函数

```ts
const escapeHtml = (s: string) =>
  s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')

const normalizeForCompare = (s: string) =>
  s
    .normalize('NFKC')
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[，。；：！？]/g, (m) => ({ '，': ',', '。': '.', '；': ';', '：': ':', '！': '!', '？': '?' }[m] ?? m))
    .replace(/\s+/g, ' ')
    .trim()

const jaccardCharBigrams = (a: string, b: string) => {
  const bi = (x: string) => {
    const s = new Set<string>()
    if (x.length <= 1) {
      if (x) s.add(x)
      return s
    }
    for (let i = 0; i < x.length - 1; i++) s.add(x.slice(i, i + 2))
    return s
  }
  const sa = bi(a)
  const sb = bi(b)
  let inter = 0
  for (const t of sa) if (sb.has(t)) inter++
  const uni = sa.size + sb.size - inter
  return uni === 0 ? 1 : inter / uni
}
```

---

## 2) 替换相似度与配对策略（核心）

把你原来的 `calcLineSimilarity` 和 `buildChangedBlock` 替换为下面版本。

```ts
const calcLineSimilarity = (a: string, b: string) => {
  const aa = normalizeForCompare(a)
  const bb = normalizeForCompare(b)

  const diffs = dmp.diff_main(aa, bb, false)
  dmp.diff_cleanupSemantic(diffs)
  const eq = diffs.filter(([op]) => op === DIFF_EQUAL).reduce((s, [, t]) => s + t.length, 0)
  const charScore = eq / Math.max(aa.length, bb.length, 1)
  const biScore = jaccardCharBigrams(aa, bb)

  // 中文场景下二元组更稳，字符编辑距离补充细节
  return 0.55 * biScore + 0.45 * charScore
}

const adaptiveSimilarityThreshold = (oldLine: string, newLine: string) => {
  const a = normalizeForCompare(oldLine)
  const b = normalizeForCompare(newLine)
  const maxLen = Math.max(a.length, b.length)
  if (maxLen <= 8) return 0.52
  if (maxLen <= 20) return 0.45
  if (maxLen <= 60) return 0.38
  return 0.33
}

// 用动态规划做全局最优配对，避免原先“按顺序硬配”导致错位
const buildChangedBlock = (removed: string[], added: string[], startOld: number, startNew: number) => {
  const m = removed.length
  const n = added.length
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0))

  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      const sim = calcLineSimilarity(removed[i], added[j])
      const th = adaptiveSimilarityThreshold(removed[i], added[j])
      const match = sim >= th ? sim + dp[i + 1][j + 1] : -1e9
      const skipOld = dp[i + 1][j]
      const skipNew = dp[i][j + 1]
      dp[i][j] = Math.max(match, skipOld, skipNew)
    }
  }

  const out: LineDiff[] = []
  let i = 0
  let j = 0
  let o = startOld
  let nn = startNew

  while (i < m && j < n) {
    const sim = calcLineSimilarity(removed[i], added[j])
    const th = adaptiveSimilarityThreshold(removed[i], added[j])
    const match = sim >= th ? sim + dp[i + 1][j + 1] : -1e9

    if (Math.abs(dp[i][j] - match) < 1e-9) {
      out.push({
        type: 'modify',
        originalContent: removed[i],
        newContent: added[j],
        wordDiffs: getWordDiffs(removed[i], added[j]),
        originalIndex: o,
        newIndex: nn
      })
      i++; j++; o++; nn++
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      out.push({ type: 'delete', originalContent: removed[i], originalIndex: o })
      i++; o++
    } else {
      out.push({ type: 'add', newContent: added[j], newIndex: nn })
      j++; nn++
    }
  }

  while (i < m) out.push({ type: 'delete', originalContent: removed[i++], originalIndex: o++ })
  while (j < n) out.push({ type: 'add', newContent: added[j++], newIndex: nn++ })

  return { results: out, nextOld: o, nextNew: nn }
}
```

---

## 3) 渲染安全 + 查询性能优化

```ts
const diffByOldIndex = computed(() => {
  const m = new Map<number, LineDiff>()
  for (const r of diffResults.value) {
    if (r.originalIndex != null && !m.has(r.originalIndex)) m.set(r.originalIndex, r)
  }
  return m
})

const diffByNewIndex = computed(() => {
  const m = new Map<number, LineDiff>()
  for (const r of diffResults.value) {
    if (r.newIndex != null && !m.has(r.newIndex)) m.set(r.newIndex, r)
  }
  return m
})

const panelLeftClass = (idx: number) => diffByOldIndex.value.get(idx)?.type || ''
const panelRightClass = (idx: number) => diffByNewIndex.value.get(idx)?.type || ''

const renderLineContent = (result: LineDiff) => {
  if (result.type === 'modify' && result.wordDiffs) {
    return result.wordDiffs
      .map((w) => `<span class="w-${w.type}">${escapeHtml(w.content)}</span>`)
      .join('')
  }
  const raw = result.type === 'add' ? (result.newContent ?? '') : (result.originalContent ?? '')
  return escapeHtml(raw)
}

const renderLeftLine = (idx: number, content: string) => {
  const r = diffByOldIndex.value.get(idx)
  return r?.type === 'modify' ? renderLineContent(r) : escapeHtml(content)
}

const renderRightLine = (idx: number, content: string) => {
  const r = diffByNewIndex.value.get(idx)
  return r?.type === 'modify' ? renderLineContent(r) : escapeHtml(content)
}
```

---

## 4) PDF 细粒度误报抑制（轻量版）

将 `normalizeItemText` 升级为：

```ts
const normalizeItemText = (s: string) =>
  normalizeForCompare(s).replace(/\s+/g, '')
```

并在 `shouldHighlightItem` 顶部增加：

```ts
const shortNoisy = key.length <= 2 && ratio < 0.5
if (shortNoisy) return false

// 重复短 token（如表格“是/否”、“√”）更严格
const sameCount = appearsStableOnBothSides(side, page, key)
if (key.length <= 3 && sameCount && changed <= 1) return false
```

---

## 5) 额外建议（你这份代码特别值得做）

- 你粘贴的片段出现了**整段重复**（`template/script/style` 重复一次），请删除重复块，否则构建会报错。
- `watch([...], { deep: true })` 对 `computed + ref` 的组合会触发很频繁，可加节流（200ms）减少 PDF 重绘。
- 对超长文档建议加 Web Worker 做 diff，避免主线程卡顿。

