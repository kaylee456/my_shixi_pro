# PDF 全量导出：目录页码错位与重复章节排查/修复

## 问题 1：目录页码不匹配标题

根因通常有两类：

1. **页码基准不一致**：`dump-outline` 得到的是“正文单独渲染”页码，但最终是“目录+正文合并渲染”，目录插入后正文整体后移。
2. **标题匹配不稳定**：按标题文本匹配时，若标题重复（如“概述”“小结”），会把页码对应到错误条目。

### 修复建议

- **强制使用“锚点 ID”关联 TOC**，不要按标题文本关联。
- **统一渲染参数**：Step1 与 Step2 的字体、页边距、宽度、DPI、缩放必须完全一致。
- **对最终 TOC 页码做偏移**：最终页码 = outline 页码 + TOC 页数。

推荐流程：

1. Step1 正文渲染时给 `h1/h2/h3` 注入唯一 `id="hN"`（你已有）。
2. Step1 生成 outline 后，按**出现顺序**与注入顺序进行绑定（或在标题文本中临时注入不可见 token）。
3. 先渲染一版“目录+正文”草稿，计算目录占用页数 `toc_pages`。
4. 重新生成 TOC，将每个页码加上 `toc_pages`，再做最终渲染。

## 问题 2：重复章节出现

根因通常是：

1. 同时使用了 **wkhtmltopdf 自动目录** 与 **自定义目录 HTML**。
2. 文件扫描重复（同一章节被加入两次，或软链接/缓存文件被扫描）。
3. 合并时把同一 PDF 段重复 merge。

### 修复建议

- 仅保留一种目录机制：你已使用自定义 TOC，就不要再启用 `toc` 参数。
- 章节文件先去重再渲染。
- 合并前打印并断言输入 PDF 列表唯一且顺序正确。

示例（Python）

```python
# 章节去重（保留第一次出现）
seen = set()
html_files_unique = []
for f in html_files:
    key = os.path.realpath(os.path.join(output_dir, f))
    if key in seen:
        continue
    seen.add(key)
    html_files_unique.append(f)
html_files = html_files_unique

# 合并前防重复
input_pdfs = [cover_pdf_path, combined_pdf_path]
assert len(input_pdfs) == len(set(input_pdfs)), f"重复合并输入: {input_pdfs}"
```

## 最稳妥的结构调整（推荐）

- Step1：正文渲染（仅用于拿 outline）。
- Step2：计算 TOC 页数并修正页码。
- Step3：最终渲染“目录+正文”。
- Step4：封面与 Step3 合并。

并且在日志中输出：

- 注入标题数
- outline 解析条目数
- TOC 渲染条目数
- 合并 PDF 输入列表

这样可以很快定位“页码错位”与“重复章节”到底发生在哪一层。
