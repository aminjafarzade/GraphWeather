---
name: implementation-docs
description: Generate interactive HTML documentation for branch implementation — SVG computational flow diagrams, right-side code panel with click navigation, numeric trace examples. TRIGGER when user says "구현 문서", "implementation docs", "구현 다이어그램", "파이프라인 문서", "branch docs", "implementation HTML".
allowed-tools: Bash, Read, Grep, Glob, Agent, Artifact, Write, Edit, Workflow
---

# Implementation Docs Generator

Generate a self-contained interactive HTML artifact documenting **all** code implemented on the current branch.

## Arguments

$ARGUMENTS — scope hint. If empty, auto-detect from `git diff master --name-only`.

## Core Value

**This document's purpose is: DIAGRAM ↔ CODE MAPPING.**
A reader looks at a visual diagram, clicks a node, and sees the exact source code highlighted. That is the entire point. Everything else (trace, formulas, quant tables) is supporting material.

If a tab has no SVG diagram, it has failed its purpose. If clicking a diagram node doesn't open highlighted code, it has failed its purpose.

## Principles

1. **Every tab has an SVG diagram.** No exceptions. No text-only tabs. The diagram IS the tab.
2. **Click any node → see its code highlighted.** This is the #1 feature. Diagram nodes open the right code panel scrolled to the matching stage-marked lines.
3. **Nothing omitted.** Every new/changed layer, module, utility on the branch gets a tab.
4. **One tab = one functional unit.** Never split by information type.
5. **Full code in templates, zero code in tab body.** Tab body = diagram + explanation + trace. Code = right panel only.
6. **Verify visually with Playwright.** Screenshot every tab. Fix before presenting.

## Multi-Agent Orchestration

Execute this skill using **4 sequential phases**, each with specialized agents:

### Phase 1: Scope & Structure (architect agent)

Spawn a **read-only architect agent** that:

1. Runs `git diff master --name-only` and reads ALL changed source files.
2. Uses codegraph to map the call graph of changed code.
3. For each functional unit, answers **WHY and WHERE**:
   - **WHY**: What problem does this solve? What motivated building it? (read commit messages, comments, PR descriptions, related code context)
   - **WHERE**: Where is this used? Which modules/layers consume it? Which HW targets support it? (use `codegraph_callers` to find all call sites)
   - This context goes into each tab's "Context box" at the top.
4. Produces a **tab plan** as structured output:

```
Tab Plan:
1. "VpuLayer" — predict pipeline (8 stages), helpers (gather, pad, cycle, etc.)
2. "SoftmaxType3" — VPU module (5 sublayers), merged sub+exp optimization
3. "LayerNormType4/5" — VPU module, prescale vs no-prescale variants
4. "SamplingLayer" — block gather operation, overflow modes
5. "GridSampleModule" — 7-sublayer decomposition, bilinear interpolation
6. "GridScale" — coordinate transform math
7. "IntExtract" — floor via conv trick
...
```

**Completeness check:** every file in `git diff` output must map to at least one tab. If a file is not covered, add a tab or merge it into an existing one.

Each tab entry specifies:
- Tab name (= functional unit name)
- Content blocks needed: `[diagram, formula, trace, quant-table, params-table]`
- Key source files and functions to read
- Diagram type: `pipeline | composition | operation | formula`

### Phase 2: Content Generation (parallel builder agents)

Spawn **one agent per tab** (in parallel) using `Agent` tool. Each builder agent:

1. Reads the full source code for its assigned functional unit.
2. Reads ALL called helper functions (follow the call graph — do not stop at the first level).
3. Generates one `<div class="tab-panel">` with ONLY visual elements (**NO raw source code in the tab body**):
   - **Section 1: Context + Overview** — WHY/WHERE box + 2-3 sentence summary.
   - **Section 2: SVG Diagram** — THE MOST IMPORTANT ELEMENT. Must be an inline `<svg>` (not an image). Follow the concrete templates below. Every node has `class="click" onclick="openCode('template-id','STAGE LABEL')"`.

   **Minimum viable SVG for a pipeline tab (copy this pattern):**
   ```svg
   <svg viewBox="0 0 800 600">
     <defs><marker id="ah" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
       <path d="M0,0 L7,2.5 L0,5" fill="var(--td)"/></marker></defs>

     <!-- 3D tensor input block -->
     <g transform="translate(50,30)">
       <polygon points="0,30 70,30 70,55 0,55" fill="var(--acc)" opacity=".55"/>
       <polygon points="0,30 15,18 85,18 70,30" fill="var(--acc)" opacity=".38"/>
       <polygon points="70,30 85,18 85,43 70,55" fill="var(--acc)" opacity=".3"/>
       <text x="10" y="47" fill="#fff" font-size="9">B×C×H×W</text>
     </g>

     <!-- Pipeline stage node (clickable) -->
     <g class="click" onclick="openCode('code-main','STEP 1')">
       <rect x="50" y="100" width="180" height="45" rx="6"
             fill="var(--card)" stroke="var(--grn)" stroke-width="1.5"/>
       <rect x="55" y="105" width="30" height="13" fill="var(--grn)" opacity=".15" rx="3"/>
       <text x="60" y="114" font-size="8" fill="var(--grn)" font-weight="600">VPU</text>
       <text x="90" y="114" font-size="10" fill="var(--t)" font-weight="600">preProcess</text>
       <text x="90" y="130" font-size="8" fill="var(--td)">scale·x + bias</text>
     </g>

     <!-- Arrow between stages -->
     <path d="M140,145 L140,170" stroke="var(--td)" stroke-width="1.5"
           fill="none" marker-end="url(#ah)"/>

     <!-- Next stage... repeat pattern -->
   </svg>
   ```

   **Minimum viable SVG for a composition tab (sublayer chain):**
   ```svg
   <g class="click" onclick="openCode('code-init','SUBLAYER_NAME')">
     <rect x="20" y="Y" width="220" height="26" rx="6"
           fill="var(--card)" stroke="var(--grn)" stroke-width="1.5"/>
     <rect x="24" y="Y+3" width="24" height="12" fill="var(--grn)" opacity=".15" rx="3"/>
     <text x="28" y="Y+12" font-size="7.5" fill="var(--grn)" font-weight="600">VPU</text>
     <text x="52" y="Y+12" font-size="10" fill="var(--t)" font-weight="600">layer_name</text>
     <text x="120" y="Y+12" font-size="8" fill="var(--td)">description</text>
   </g>
   <!-- Arrow down -->
   <path d="M130,Y+26 L130,Y+34" stroke="var(--td)" stroke-width="1.5"
         fill="none" marker-end="url(#ah)"/>
   ```

   Builder agents: copy these patterns literally and adapt coordinates/colors. Do NOT skip the SVG and replace with text descriptions.
   - **Section 3: Code buttons** — STAGES | FUNCTIONS button row.
   - **Section 4: Explanation** — Short text, math formulas (`.eq-block`). No code.
   - **Section 5: Numeric Trace** — Table (`.trace`). NOT code.
   - **Section 6: Config/Quant** — Parameter table (if applicable).

   **CRITICAL — Where code goes:**
   - Source code goes ONLY in `<template>` elements (hidden, rendered in the right code panel on click).
   - The tab body has ZERO `<pre>` blocks with C++/Python source. Zero.
   - A tab body should be ~800-1500px tall max. If 5000px+ → code leaked into body → fix it.
   - Reader sees full diagram without scrolling, then trace/config below. Code → right panel on click.
4. Generates all `<template>` elements for code panel, with:
   - **THE COMPLETE FUNCTION BODY** — copy the raw source code verbatim. Do not summarize, elide with `...`, omit error handling, or skip "boring" parts. If a function is 200 lines, the template is 200 lines. The code panel scrolls.
   - Every function referenced in the diagram gets its own template. Every helper function called by those functions also gets a template.
   - Korean docstring at top (what it does, key params, non-obvious decisions)
   - Stage markers (`<span class="stg">`) matching diagram colors
   - Line numbers from source (actual file line numbers)
   - `data-ref` on every function call that has a corresponding template

   **Code completeness checklist for each tab:**
   - Main function: full body ✓
   - Each sub-function called by main: full body ✓
   - Each helper/utility called by sub-functions: full body ✓
   - If a function is too long for one template, split into STAGES within the same template — never truncate

### Phase 3: Assembly (main thread)

The main thread:

1. Collects all tab panels and code templates from builder agents.
2. Wraps in the standard HTML skeleton (CSS theme, tab bar, code panel, JS).
3. Writes to scratchpad file.
4. Publishes via Artifact.

### Phase 4: Visual QA (reviewer agent)

Spawn a **reviewer agent** that:

1. Uses Playwright to screenshot every tab (both with and without code panel open).

2. **Content completeness check** (parse the HTML, not just visual):
   - Every tab MUST have an `<svg>` diagram. No exceptions. If missing → FAIL.
   - Every SVG diagram MUST have at least one `class="click"` node. If no clickable nodes → FAIL.
   - Every `class="click"` node must have an `onclick="openCode(...)"` attribute. Check all exist.
   - Every `onclick` target template (`code-{id}`) must exist as a `<template>` element. If missing → FAIL.
   - Every code template that represents a main function must have `<span class="stg">` stage markers. If none → FAIL.
   - Pipeline diagrams must have `<polygon>` elements (3D tensor blocks). If only text shapes → FAIL.
   - Math formulas must use `<div class="eq-block">`, NOT inline `<code>` or `<pre>`. If formulas in code blocks → FAIL.
   - Tab body must NOT contain `<pre>` with source code. All code in `<template>` only. If a `.tab-panel` has a `<pre>` longer than 10 lines → FAIL (code leaked into body).
   - Tab panel height check: measure each tab panel's scrollHeight via Playwright. If any exceeds 3000px → likely code in body → FAIL.

3. **Visual checks** (via Playwright screenshots):
   - Text overlap / clipping (badge text wider than badge rect)
   - Missing right faces on 3D tensors (must have 3 faces)
   - Polygon opacity < .28 (invisible against background)
   - SVG content cut off at bottom (viewBox too small)
   - Code panel horizontal overflow (long lines cut off)
   - Nodes too large (diagram doesn't fit in one viewport — check if SVG height > 800px)

4. **Interaction checks** (via Playwright click actions):
   - Click each `class="click"` node → verify code panel opens (`.code-open` class appears on `.content`)
   - Click each `.cbtn.stg-btn` → verify code panel scrolls (a `.stg` element becomes visible)
   - If any click does nothing → FAIL with the specific node/button that's broken.

5. Reports ALL issues as a checklist. Main thread fixes them. Re-run checks until all pass.

## Tab Organization Rules

**One tab per functional unit.** A functional unit is:
- A Layer class (e.g., VpuLayer, SamplingLayer)
- A Module class (e.g., SoftmaxLayer Type3, GridSampleLayer)
- A sub-component with self-contained logic (e.g., GridScale, IntExtract)
- A scheme/algorithm class (e.g., AdaptiveLayerNormType)

**Within each tab, content flows top-to-bottom:**

```
┌─ Context box (WHY & WHERE) ──────────────────────────────────┐
│  Why this was built, what problem it solves, where it's used │
│  e.g., "VPU layer simulates the Vector Processing Unit HW   │
│  pipeline. Used as a building block in GridSample, Softmax,  │
│  LayerNorm modules on Aries2+ hardware."                     │
│                                                              │
├─ Overview box (WHAT, 2-3 sentences) ─────────────────────────┤
│                                                              │
├─ SVG Diagram ────────────────────────────────────────────────┤
│  (pipeline / composition / operation diagram)                │
│  Every node is clickable → opens code panel                  │
│                                                              │
├─ Code Buttons ───────────────────────────────────────────────┤
│  STAGES │ btn btn btn │ FUNCTIONS │ btn btn                  │
│                                                              │
├─ Explanation (if needed) ────────────────────────────────────┤
│  Math formulas (.eq-block), design decisions, key concepts   │
│                                                              │
├─ Numeric Trace ──────────────────────────────────────────────┤
│  <table class="trace"> with concrete values per stage        │
│                                                              │
├─ Quant/Config Table (if applicable) ─────────────────────────┤
│  sublayer types, quantDep, actScaleMin, output dtype         │
└──────────────────────────────────────────────────────────────┘
```

**NEVER create these as standalone tabs:**
- "Pipeline" / "Overview" — the module's own tab IS the overview
- "Quant Config" — embed as section within each component's tab
- "Math" / "Formulas" — belongs inside the component that does the math

## Completeness Guarantee

After the architect agent produces the tab plan, verify:

```
for each file in git diff --name-only:
    assert file is referenced by at least one tab
```

If any file is orphaned, either:
- Add it to an existing tab's code collection
- Create a new tab for it

Common things that get missed:
- Predict pipeline layers (VpuLayer, etc.) — these are NEW layers, not just modules
- Helper functions in anonymous namespaces — gather, pad, cycle, etc.
- Enum/config changes that affect behavior
- Test files (skip these — no tab needed for tests)

## HTML Structure Reference

```
<div class="page">
  <div class="tab-bar">...</div>
  <div class="content" id="content">
    <div class="main">
      <div class="tab-panel" id="tab-0">...</div>
      ...
    </div>
    <div id="code-panel">
      <div class="cp-resize"/>
      <div class="cp-header">... ← back | title | file | ✕</div>
      <div class="cp-body"><pre id="cp-code"/></div>
    </div>
  </div>
</div>
<template id="code-{id}" data-title="..." data-file="...">...</template>
<script>/* tab switching, code panel, history, drag resize */</script>
```

## Visual Rules

### Diagram Sizing — CRITICAL

Diagrams must be **compact enough to see the full flow on one screen** without scrolling. Common mistake: making nodes too large so only 2-3 fit on screen.

| Diagram type | Target | Node size | Font |
|-------------|--------|-----------|------|
| Pipeline (≤8 stages) | Fits in 600-800px height | 40-55px tall, 120-180px wide | 10-12px name, 8-9px description |
| Composition (≤12 sublayers) | Fits in 600px height | 22-30px tall, 180-220px wide | 10px name, 8px description |
| Operation (2-3 elements) | Fits in 350px height | 60-80px tall | 11px name |

If a pipeline has 8 stages, each node should be ~55px tall with ~15px gap = ~560px total. NOT 120px tall nodes that push the diagram to 1500px.

### 3D Tensor Blocks — REQUIRED for shape transforms

Pipeline diagrams that show tensor shape changes MUST use isometric 3D blocks, not just text labels. This is the primary way to visualize what each stage does to the data.

```svg
<!-- 3-face isometric tensor block -->
<polygon fill="var(--color)" opacity=".55"/>  <!-- front: shape label inside -->
<polygon fill="var(--color)" opacity=".38"/>  <!-- top face -->
<polygon fill="var(--color)" opacity=".3"/>   <!-- right face, NEVER < .28 -->
```

Use 3D blocks at pipeline input/output and at key shape-change points. Between blocks, show the operation that transforms the shape (arrow + label).

Example pattern for a pipeline stage:
```
[3D block B×C×H×W] --movedim--> [3D block B×H×W×C] --gather--> [3D block oH×oW×sH×sW×C]
```

### Pipeline Diagrams — Follow the Code

For predict/forward pipelines, the diagram should mirror the code's execution order:

1. Show the **main function** (e.g., predictImpl) as the backbone.
2. Each stage is a step in that function — numbered, with the function call visible.
3. Between stages, show tensor shape as 3D blocks with dimension labels.
4. The reader should be able to follow the diagram AND the code in parallel.

Do NOT just list stage names vertically. Show the data transformation at each step.

### Diagram ↔ Code Linking — CRITICAL

Every clickable node in a diagram must do TWO things:

1. **Open the main function** (e.g., `predictImpl`, `initModuleType1`) in the code panel.
2. **Scroll to AND highlight the exact lines** where that stage/sublayer is defined.

Implementation:

```html
<!-- Diagram node -->
<g class="click" onclick="openCode('main-template-id', 'STAGE LABEL')">

<!-- In the code template, wrap the corresponding lines -->
<span class="stg" style="--c:var(--grn)">
  <span class="stg-label">STAGE LABEL</span>
  <span class="ln-n">1249</span>    auto opOut = vpuOperationInt(pre0, pre1);
</span>
```

The `openCode(templateId, scrollTarget)` function:
- Opens the code template in the right panel
- Finds the `stg-label` containing `scrollTarget` text
- Scrolls to it with `scrollIntoView`
- Applies a pulse animation (colored background flash)

This means the main function template MUST have stage markers for EVERY diagram node. If the diagram has 8 stages, the code template has 8 colored `stg` blocks.

Stage button row should also use the same mechanism:
```html
<button class="cbtn stg-btn" style="--c:var(--grn)" 
        onclick="openCode('main-template', 'STEP 2')">Operation</button>
```

Clicking a stage button scrolls to the stage in code. Clicking a FUNCTION button opens that function's own template directly.

### Other Visual Elements

| Element | Rule |
|---------|------|
| Flow node | `class="click" onclick="openCode(...)"`. Badge + name + description |
| Arrows | Solid for data flow, dashed for bypass. Triangle marker. |
| Stage markers | Colored left border matching diagram. Line numbers from source. |
| `data-ref` | Every function call with a template gets dotted-underline click link |
| ViewBox | Tight. No inline max-width. CSS handles responsive sizing. |
| Descriptions | Below tensor row, not on arrows |
| Badge text | Must fit within badge rect width |
| Code panel | Right side, drag-resizable, `overflow-x:auto` on pre |

## Anti-Patterns

- Never abbreviate code
- Never split a functional unit across tabs by information type
- Never set polygon opacity < .28
- Never use 2-face tensors
- Never put description text on arrows
- Never create "Pipeline" or "Quant Config" as standalone tabs
- Never skip Playwright QA
- Never present to user without fixing visual issues first
