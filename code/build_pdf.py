#!/usr/bin/env python3
"""Assemble the complete pilot findings PDF (Chinese text + English figures)."""
import pathlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "data" / "figs"
OUT = ROOT / "Pilot_Findings_and_Outlook.pdf"

F = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(F))
pdfmetrics.registerFontFamily(F, normal=F, bold=F, italic=F, boldItalic=F)

def style(name, **kw):
    base = dict(fontName=F, fontSize=10.5, leading=16, spaceAfter=6)
    base.update(kw); return ParagraphStyle(name, **base)
H1 = style("H1", fontSize=16, leading=20, textColor=colors.HexColor("#0b3d91"), spaceBefore=10, spaceAfter=8)
H2 = style("H2", fontSize=12.5, leading=16, textColor=colors.HexColor("#1f6feb"), spaceBefore=8, spaceAfter=5)
BODY = style("BODY")
BULLET = style("BULLET", leftIndent=14, spaceAfter=3)
SMALL = style("SMALL", fontSize=8.5, leading=11, textColor=colors.HexColor("#57606a"))
TITLE = style("TITLE", fontSize=22, leading=27, textColor=colors.HexColor("#0b3d91"))
SUB = style("SUB", fontSize=12, leading=16, textColor=colors.HexColor("#57606a"))

story = []
def P(t, s=BODY): story.append(Paragraph(t, s))
def B(t): story.append(Paragraph("• " + t, BULLET))
def sp(h=6): story.append(Spacer(1, h))
def fig(name, w_cm=15.5):
    p = FIG / name
    iw, ih = ImageReader(str(p)).getSize()
    w = w_cm * cm; h = w * ih / iw
    story.append(Image(str(p), width=w, height=h)); sp(4)
def cap(t): story.append(Paragraph(t, SMALL)); sp(8)
def tbl(data, widths):
    t = Table(data, colWidths=[w*cm for w in widths])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), F), ("FONTSIZE", (0,0), (-1,-1), 9),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#c9d1d9")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f6f8fa")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (1,0), (-1,-1), "CENTER")]))
    story.append(t); sp(8)

# ---------------- Title ----------------
P('''动态任务迁移图驱动的<br/>Auto-Research 探索系统''', TITLE); sp(4)
P('''试点结果与全量展望 · Pilot Findings &amp; Outlook''', SUB); sp(2)
P('''脑医学影像域 · 886 篇子样 · 2026-06-01 · github.com/vveii14/Auto-Research''', SMALL)
sp(10); fig("../../figure1_overview.png", 16)
cap('''图 0. 系统总览:语料 → LLM 抽取(任务/方法节点 + 有向迁移边）→ 导航（富节点投影 + 结构空洞）→ 探索回流（图生长）→ 治理;底部为「历史回放」评测。''')
story.append(PageBreak())

# ---------------- 1 执行摘要 ----------------
P('''1. 执行摘要''', H1)
P('''本系统把「自动科研方向发现」建模为<b>时序图上的候选迁移边预测</b>:从带时间戳的论文中抽取任务/方法节点与<b>有向、有符号的「A 是否帮助 B」迁移边</b>,构成一张随探索生长、并反过来导航探索的活体图;再用「历史当标签」的时序回放检验它是否越用越准。本报告给出在脑医学影像 886 篇子样上的端到端试点结果。''', BODY)
sp(2); P('''核心结论(诚实标注):''', H2)
B('''<b>已显著</b>:图结构导航预测未来研究方向,<b>显著优于「无图直接问 LLM」、共现图、随机</b>(配对检验 CI 不含 0)。这同时证明收益来自图结构、而非 LLM 记忆泄漏。''')
B('''<b>尚未显著</b>:核心卖点「复利」(growing &gt; frozen)方向一致为正、但在 900 篇规模下 CI 仍跨 0,需扩量定论。''')
B('''<b>真实弱点</b>:当前 method→task 边里 78% 是「纯使用」而非「真迁移」;全图真迁移边仅约 23%。已定位主因为抽取 prompt,可修。''')
B('''<b>判断</b>:idea 的基础设施与「图 &gt; 基线」已坐实(可作安全网论文);「复利」待一次有止损线的扩量赌定。''')
story.append(PageBreak())

# ---------------- 2 idea & workflow ----------------
P('''2. 核心 idea 与工作流''', H1)
P('''<b>定位</b>:第一个把研究关系图从「静态只读底座」变成「<b>随探索动态生长、并反过来导航探索</b>」的活体结构。对手 Intern-Atlas(静态只读)、Deep Ideation(底图固定)、AC/DC(无关系图)。novelty 在于<b>有向迁移语义 + 活体生长 + 图导航 + 复利</b>,对 method→task 与 task→task 两类边平权。''', BODY)
P('''工作流(8 步):''', H2)
for t in ['''定语料范围(可调参数:会议/期刊/年份/规模)''',
          '''无差别读入带时间戳论文''',
          '''LLM 抽取:任务/方法节点 + 有向迁移边(含逐字证据)''',
          '''传递推断:A→B、B→C ⇒ A→C,补稀疏区''',
          '''导航(非随机,核心):富节点投影 + 结构空洞 + 拓扑打分,LLM 仅笼中过滤,输出 top-k 方向''',
          '''探索 + 回流:方向交自动研究,成功/失败均回流为 verified 边,图生长''',
          '''治理:去重 / 冲突消解 / 衰减,防变脏''',
          '''持续接收新论文,图随领域生长''']:
    B(t)
P('''评测(独立注入真值):用 ≤T 年的图预测 T+1 真实出现的迁移边,先打分再回灌,逐年滚动;比较 growing / frozen / 共现图 / 无图 LLM / 随机。<b>历史发表即免费、客观的标签</b>,从而避免循环论证。''', BODY)
story.append(PageBreak())

# ---------------- 3 试点设置 + 语料 ----------------
P('''3. 试点设置与语料''', H1)
P('''来源 Semantic Scholar(仅摘要);过滤 fieldsOfStudy=Computer Science + 脑影像/方法关键词;年份 2015–2024;全量已下 8,000 篇,试点用其中 886 篇子样(≤2020 取 500 篇建图,2021–24 各 100 篇作预测目标/回灌)。抽取用 Azure 托管 Opus,embedding 用 all-MiniLM-L6-v2。''', BODY)
fig("f1_years.png", 14)
cap('''图 1. 语料年份分布,2015→2024 稳定增长,适合时序留出;红线为 T0=2020 的建图/预测分界。''')
story.append(PageBreak())

# ---------------- 4 图 ----------------
P('''4. 构建出的迁移图''', H1)
P('''886 篇 → 实体消解后 <b>2,907 节点(任务 1,064 / 方法 1,843)、1,748 条抽取边</b>,传递推断再补约 518 条。边以 method→task 为主(1,581),task→task 131,method→method 36。''', BODY)
fig("f2_graph.png", 15)
cap('''图 2. 图构成:节点类型(左)与边类型(右,叠加显示抽取 vs 传递推断)。''')
P('''真实 task→task 迁移边样例(who-helps-whom):''', H2)
tbl([['''源任务 → 目标任务''', '''证据(节选)'''],
     ['''年龄预测(结构 MRI) → 阿尔茨海默诊断''', '''sensitive to deviance from normal aging...'''],
     ['''海马分割 → 阿尔茨海默分类''', '''multi-task CNN jointly learning hippocampus...'''],
     ['''虚拟连接组推断 → AD 分类''', '''trained on virtual connectomes can be used...'''],
     ['''PET 分割 ⇄ 去噪 ⇄ 部分容积校正''', '''segmentation can help in denoising and PVC...''']],
    [7.5, 8])
story.append(PageBreak())

# ---------------- 5 结果 ----------------
P('''5. 试点结果''', H1)
P('''5.1 导航准确率(语义匹配,bootstrap 95% CI)''', H2)
P('''同一时序预测任务上,我方(growing/frozen)在各 k 下稳定优于共现图、无图 LLM 与随机。无图 LLM 几乎与随机持平——说明仅靠 LLM(含其记忆的未来)很弱,提升来自图。''', BODY)
fig("f3_precision.png", 15)
cap('''图 3. 各臂 semantic precision@k 与 95% 置信区间(4 个 T0 × 年份汇总)。''')
tbl([['''臂''', '''P@10''', '''P@20''', '''P@50'''],
     ['''growing''', "0.127", "0.093", "0.079"],
     ['''frozen''', "0.091", "0.075", "0.056"],
     ['''cooccur(共现)''', "0.027", "0.057", "0.047"],
     ['''no-graph LLM''', "0.023", "0.016", "0.013"],
     ['''random''', "0.000", "0.000", "0.001"]], [5, 3.5, 3.5, 3.5])

P('''5.2 复利探针:growing vs frozen''', H2)
P('''沿起始年 T0 扫描:底盘小、增长占比大时(T0=2017、2019)growing 明显高于 frozen;底盘大、年增量小时(T0=2020)趋于打平——符合「增量需相对底盘足够大才显现复利」的预期,也正是 900 篇子样在晚期 T0 的人为局限。''', BODY)
fig("f4_t0sweep.png", 14)
cap('''图 4. growing(图逐年生长)与 frozen(图固定于 T0)随 T0 的对比。''')

P('''5.3 决定性配对检验''', H2)
P('''growing − 无图 LLM 在所有 k 下 CI 全部 &gt; 0(<b>显著</b>);growing − frozen 均值恒为正但 CI 仍跨 0(<b>尚不显著</b>)。''', BODY)
fig("f5_paired.png", 14.5)
cap('''图 5. 配对差值与 95% CI;绿色 = CI 不含 0(显著)。''')

P('''5.4 边质量审计:真迁移 vs 纯使用''', H2)
P('''按「真迁移(预训练/表征复用/multi-task/涨点)vs 纯使用(仅把方法用于任务)」分类:method→task 仅 20% 是真迁移、78% 是纯使用;task→task 反而 65% 是真迁移。估算全图真迁移边仅约 23%。''', BODY)
fig("f6_quality.png", 13)
cap('''图 6. 边质量审计(每类 40 条抽样)。当前图大半是「方法-任务使用表」,真迁移是少数。''')

P('''5.5 抽取 prompt 是关键杠杆''', H2)
P('''全文 vs 摘要 A/B 显示:把 task→task 做厚的<b>主杠杆是抽取 prompt,而非读全文</b>。同样的摘要,改用强化迁移 prompt,task→task 占比即从 8% 升到 32%;全文反而稀释比例(加入更多 method→task)。''', BODY)
fig("f7_prompt.png", 13)
cap('''图 7. task→task 边占比:通用 prompt vs 强化迁移 prompt vs +全文。''')
story.append(PageBreak())

# ---------------- 6 结论 ----------------
P('''6. 已证明 / 待解决''', H1)
P('''已证明(显著):''', H2)
B('''图结构导航 &gt; 无图 LLM、&gt; 共现图、&gt;&gt; 随机(配对 CI 不含 0)。''')
B('''收益来自图,而非 LLM 记忆泄漏(无图 LLM 接近随机)。''')
P('''待解决:''', H2)
B('''复利(growing &gt; frozen)方向一致但未显著——需扩量(更多年份/数据点收紧 CI)。''')
B('''边质量:真迁移仅约 23%,需「区分迁移/使用」的重抽提纯。''')
B('''规模与噪声:900 篇子样、仅摘要、单 embedding,数字偏小且抖。''')

# ---------------- 7 展望 ----------------
P('''7. 全量展望与路线''', H1)
P('''最关键三步(其余为工程放大):''', H2)
B('''<b>提纯重抽</b>:用「区分真迁移 vs 纯使用、两类源平权」的 prompt 重抽,先低成本验证图质量与「图 &gt; 基线」在提纯后是否依然成立。''')
B('''<b>扩量定复利</b>:扩到约 3,000 篇并加多起始年 T0,把 growing − frozen 的 CI 推离 0;读全文进一步加厚真迁移。''')
B('''<b>完全体</b>:本地 72B 抽取(可复现、防泄漏)+ Neo4j/向量索引 + 治理常开 + 多域(脑→视网膜→心脏)+ 接真实 auto-research 闭环 demo。''')
P('''止损线:扩量后复利仍不显著,则收敛为「结构导航 &gt; 无图 LLM/共现/随机」的弱化版论文(已显著)。''', BODY)
sp(6)
P('''附:实现进度''', H2)
tbl([['''阶段''', '''内容''', '''状态'''],
     ["P0", '''取数 + 图骨架''', '''完成(8000 篇)'''],
     ["P1", '''抽取验证''', '''完成'''],
     ["P2", '''建图 + 消解 + 推断''', '''完成(886 篇)'''],
     ["P3", '''图导航''', '''完成'''],
     ["P4–P6", '''时序评测 + 置信区间''', '''完成'''],
     ["P7–P8", '''全文 A/B + 边质量审计''', '''完成'''],
     ["—", '''提纯重抽 / 扩量 / 完全体''', '''待进行''']], [2.5, 8.5, 4.5])

doc = SimpleDocTemplate(str(OUT), pagesize=A4, topMargin=1.8*cm, bottomMargin=1.8*cm,
                        leftMargin=2*cm, rightMargin=2*cm, title="Pilot Findings and Outlook")
doc.build(story)
import os; print("PDF written:", OUT, "|", os.path.getsize(OUT), "bytes")
