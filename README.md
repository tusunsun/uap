# 未知接触 · UNIDENTIFIED — A History of Contact

一部约 4 分 48 秒的 UAP / UFO / 外星接触主题宣传片，从公元前 74 年讲到 2025 年。
画面和配乐全部由代码生成：没有用任何素材库，也没有用 AI 视频/音乐模型。

成片：[`output/未知接触_UNIDENTIFIED_1080p.mp4`](output/)（1920×1080，24 fps，H.264 + AAC 立体声）

## 内容结构

| 时间 | 章节 | 事件 |
|---|---|---|
| 0:00 | 冷开场 | 2015 美国海军 “GIMBAL” 红外画面（情景再现）——“它在旋转！” |
| 0:20 | 第一章 · 古老的记载 | 公元前 74 年普鲁塔克记载的“燃烧的酒瓮”；北宋沈括《梦溪笔谈》扬州明珠；1561 年纽伦堡天空“空战” |
| 1:02 | 第二章 · 飞碟时代 | 1896–97 神秘飞艇；1942 洛杉矶之战；1947 阿诺德目击（“飞碟”一词诞生）；1947 罗斯威尔；1952–69 蓝皮书计划；1952 华盛顿雷达事件 |
| 2:05 | 第三章 · 我们也在呼唤 | 1960 奥兹玛计划；1974 阿雷西博信息；1977 旅行者金唱片；1977 Wow! 信号 |
| 2:27 | 第四章 · 目击 | 1957 利维兰、1964 索科罗、1976 德黑兰、1981 中国、1986 日航 1628、1994 阿列尔小学；1980 伦德尔舍姆森林；1989–90 比利时黑三角；1997 凤凰城之光；2010 杭州萧山机场 |
| 3:20 | 第五章 · 解密 | 2004 “Tic Tac”；2015 GOFAST / GIMBAL；2017 纽约时报披露；2020 五角大楼公开视频；2021 ODNI 报告；2023 国会听证会；2024 AARO 报告；2025 公开的“地狱火”视频 |
| 4:02 | 第六章 · 星际来客 | 2017 奥陌陌；2025 3I/ATLAS；2025 火星“潜在生命迹象”；6000+ 颗系外行星 |
| 4:20 | 片名 | 闪回混剪 → 静音 → **未 知 接 触** |

## 关于“真实性”

- 所有画面都是**程序生成的情景再现**：带有红外、报纸、档案、雷达、新闻风格的画面在角标里标注了“情景再现 RECREATION”。片中没有真实影像。
- 事件、日期、引文和数字取自公开的历史文献、政府报告和新闻报道。争议事件保留了官方的另一种说法，例如罗斯威尔事件的“气象气球 / 莫古尔计划”解释，以及 2024 年 AARO 报告称“没有证据表明任何 UAP 目击代表外星技术”。
- 报纸版面、档案正文、航班信息牌上的具体文字是示意道具，不是逐字复刻的原件。唯一例外是罗斯威尔头版标题，它按 1947 年 7 月 8 日的真实标题重排。

## 如何重新生成

依赖 Python 3.10+ 和 4 核以上的 CPU，渲染一遍约 20–30 分钟：

```bash
pip install -r requirements.txt
python3 src/render.py --full --jobs 4   # 画面 -> build/video_noaudio.mp4
python3 src/audio.py                    # 配乐与音效 -> build/score.wav
python3 src/finalize.py                 # 合成 -> output/未知接触_UNIDENTIFIED_1080p.mp4（< 100 MB）
python3 src/finalize.py --hq            # 另外输出高码率母版到 build/
```

单独预览：

```bash
python3 src/render.py --still 35.2 46 239.2   # 导出指定秒数的静帧
python3 src/render.py --range 30 40 --half    # 半分辨率预览片段（无声）
```

## 代码结构

- `src/timeline.py`：剪辑表，决定场景顺序和时长。节拍是 96 BPM，一小节 = 2.5 秒，所有剪辑点都落在小节线上。
- `src/scenes_a.py` … `scenes_d.py`：29 个场景，每个场景实现 `setup` / `render` / `overlay`。
- `src/core.py`：噪声、光晕、泛光、胶片颗粒、字幕与纪录片字幕条系统。
- `src/elements.py`：可复用的元件，包括红外吊舱 HUD、银河、雷达 PPI、报纸版面、纸张与木刻排线。
- `src/audio.py` + `src/dsp.py`：配乐与音效合成器，音色包括 BRAAM 铜管、太鼓、弦乐固定音型、古筝（Karplus-Strong）、特雷门琴、合唱共振峰、钢琴、打字机、盖章、防空警报、雷达、快门、导弹等。音效时间点直接读取场景里的事件时间，保证音画同步。
- `src/finalize.py`：两遍编码，把成片控制在 GitHub 单文件 100 MB 限制内。

## 字体

仓库里的中文字体已按片中实际用到的字符做了子集化，每个文件约 200–300 KB。如果改了画面文字，先运行 `tools/build_fonts.sh` 从 npm 重新构建完整字体。

- 思源宋体 / 思源黑体（Noto Serif SC / Noto Sans SC）：SIL OFL 1.1
- 霞鹜文楷（LXGW WenKai）：SIL OFL 1.1
- JetBrains Mono、IBM Plex Mono、Big Shoulders、Gloock、Libre Baskerville、Nothing You Could Do、Italiana：SIL OFL 1.1

各字体的许可证文本见 `assets/fonts/`。
