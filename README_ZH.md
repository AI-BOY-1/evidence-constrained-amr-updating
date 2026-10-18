# Evidence-Constrained AMR Updating Controller

**作者：Xingran Guo**

本仓库提供证据约束的抗菌药物耐药（AMR）更新控制器及其研究复现脚本。它不包含 FASTA、患者级 AST、ResFinder 数据库、MALDI 权重、照片或已生成结果。使用前请阅读 `docs/SCIENTIFIC_SCOPE.md`。`src/ncs_amr/` 是可配置入口；`historical_frozen/` 用于复核与报告分析绑定的原始代码快照。

## 本地验证

```powershell
python -m pip install -r requirements.txt
python scripts/run_demo.py --out demo_output
python -m unittest discover -s tests -v
```

## 上传前检查

1. 本项目代码和文档采用 MIT License；作者为 Xingran Guo。机构信息未在本代码发布中指定。外部数据遵循各自许可。
2. 不上传患者级或受限数据，不上传 `D:\NCS_data` 内容。
3. 不把 demo 输出当论文实验结果。
4. 如果修改方法，改变版本号并重新计算受影响结果，不能混用旧哈希。
5. 软件引用信息及 GitHub 使用说明见英文 `README.md`。


论文分析、验证、制图脚本和本地依赖模块位于 `research/`；数据根目录通过 `NCS_DATA_ROOT` 配置。完整队列运行说明见 `docs/REPRODUCE_STUDY.md`。核心计算函数与历史快照逐函数核对；可移植入口增加了输入验证。代码发布不代表完整数据发布，也不代表完整重跑了全部科学实验。
