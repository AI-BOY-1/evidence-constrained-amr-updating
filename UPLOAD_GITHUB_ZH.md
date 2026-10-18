# 上传 GitHub

1. 解压 `evidence_constrained_amr_updating_controller_0.1.0.zip`，或克隆本仓库。
2. 将解压后这一层目录内的 README.md、src、research、tests、docs、.github、依赖文件和LICENSE上传到仓库根目录，不要只上传ZIP。
3. 网页上传时 `.github` 和 `.gitignore` 可能被文件选择界面隐藏；推荐用 Git 命令完整提交。
4. 按README安装核心依赖，执行 `python scripts/verify_release.py`、`python -m unittest discover -s tests -v` 和合成demo。
5. 真实数据放仓库外，通过NCS_DATA_ROOT配置。完整队列需要另行获取的数据资产；不能将代码包自检称全论文重现。

```bash
git init
git add .
git commit -m "Release evidence-constrained AMR updating research code"
git branch -M main
git remote add origin <你的GitHub仓库URL>
git push -u origin main
```

仓库名称：`evidence-constrained-amr-updating`。作者：Xingran Guo。代码和文档采用 MIT License；外部数据和权重不在包内，继续遵循原许可。论文 DOI 或其他归档标识应在正式发布后补充。
