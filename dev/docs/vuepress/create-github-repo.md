# 创建github库
因为最终要将vuepress的网站部署到github pages上面，所以第一步首先创建GitHub库，如果不打算部署到github pages上面，可以跳过此步骤。

## 创建public库
对于部署到github pages上面的库要求必须是公开的。创建库完毕后，git到本地就可以进行本地编辑。这里以learning-blogs为例。

## 文档结构
git到刚才创建的库之后，

```
cd learning-blogs
```

然后我们创建这样的目录结构
```
.
├─ dev
└─ docs
└─ README.md
└─ .gitignore
```

dev目录作为我们的主要开发目录，也就是markdown文档以及vuepress代码的存储地方。docs文档这里与vuepress中的docs文档不同， 这个docs文档作为最终部署的文档，我们也称为docs.